import os
import sys
sys.path.append('/your/path/to/Grounded-Segment-Anything/')
sys.path.append('/your/path/to/concept-graphs/conceptgraph')
sys.path.append('/your/work/directory/')
import cv2
import numpy as np
import torch
import math
import dataclasses
import omegaconf
import supervision as sv
from PIL import Image
from sklearn.cluster import DBSCAN  
from collections import Counter 
from omegaconf import DictConfig
from pathlib import PosixPath, Path
from supervision.draw.color import Color, ColorPalette
from utils.utils_scenegraph.slam_classes import MapObjectList
from utils.utils_scenegraph.utils import filter_objects, gobs_to_detection_list
from utils.utils_scenegraph.mapping import compute_spatial_similarities, merge_detections_to_objects

from grounded_sam_demo import load_image, load_model, get_grounding_output
import GroundingDINO.groundingdino.datasets.transforms as T
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

from model_server.llm_client import LLM_Client
from model_server.vlm_client import VLM_Client


ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]


class RoomNode():
    def __init__(self, caption):
        self.caption = caption
        self.exploration_level = 0
        self.nodes = set()
        self.group_nodes = []


class GroupNode():
    def __init__(self, caption=''):
        self.caption = caption
        self.exploration_level = 0
        self.corr_score = 0
        self.center = None
        self.center_node = None
        self.nodes = []
        self.edges = set()
    
    def __lt__(self, other):
        return self.corr_score < other.corr_score
    
    def get_graph(self):
        self.center = np.array([node.center for node in self.nodes]).mean(axis=0)
        min_distance = np.inf
        for node in self.nodes:
            distance = np.linalg.norm(np.array(node.center) - np.array(self.center))
            if distance < min_distance:
                min_distance = distance
                self.center_node = node
            self.edges.update(node.edges)
        self.caption = self.graph_to_text(self.nodes, self.edges)

    def graph_to_text(self, nodes, edges):
        nodes_text = ', '.join([node.caption for node in nodes])
        edges_text = ', '.join([f"{edge.node1.caption} {edge.relation} {edge.node2.caption}" for edge in edges])
        return f"Nodes: {nodes_text}. Edges: {edges_text}."


class ObjectNode():
    def __init__(self):
        self.is_new_node = True
        self.caption = None
        self.object = None
        self.reason = None
        self.center = None
        self.room_node = None
        self.exploration_level = 0
        self.distance = 2
        self.score = 0.5
        self.edges = set()

    def __lt__(self, other):
        return self.score < other.score

    def add_edge(self, edge):
        self.edges.add(edge)

    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def set_caption(self, new_caption):
        for edge in list(self.edges):
            edge.delete()
        self.is_new_node = True
        self.caption = new_caption
        self.reason = None
        self.distance = 2
        self.score = 0.5
        self.exploration_level = 0
        self.edges.clear()
    
    def set_object(self, object):
        self.object = object
        self.object['node'] = self
    
    def set_center(self, center):
        self.center = center


class Edge():
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        node1.add_edge(self)
        node2.add_edge(self)
        self.relation = None

    def set_relation(self, relation):
        self.relation = relation

    def delete(self):
        self.node1.remove_edge(self)
        self.node2.remove_edge(self)

    def text(self):
        text = '({}, {}, {})'.format(self.node1.caption, self.node2.caption, self.relation)
        return text


class SceneGraph():
    def __init__(self, map_resolution, map_size_cm, map_size, camera_matrix, is_navigation=True) -> None:
        self.map_resolution = map_resolution
        self.map_size_cm = map_size_cm
        self.map_size = map_size
        full_w, full_h = self.map_size, self.map_size
        self.full_w = full_w
        self.full_h = full_h
        self.visited = torch.zeros(full_w, full_h).float().cpu().numpy()
        self.num_of_goal = torch.zeros(full_w, full_h).int()
        self.camera_matrix = camera_matrix
        self.GSA_PATH = os.environ["GSA_PATH"]
        self.SAM_ENCODER_VERSION = "vit_h"
        self.SAM_CHECKPOINT_PATH = os.path.join(self.GSA_PATH, "./sam_vit_h_4b8939.pth")
        self.sam_variant = 'sam'
        self.sam_variant = 'groundedsam'
        self.device = 'cuda'
        self.classes = ['item']
        self.BG_CLASSES = ["wall", "floor", "ceiling"]
        self.rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.edge_text = ''
        self.edge_list = []
        self.group_nodes = []
        self.init_room_nodes()
        self.reason_visualization = ''
        self.is_navigation = is_navigation
        self.reasoning = 'both'
        self.PSL_infer = 'one_hot'
        self.set_cfg()
        
        self.groundingdino_config_file = '/home/user001/yh/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.groundingdino_checkpoint = '/home/user001/yh/Grounded-Segment-Anything/groundingdino_swint_ogc.pth'
        self.sam_version = 'vit_h'
        self.sam_checkpoint = '/home/user001/yh/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
        self.segment2d_results = []
        self.max_detections_per_object = 10
        self.threshold_list = {'bathtub': 3, 'bed': 3, 'cabinet': 2, 'chair': 1, 'chest_of_drawers': 3, 'clothes': 2, 'counter': 1, 'cushion': 3, 'fireplace': 3, 'gym_equipment': 2, 'picture': 3, 'plant': 3, 'seating': 0, 'shower': 2, 'sink': 2, 'sofa': 2, 'stool': 2, 'table': 1, 'toilet': 3, 'towel': 2, 'tv_monitor': 0}
        self.found_goal_times_threshold = 1
        self.N_max = 10
        self.node_space = 'table. tv. chair. cabinet. sofa. bed. windows. kitchen. bedroom. living room. mirror. plant. curtain. painting. picture'
        self.prompt_edge_proposal = '''
Provide the most possible single spatial relationship for each of the following object pairs. Answer with only one relationship per pair, and separate each answer with a newline character.
Examples:
Input:
Object pair(s):
(cabinet, chair)
Output:
next to
Input:
Object pair(s):
(table, lamp)
(bed, nightstand)
Output:
on
next to
Object pair(s):
        '''
        self.prompt_discriminate_relation = 'In the image, do {} and {} satisfy the relationship of {}? Answer "yes" or "no".'
        self.prompt_room_predict = 'Which room is the most likely to have the [{}] in: [{}]. Only answer the room.'
        self.prompt_graph_corr_0 = 'What is the probability of A and B appearing together. [A:{}], [B:{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_graph_corr_1 = 'What else do you need to know to determine the probability of A and B appearing together? [A:{}], [B:{}]. Please output a short question (output only one sentence with no additional text).'
        self.prompt_graph_corr_2 = 'Here is the objects and relationships near A: [{}] You answer the following question with a short sentence based on this information. Question: {}'
        self.prompt_graph_corr_3 = 'The probability of A and B appearing together is about {}. Based on the dialog: [{}], re-determine the probability of A and B appearing together. A:[{}], B:[{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.mask_generator = self.get_sam_mask_generator(self.sam_variant, self.device)
        self.llm = LLM_Client()
        self.vlm = VLM_Client()

    def set_cfg(self):
        cfg = {'dataset_root': PosixPath('/your/path/to/Replica'), 'dataset_config': PosixPath('/your/path/to/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica.yaml'), 'scene_id': 'room0', 'start': 0, 'end': -1, 'stride': 5, 'image_height': 680, 'image_width': 1200, 'gsa_variant': 'none', 'detection_folder_name': 'gsa_detections_${gsa_variant}', 'det_vis_folder_name': 'gsa_vis_${gsa_variant}', 'color_file_name': 'gsa_classes_${gsa_variant}', 'device': 'cuda', 'use_iou': True, 'spatial_sim_type': 'overlap', 'phys_bias': 0.0, 'match_method': 'sim_sum', 'semantic_threshold': 0.5, 'physical_threshold': 0.5, 'sim_threshold': 1.2, 'use_contain_number': False, 'contain_area_thresh': 0.95, 'contain_mismatch_penalty': 0.5, 'mask_area_threshold': 25, 'mask_conf_threshold': 0.95, 'max_bbox_area_ratio': 0.5, 'skip_bg': True, 'min_points_threshold': 16, 'downsample_voxel_size': 0.025, 'dbscan_remove_noise': True, 'dbscan_eps': 0.1, 'dbscan_min_points': 10, 'obj_min_points': 0, 'obj_min_detections': 3, 'merge_overlap_thresh': 0.7, 'merge_visual_sim_thresh': 0.8, 'merge_text_sim_thresh': 0.8, 'denoise_interval': 20, 'filter_interval': -1, 'merge_interval': 20, 'save_pcd': True, 'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub', 'vis_render': False, 'debug_render': False, 'class_agnostic': True, 'save_objects_all_frames': True, 'render_camera_path': 'replica_room0.json', 'max_num_points': 512}
        cfg = DictConfig(cfg)
        if self.is_navigation:
            cfg.sim_threshold = 0.8
            cfg.sim_threshold_spatial = 0.01
        self.cfg = cfg

    def set_agent(self, agent):
        self.agent = agent

    def set_obj_goal(self, obj_goal):
        self.obj_goal = obj_goal

    def set_navigate_steps(self, navigate_steps):
        self.navigate_steps = navigate_steps

    def set_room_map(self, room_map):
        self.room_map = room_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map
    
    def set_observations(self, observations):
        self.observations = observations
        self.image_rgb = observations['rgb'].copy()
        self.image_depth = observations['depth'].copy()
        self.pose_matrix = self.get_pose_matrix()

    def set_frontier_map(self, frontier_map):
        self.frontier_map = frontier_map

    def set_full_map(self, full_map):
        self.full_map = full_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.edges)
        edges = list(edges)
        return edges

    def get_seg_xyxy(self):
        return self.seg_xyxy

    def get_seg_caption(self):
        return self.seg_caption

    def init_room_nodes(self):
        room_nodes = []
        for caption in self.rooms:
            room_node = RoomNode(caption)
            room_nodes.append(room_node)
        self.room_nodes = room_nodes

    def get_sam_mask_generator(self, variant:str, device) -> SamAutomaticMaskGenerator:
        if variant == "sam":
            sam = sam_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.SAM_CHECKPOINT_PATH)
            sam.to(device)
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=12,
                points_per_batch=144,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=0,
                min_mask_region_area=100,
            )
            return mask_generator
        elif variant == "fastsam":
            raise NotImplementedError
            # from ultralytics import YOLO
            # from FastSAM.tools import *
            # FASTSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/FastSAM-x.pt")
            # model = YOLO(args.model_path)
            # return model
        elif variant == "groundedsam":
            model = load_model(self.groundingdino_config_file, self.groundingdino_checkpoint, device=device)
            predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(device))
            return model, predictor
        else:
            raise NotImplementedError
    
    def get_sam_segmentation_dense(
        self, variant:str, model, image: np.ndarray
    ) -> tuple:
        '''
        The SAM based on automatic mask generation, without bbox prompting
        
        Args:
            model: The mask generator or the YOLO model
            image: )H, W, 3), in RGB color space, in range [0, 255]
            
        Returns:
            mask: (N, H, W)
            xyxy: (N, 4)
            conf: (N,)
        '''
        if variant == "sam":
            results = model.generate(image)  # type(results) == list
            mask = []
            xyxy = []
            conf = []
            for r in results:  # type(r) == dict
                mask.append(r["segmentation"])  # type(r["segmentation"]) == np.ndarray, r["segmentation"] == [480, 640]
                r_xyxy = r["bbox"].copy()  # type(r["bbox"]) == list, [x, y, h, w]
                # Convert from xyhw format to xyxy format
                r_xyxy[2] += r_xyxy[0]
                r_xyxy[3] += r_xyxy[1]
                xyxy.append(r_xyxy)
                conf.append(r["predicted_iou"])  # type(r["predicted_iou"]) == float
            mask = np.array(mask)
            xyxy = np.array(xyxy)
            conf = np.array(conf)
            return mask, xyxy, conf
        elif variant == "fastsam":
            # The arguments are directly copied from the GSA repo
            results = model(
                image,
                imgsz=1024,
                device="cuda",
                retina_masks=True,
                iou=0.9,
                conf=0.4,
                max_det=100,
            )
            raise NotImplementedError
        elif variant == "groundedsam":
            groundingdino = model[0]
            sam_predictor = model[1]
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image_resized, _ = transform(Image.fromarray(image), None)  # 3, h, w
            boxes_filt, caption = get_grounding_output(groundingdino, image_resized, caption=self.node_space, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=self.device)
            if len(caption) == 0:
                return None, None, None, None
            sam_predictor.set_image(image)

            # size = image_pil.size
            H, W = image.shape[0], image.shape[1]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

            mask, conf, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(self.device),
                multimask_output = False,
            )
            mask, xyxy, conf = mask.squeeze(1).cpu().numpy(), boxes_filt.squeeze(1).numpy(), conf.squeeze(1).cpu().numpy()
            return mask, xyxy, conf, caption
        else:
            raise NotImplementedError

    def compute_clip_features(self, image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
        backup_image = image.copy()
        
        image = Image.fromarray(image)
        
        # padding = args.clip_padding  # Adjust the padding amount as needed
        padding = 20  # Adjust the padding amount as needed
        
        image_crops = []
        image_feats = []
        text_feats = []

        
        for idx in range(len(detections.xyxy)):
            # Get the crop of the mask with padding
            x_min, y_min, x_max, y_max = detections.xyxy[idx]

            # Check and adjust padding to avoid going beyond the image borders
            image_width, image_height = image.size
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            # Apply the adjusted padding
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            
            # Get the preprocessed image for clip from the crop 
            preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

            crop_feat = clip_model.encode_image(preprocessed_image)
            crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
            
            class_id = detections.class_id[idx]
            tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
            text_feat = clip_model.encode_text(tokenized_text)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            
            crop_feat = crop_feat.cpu().numpy()
            text_feat = text_feat.cpu().numpy()

            image_crops.append(cropped_image)
            image_feats.append(crop_feat)
            text_feats.append(text_feat)
            
        # turn the list of feats into np matrices
        image_feats = np.concatenate(image_feats, axis=0)
        text_feats = np.concatenate(text_feats, axis=0)

        return image_crops, image_feats, text_feats

    def vis_result_fast(
        self,
        image: np.ndarray, 
        detections: sv.Detections, 
        classes: list, 
        color = ColorPalette.default(), 
        instance_random_color: bool = False,
        draw_bbox: bool = True,
    ) -> np.ndarray:
        '''
        Annotate the image with the detection results. 
        This is fast but of the same resolution of the input image, thus can be blurry. 
        '''
        # annotate image with detections
        box_annotator = sv.BoxAnnotator(
            color = color,
            text_scale=0.3,
            text_thickness=1,
            text_padding=2,
        )
        mask_annotator = sv.MaskAnnotator(
            color = color
        )
        labels = [f"{classes[class_id]} {confidence:0.2f}" for confidence, class_id in zip(detections.confidence, detections.class_id)]  # added by someone
        
        if instance_random_color:
            # generate random colors for each segmentation
            # First create a shallow copy of the input detections
            detections = dataclasses.replace(detections)
            detections.class_id = np.arange(len(detections))
            
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        
        if draw_bbox:
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image, labels

    def process_cfg(self, cfg: DictConfig):
        cfg.dataset_root = Path(cfg.dataset_root)
        cfg.dataset_config = Path(cfg.dataset_config)
        
        if cfg.dataset_config.name != "multiscan.yaml":
            # For datasets whose depth and RGB have the same resolution
            # Set the desired image heights and width from the dataset config
            dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
            if cfg.image_height is None:
                cfg.image_height = dataset_cfg.camera_params.image_height
            if cfg.image_width is None:
                cfg.image_width = dataset_cfg.camera_params.image_width
            print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
        else:
            # For dataset whose depth and RGB have different resolutions
            assert cfg.image_height is not None and cfg.image_width is not None, \
                "For multiscan dataset, image height and width must be specified"

        return cfg

    def crop_image_and_mask(self, image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
        """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""
        
        image = np.array(image)
        # Verify initial dimensions
        if image.shape[:2] != mask.shape:
            print("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))
            return None, None

        # Define the cropping coordinates
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        # round the coordinates to integers
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

        # Crop the image and the mask
        image_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        # Verify cropped dimensions
        if image_crop.shape[:2] != mask_crop.shape:
            print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
            return None, None
        
        # convert the image back to a pil image
        image_crop = Image.fromarray(image_crop)

        return image_crop, mask_crop
    
    def get_pose_matrix(self):
        x = self.map_size_cm / 100.0 / 2.0 + self.observations['gps'][0]
        y = self.map_size_cm / 100.0 / 2.0 - self.observations['gps'][1]
        t = (self.observations['compass'] - np.pi / 2)[0] # input degrees and meters
        pose_matrix = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t), np.cos(t), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return pose_matrix

    def segment2d(self):
        if self.sam_variant == 'sam' or self.sam_variant == 'groundedsam':
            with torch.no_grad():
                mask, xyxy, conf, caption = self.get_sam_segmentation_dense(self.sam_variant, self.mask_generator, self.image_rgb)
                self.seg_xyxy = xyxy
                self.seg_caption = caption
            if caption is None:
                return
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            # with torch.no_grad():
            #     image_crops, image_feats, text_feats = self.compute_clip_features(image_rgb, detections, self.clip_model, self.clip_preprocess, self.clip_tokenizer, self.classes, self.device)
            # image_appear_efficiency = [''] * len(image_crops)
            image_appear_efficiency = [''] * len(mask)
            self.segment2d_results.append({
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "class_id": detections.class_id,
                "mask": detections.mask,
                "classes": self.classes,
                # "image_crops": image_crops,
                # "image_feats": image_feats,
                # "text_feats": text_feats,
                "image_appear_efficiency": image_appear_efficiency,
                "image_rgb": self.image_rgb,
                "caption": caption,
            })

    def mapping3d(self):
        depth_array = self.image_depth
        depth_array = depth_array[..., 0]
        gobs = self.segment2d_results[-1]
        cam_K = self.camera_matrix
            
        idx = len(self.segment2d_results) - 1

        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = self.image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = self.pose_matrix,
            class_names = self.classes,
            BG_CLASSES = self.BG_CLASSES,
            is_navigation = self.is_navigation
            # color_path = color_path,
        )
        
        if len(fg_detection_list) == 0:
            return
            
        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])

            # Skip the similarity computation 
            self.objects_post = filter_objects(self.cfg, self.objects)
            return
                
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        # visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects)
        # agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)
        
        # Threshold sims according to cfg. Set to negative infinity if below threshold
        # agg_sim[agg_sim < self.cfg.sim_threshold] = float('-inf')
        spatial_sim[spatial_sim < self.cfg.sim_threshold_spatial] = float('-inf')
        
        # self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, agg_sim)
        self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, spatial_sim)
        self.objects_post = filter_objects(self.cfg, self.objects)
            
    def get_caption(self):
        if self.sam_variant == 'groundedsam':
            for idx, object in enumerate(self.objects_post):
                caption_list = []
                for idx_det in range(len(object["image_idx"])):
                    caption = self.segment2d_results[object["image_idx"][idx_det]]['caption'][object["mask_idx"][idx_det]]
                    caption_list.append(caption)
                caption = self.find_modes(caption_list)[0]
                object['captions'] = [caption]

    def update_node(self):
        # update nodes
        for i, node in enumerate(self.nodes):
            caption_ori = node.caption
            # caption_new = self.find_modes(self.objects_post[i]['captions'])[0]
            caption_new = node.object['captions'][0]
            if caption_ori != caption_new:
                node.set_caption(caption_new)
        # add new nodes
        new_objects = list(filter(lambda object: 'node' not in object, self.objects_post))
        for new_object in new_objects:
            new_node = ObjectNode()
            # caption = self.find_modes(self.objects_post[i]['captions'])[0]
            caption = self.objects_post[i]['captions'][0]
            new_node.set_caption(caption)
            new_node.set_object(self.objects_post[i])
            self.nodes.append(new_node)
        # get node.center and node.room
        for node in self.nodes:
            points = np.asarray(node.object['pcd'].points)
            center = points.mean(axis=0)
            x = int(center[0] * 100 / self.map_resolution)
            y = int(center[1] * 100 / self.map_resolution)
            y = self.map_size - 1 - y
            node.set_center([x, y])
            if 0 <= x < self.map_size and 0 <= y < self.map_size and hasattr(self, 'room_map'):
                if sum(self.room_map[0, :, y, x]!=0).item() == 0:
                    room_label = 0
                else:
                    room_label = torch.where(self.room_map[0, :, y, x]!=0)[0][0].item()
            else:
                room_label = 0
            if node.room_node is not self.room_nodes[room_label]:
                if node.room_node is not None:
                    node.room_node.nodes.discard(node)
                node.room_node = self.room_nodes[room_label]
                node.room_node.nodes.add(node)

    def update_edge(self):
        old_nodes = []
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                new_nodes.append(node)
                node.is_new_node = False
            else:
                old_nodes.append(node)
        if len(new_nodes) == 0:
            return
        # create the edge between new_node and old_node
        new_edges = []
        for i, new_node in enumerate(new_nodes):
            for j, old_node in enumerate(old_nodes):
                new_edge = Edge(new_node, old_node)
                new_edges.append(new_edge)
        # create the edge between new_node
        for i, new_node1 in enumerate(new_nodes):
            for j, new_node2 in enumerate(new_nodes[i + 1:]):
                new_edge = Edge(new_node1, new_node2)
                new_edges.append(new_edge)
        # get all new_edges
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        # get all relation proposals
        if len(new_edges) > 0:
            node_pairs = []
            for new_edge in new_edges:
                node_pairs.append(new_edge.node1.caption)
                node_pairs.append(new_edge.node2.caption)
            prompt = self.prompt_edge_proposal + '\n({}, {})' * len(new_edges)
            prompt = prompt.format(*node_pairs)
            relations = self.get_llm_response(prompt=prompt)
            relations = relations.split('\n')
            if len(relations) == len(new_edges):
                for i, relation in enumerate(relations):
                    new_edges[i].set_relation(relation)
            # discriminate all relation proposals
            self.free_map = self.fbe_free_map.cpu().numpy()[0,0,::-1].copy() > 0.5
            for i, new_edge in enumerate(new_edges):
                if new_edge.relation == None or not self.discriminate_relation(new_edge):
                    new_edge.delete()

    def update_group(self):
        for room_node in self.room_nodes:
            if len(room_node.nodes) > 0:
                room_node.group_nodes = []
                object_nodes = list(room_node.nodes)
                centers = [object_node.center for object_node in object_nodes]
                centers = np.array(centers)
                dbscan = DBSCAN(eps=10, min_samples=1)  
                clusters = dbscan.fit_predict(centers)  
                for i in range(clusters.max() + 1):
                    group_node = GroupNode()
                    indices = np.where(clusters == i)[0]
                    for index in indices:
                        group_node.nodes.append(object_nodes[index])
                    group_node.get_graph()
                    room_node.group_nodes.append(group_node)

    def insert_goal(self, goal=None):
        if goal is None:
            goal = self.obj_goal
        self.update_group()
        room_node_text = ''
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0:
                room_node_text = room_node_text + room_node.caption + ','
        # room_node_text[-2] = '.'
        if room_node_text == '':
            return None
        prompt = self.prompt_room_predict.format(goal, room_node_text)
        response = self.get_llm_response(prompt=prompt)
        response = response.lower()
        predict_room_node = None
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0 and room_node.caption.lower() in response:
                predict_room_node = room_node
        if predict_room_node is None:
            return None
        for group_node in predict_room_node.group_nodes:
            corr_score = self.graph_corr(goal, group_node)
            group_node.corr_score = corr_score
        sorted_group_nodes = sorted(predict_room_node.group_nodes)
        self.mid_term_goal = sorted_group_nodes[-1].center
        return self.mid_term_goal
    
    def update_scenegraph(self):
        print(f'update_observation {self.navigate_steps}...')
        self.segment2d()
        if len(self.segment2d_results) == 0:
            return
        self.mapping3d()
        self.get_caption()
        self.update_node()
        self.update_edge()
    
    def get_llm_response(self, prompt):
        response = self.llm(prompt)
        return response
        
    def find_modes(self, lst):  
        if len(lst) == 0:
            return ['object']
        else:
            counts = Counter(lst)  
            max_count = max(counts.values())  
            modes = [item for item, count in counts.items() if count == max_count]  
            return modes  
        
    def get_joint_image(self, node1, node2):
        image_idx1 = node1.object["image_idx"]
        image_idx2 = node2.object["image_idx"]
        image_idx = set(image_idx1) & set(image_idx2)
        if len(image_idx) == 0:
            return None
        conf_max = -np.inf
        # get joint images of the two nodes
        for idx in image_idx:
            conf1 = node1.object["conf"][image_idx1.index(idx)]
            conf2 = node2.object["conf"][image_idx2.index(idx)]
            conf = conf1 + conf2
            if conf > conf_max:
                conf_max = conf
                idx_max = idx
        image = self.segment2d_results[idx_max]["image_rgb"]
        image = Image.fromarray(image)
        return image

    def discriminate_relation(self, edge):
        image = self.get_joint_image(edge.node1, edge.node2)
        if image is not None:
            response = self.vlm(self.prompt_discriminate_relation.format(edge.node1.caption, edge.node2.caption, edge.relation), image)
            if 'yes' in response.lower():
                return True
            else:
                return False
        else:
            if edge.node1.room_node != edge.node2.room_node:
                return False
            x1, y1 = edge.node1.center
            x2, y2 = edge.node2.center
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance > self.map_size // 40:
                return False
            alpha = math.atan2(y2 - y1, x2 - x1)  
            sin_2alpha = 2 * math.sin(alpha) * math.cos(alpha)
            if not -0.05 < sin_2alpha < 0.05:
                return False
            n = 3
            for i in range(1, n):
                x = int(x1 + (x2 - x1) * i / n)
                y = int(y1 + (y2 - y1) * i / n)
                if not self.free_map[y, x]:
                    return False
            return True
        
    def perception(self):
        N_stop = self.threshold_list[self.obj_goal]
        N_stop = min(N_stop, self.N_max)
        if self.agent.found_goal_times < N_stop:
            # perform object detection
            self.agent.detect_objects(self.observations)
            if self.agent.total_steps % 2 == 0 and self.agent.args.reasoning in ['both','room']:
                room_detection_result = self.agent.glip_demo.inference(self.observations["rgb"][:,:,[2,1,0]], self.agent.rooms_captions)
                self.agent.update_room_map(self.observations, room_detection_result)

    def score(self, frontier_locations_16, num_16_frontiers):
        scores = np.zeros((num_16_frontiers))
        for i in range(21):
            num_obj = len(self.agent.obj_locations[i])
            if num_obj <= 0:
                continue
            frontier_location_mtx = np.tile(frontier_locations_16, (num_obj,1,1))
            obj_location_mtx = np.array(self.agent.obj_locations[i])[:,1:]
            obj_confidence_mtx = np.tile(np.array(self.agent.obj_locations[i])[:,0],(num_16_frontiers,1)).transpose(1,0)
            obj_location_mtx = np.tile(obj_location_mtx, (num_16_frontiers,1,1)).transpose(1,0,2)
            dist_frontier_obj = np.square(frontier_location_mtx - obj_location_mtx)
            dist_frontier_obj = np.sqrt(np.sum(dist_frontier_obj, axis=2)) / 20
            near_frontier_obj = dist_frontier_obj < 1.6
            obj_confidence_mtx[near_frontier_obj==False] = 0
            obj_confidence_max = np.max(obj_confidence_mtx, axis=0)
            score_1 = np.clip(1-(1-self.agent.prob_array_obj[i])-(1-obj_confidence_max), 0, 10)
            score_2 = 1- np.clip(self.agent.prob_array_obj[i]+(1-obj_confidence_max), -10,1)
            scores += score_1 - score_2

        predict_goal_xy = self.insert_goal()
        if predict_goal_xy is not None:
            predict_goal_xy = np.array(predict_goal_xy).reshape(1, 2)
            distance = np.linalg.norm(predict_goal_xy - frontier_locations_16, axis=1)
            score = np.tile(np.array(10), (num_16_frontiers))
            score[distance > 32] = 0
            score = score / distance
            scores += score
        return scores

    def reset(self):
        full_w, full_h = self.map_size, self.map_size
        self.full_w = full_w
        self.full_h = full_h
        self.visited = torch.zeros(full_w, full_h).float().cpu().numpy()
        self.num_of_goal = torch.zeros(full_w, full_h).int()
        self.segment2d_results = []
        self.reason = ''
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.group_nodes = []
        self.init_room_nodes()
        self.edge_text = ''
        self.edge_list = []
        self.reason_visualization = ''

    def graph_corr(self, goal, graph):
        prompt = self.prompt_graph_corr_0.format(graph.center_node.caption, goal)
        response_0 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_1.format(graph.center_node.caption, goal)
        response_1 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_2.format(graph.caption, response_1)
        response_2 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_3.format(response_0, response_1 + response_2, graph.center_node.caption, goal)
        response_3 = self.get_llm_response(prompt=prompt)
        corr_score = self.text2value(response_3)
        return corr_score
    
    def text2value(self, text):
        try:
            value = float(text)
        except:
            value = 0
        return value
   