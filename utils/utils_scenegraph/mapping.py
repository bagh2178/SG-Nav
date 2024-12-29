import torch
from .slam_classes import MapObjectList, DetectionList
from .utils import compute_overlap_matrix_2set, merge_obj2_into_obj1



def compute_spatial_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the spatial similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of spatial similarities
    '''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    if cfg.spatial_sim_type == "iou":
        spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou":
        spatial_sim = compute_giou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "iou_accurate":
        spatial_sim = compute_3d_iou_accuracte_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou_accurate":
        spatial_sim = compute_3d_giou_accurate_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "overlap":
        spatial_sim = compute_overlap_matrix_2set(cfg, objects, detection_list)
        spatial_sim = torch.from_numpy(spatial_sim).T
    else:
        raise ValueError(f"Invalid spatial similarity type: {cfg.spatial_sim_type}")
    
    return spatial_sim


def merge_detections_to_objects(
    cfg, 
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    # Iterate through all detections and merge them into objects
    for i in range(agg_sim.shape[0]):
        # If not matched to any object, add it as a new object
        if agg_sim[i].max() == float('-inf'):
            # detection_list[i]['id'] = len(objects)
            objects.append(detection_list[i])
        # Merge with most similar existing object
        else:
            j = agg_sim[i].argmax()
            matched_det = detection_list[i]
            matched_obj = objects[j]
            merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
            objects[j] = merged_obj
            
    return objects