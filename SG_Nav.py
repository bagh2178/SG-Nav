import argparse
import imp
from multiprocessing.context import ForkContext
import os
import math
import numba
import time
import random
import numpy as np
import skimage
import torch
from torchvision.utils import save_image
import copy
from PIL import Image, ImageDraw, ImageFont
import pandas
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
import matplotlib.pyplot as plt
from matplotlib import colors
import colorsys

import habitat
from habitat.config import Config
from habitat.core.agent import Agent
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from GLIP.maskrcnn_benchmark.config import cfg as glip_cfg
from utils_glip import *

from utils_fmm.fmm_planner import FMMPlanner
from utils_fmm.mapping import Semantic_Mapping
import utils_fmm.control_helper as CH
import utils_fmm.pose_utils as pu

from pslpython.model import Model as PSLModel
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

from .scenegraph import SceneGraph


ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]

class CLIP_LLM_FMMAgent_NonPano(Agent):
    """
    New in this version: 
    1. use obj and room reasoning by record object locations and build a room map 
    experiments: v4_4
    """
    def __init__(self, task_config, args=None):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.config = task_config
        self.args = args
        self.panoramic = []
        self.panoramic_depth = []
        self.turn_angles = 0
        self.device = (
            torch.device("cuda:{}".format(0))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.prev_action = 0
        self.current_search_room = ''
        self.navigate_steps = 0
        self.move_steps = 0
        self.total_steps = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.threshold_list = {'bathtub': 3, 'bed': 3, 'cabinet': 2, 'chair': 1, 'chest_of_drawers': 3, 'clothes': 2, 'counter': 1, 'cushion': 3, 'fireplace': 3, 'gym_equipment': 2, 'picture': 3, 'plant': 3, 'seating': 0, 'shower': 2, 'sink': 2, 'sofa': 2, 'stool': 2, 'table': 1, 'toilet': 3, 'towel': 2, 'tv_monitor': 0}
        self.found_goal_times_threshold = 3
        self.correct_room = False
        self.changing_room = False
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.force_change_room = False
        self.current_room_search_step = 0
        self.target_room = ''
        self.current_rooms = []
        self.nav_without_goal_step = 0
        self.former_collide = 0
        self.history_pose = []
        self.loop_time = 0
        self.stuck_time = 0
        self.rooms = rooms
        self.room_captions = rooms_captions
        self.split = (self.args.split_l >= 0)

        ### ------ init glip model ------ ###
        config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml" 
        weight_file = "GLIP/MODEL/glip_large_model.pth"
        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file) 
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.glip_demo = GLIPDemo(
            glip_cfg,
            min_image_size=800,
            confidence_threshold=0.61,
            show_mask_heatmaps=False
        )
        print('glip init finish!!!')

        ### ----- init some static variables ----- ###
        self.map_size_cm = 4000
        self.resolution = self.map_resolution = 5
        self.camera_horizon = 0
        self.dilation_deg = 0
        self.collision_threshold = 0.08
        self.col_width = 5
        self.selem = skimage.morphology.square(1)
        
        ### ----- init maps ----- ###
        self.init_map()
        self.sem_map_module = Semantic_Mapping(self).to(self.device) 
        self.free_map_module = Semantic_Mapping(self, max_height=10,min_height=-150).to(self.device)
        self.room_map_module = Semantic_Mapping(self, max_height=200,min_height=-10, num_cats=9).to(self.device)
        
        self.free_map_module.eval()
        self.free_map_module.set_view_angles(self.camera_horizon)
        self.sem_map_module.eval()
        self.sem_map_module.set_view_angles(self.camera_horizon)
        self.room_map_module.eval()
        self.room_map_module.set_view_angles(self.camera_horizon)

        self.camera_matrix = self.free_map_module.camera_matrix  # added by someone
        
        print('FMM navigate map init finish!!!')
        
        ### ----- load commonsense from LLMs ----- ###
        self.goal_idx = {}
        for key in projection:
            self.goal_idx[projection[key]] = categories_21.index(projection[key]) # each goal corresponding to which column in co-orrcurance matrix 
        self.co_occur_mtx = np.load('ablations/npys/deberta_predict.npy')
        self.co_occur_mtx -= self.co_occur_mtx.min()
        self.co_occur_mtx /= self.co_occur_mtx.max() 
        
        self.co_occur_room_mtx = np.load('ablations/npys/deberta_predict_room.npy')
        self.co_occur_room_mtx -= self.co_occur_room_mtx.min()
        self.co_occur_room_mtx /= self.co_occur_room_mtx.max()
        
        ### ----- option: using PSL optimization ADMM ----- ###
        if self.args.PSL_infer:
            self.psl_model = PSLModel('objnav1')  ## important: please use different name here for different process in the same machine. eg. objnav, objnav2, ...
            # Add Predicates
            self.add_predicates(self.psl_model)

            # Add Rules
            self.add_rules(self.psl_model)

        ### ----- load scene graph module ----- ###
        self.goal_verification = True
        self.scenegraph = SceneGraph(agent=self)

        self.experiment_name = 'experiment_0'

        if self.split:
            self.experiment_name = self.experiment_name + f'/[{self.args.split_l}:{self.args.split_r}]'

        self.save_image_dir = f'figures/{self.experiment_name}/image'

        print('scene graph module init finish!!!')

    def add_predicates(self, model):
        """
        add predicates for ADMM PSL inference
        """
        if self.args.reasoning in ['both', 'obj']:

            predicate = Predicate('IsNearObj', closed = True, size = 2)
            model.add_predicate(predicate)
            
            predicate = Predicate('ObjCooccur', closed = True, size = 1)
            model.add_predicate(predicate)
        if self.args.reasoning in ['both', 'room']:

            predicate = Predicate('IsNearRoom', closed = True, size = 2)
            model.add_predicate(predicate)
            
            predicate = Predicate('RoomCooccur', closed = True, size = 1)
            model.add_predicate(predicate)
        
        predicate = Predicate('Choose', closed = False, size = 1)
        model.add_predicate(predicate)
        
        predicate = Predicate('ShortDist', closed = True, size = 1)
        model.add_predicate(predicate)
        
    def add_rules(self, model):
        """
        add rules for ADMM PSL inference
        """
        if self.args.reasoning in ['both', 'obj']:
            model.add_rule(Rule('2: ObjCooccur(O) & IsNearObj(O,F)  -> Choose(F)^2'))
            model.add_rule(Rule('2: !ObjCooccur(O) & IsNearObj(O,F) -> !Choose(F)^2'))
        if self.args.reasoning in ['both', 'room']:
            model.add_rule(Rule('2: RoomCooccur(R) & IsNearRoom(R,F) -> Choose(F)^2'))
            model.add_rule(Rule('2: !RoomCooccur(R) & IsNearRoom(R,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: ShortDist(F) -> Choose(F)^2'))
        model.add_rule(Rule('Choose(+F) = 1 .'))
    
    def reset(self):
        """
        reset variables for each episodes
        """
        self.navigate_steps = 0
        self.turn_angles = 0
        self.move_steps = 0
        self.total_steps = 0
        self.current_room_search_step = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.ever_long_goal = False
        self.correct_room = False
        self.changing_room = False
        self.goal_loc = None
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.prev_action = 0
        self.col_width = 5
        self.former_collide = 0
        self.goal_gps = np.array([0.,0.])
        self.long_goal_temp_gps = np.array([0.,0.])
        self.last_gps = np.array([11100.,11100.])
        self.has_panarama = False
        self.init_map()
        self.last_loc = self.full_pose
        self.panoramic = []
        self.panoramic_depth = []
        self.current_rooms = []
        self.dist_to_frontier_goal = 10
        self.first_fbe = False
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        self.found_long_goal = False
        self.history_pose = []
        self.loop_time = 0
        self.stuck_time = 0
        self.found_goal_times_threshold = self.threshold_list[self.benchmark._env.current_episode.object_category]
        ###########
        self.current_obj_predictions = []
        self.obj_locations = [[] for i in range(21)] # length equal to all the objects in reference matrix 
        self.not_move_steps = 0
        self.move_since_random = 0
        self.using_random_goal = False
        
        self.fronter_this_ex = 0
        self.random_this_ex = 0
        ########### error analysis
        self.detect_true = False
        self.goal_appear = False
        self.frontiers_gps = []
        
        self.last_location = np.array([0.,0.])
        self.current_stuck_steps = 0
        self.total_stuck_steps = 0
        self.scenegraph.reset()
        
        os.system(f'rm -r {self.save_image_dir}')
        
    def detect_objects(self, observations):
        """
        detect objects from current observations and update semantic map.
        """
        self.current_obj_predictions = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], object_captions) # GLIP object detection, time cosuming
        new_labels = self.get_glip_real_label(self.current_obj_predictions) # transfer int labels to string labels
        self.current_obj_predictions.add_field("labels", new_labels)

        observations["rgb_annotated"] = self.draw_bboxes_with_labels(rgb=observations["rgb_annotated"], bboxes=self.current_obj_predictions.bbox.numpy(), labels=self.current_obj_predictions.get_field("labels"), color='pink')
        
        shortest_distance = 120 # TODO: shortest distance  or most confident?
        shortest_distance_angle = 0
        goal_prediction = copy.deepcopy(self.current_obj_predictions)
        obj_labels = self.current_obj_predictions.get_field("labels")
        goal_bbox = []
        ### save the bounding boxes if there is a goal object
        for j, label in enumerate(obj_labels):
            if self.obj_goal in label:
                goal_bbox.append(self.current_obj_predictions.bbox[j])
                observations["rgb_annotated"] = self.draw_bboxes_with_labels(rgb=observations["rgb_annotated"], bboxes=np.expand_dims(self.current_obj_predictions.bbox[j].numpy(), axis=0), labels=[self.obj_goal], color='red')
            elif self.obj_goal == 'gym_equipment' and (label in ['treadmill', 'exercise machine']):
                goal_bbox.append(self.current_obj_predictions.bbox[j])
        
        ### record the location of object center in the semantic map for object reasoning.
        if self.args.reasoning == 'both' or 'obj':
            for j, label in enumerate(obj_labels):
                if label in categories_21_origin:
                    confidence = self.current_obj_predictions.get_field("scores")[j]
                    bbox = self.current_obj_predictions.bbox[j].to(torch.int64)
                    center_point = (bbox[:2] + bbox[2:]) // 2
                    temp_direction = (center_point[0] - 320) * 79 / 640
                    temp_distance = self.depth[center_point[1],center_point[0],0]
                    if temp_distance >= 4.999:
                        continue
                    obj_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                    x = int(self.map_size_cm/10-obj_gps[1]*100/self.resolution)
                    y = int(self.map_size_cm/10+obj_gps[0]*100/self.resolution)
                    self.obj_locations[categories_21_origin.index(label)].append([confidence, x, y])
        
        ### if detect a goal object, determine if it's beyond 5 meters or not. 
        if len(goal_bbox) > 0:
            long_goal_detected_before = copy.deepcopy(self.found_long_goal)
            goal_prediction.bbox = torch.stack(goal_bbox)
            for box in goal_prediction.bbox:  ## select the closest goal as the detected goal
                box = box.to(torch.int64)
                center_point = (box[:2] + box[2:]) // 2
                temp_direction = (center_point[0] - 320) * 79 / 640
                temp_distance = self.depth[center_point[1],center_point[0],0]
                k = 0
                pos_neg = 1
                ## case that a detected goal is within 0.5 meters, maybe it's because the image is corrupted, let's find another points in the image instead of the center point
                while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                    pos_neg *= -1
                    k += 0.5
                    temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                    self.depth[center_point[1],center_point[0]+int(pos_neg*k),0])
                    
                if temp_distance >= 4.999:
                    self.found_long_goal = True
                    self.ever_long_goal = True
                else:
                    if self.found_goal:  # added by someone
                        # if self.double_found_goal:
                        #     self.triple_found_goal = True
                        # self.double_found_goal = True
                        self.found_goal_times = self.found_goal_times + 1
                    goal_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
                    goal_xy = [int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution), int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)]
                    verification_passed = self.scenegraph.verify_goal(goal_xy)
                    if verification_passed:
                        self.found_goal = True
                    self.found_long_goal = False
                
                ## select the closest goal
                direction = temp_direction
                distance = temp_distance
                if distance < shortest_distance:
                    shortest_distance = distance
                    shortest_distance_angle = direction
                    box_shortest = copy.deepcopy(box)
            
            if self.found_goal:
                self.goal_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            elif not long_goal_detected_before:
                # if detected a long goal before, then don't change it until see a goal within 5 meters
                self.long_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            if self.args.error_analysis and self.found_goal:
                if (observations['semantic'][box_shortest[0]:box_shortest[2],box_shortest[1]:box_shortest[3]] == self.goal_mp3d_idx).sum() > min(300, 0.2 * (box_shortest[2]-box_shortest[0])*(box_shortest[3]-box_shortest[1])):
                     self.detect_true = True
        else:  # added by someone
            if self.found_goal:
                self.found_goal = False
                self.found_goal_times = 0
                # self.double_found_goal = False
                # self.triple_found_goal = False
            
                   
    def act(self, observations):
        """ 
        observations: 
        """ 
        # if self.total_steps >= 482:  # due to using turning 60 degree action at the beginning. 
        if self.total_steps >= 500:  # 230 is setted by someone
            return {"action": 0}
        
        self.total_steps += 1
        if self.navigate_steps == 0:
            self.obj_goal = projection[int(observations["objectgoal"])]
            self.prob_array_room = self.co_occur_room_mtx[self.goal_idx[self.obj_goal]]
            self.prob_array_obj = self.co_occur_mtx[self.goal_idx[self.obj_goal]]
            ## ADMM PSL optim only 
            if self.args.PSL_infer == 'optim':
                if self.args.reasoning in ['both','room']:
                    for predicate in self.psl_model.get_predicates().values():
                        if predicate.name() in ['ROOMCOOCCUR']:
                            predicate.clear_data()
                    prob_array_room_list = list(self.prob_array_room)
                    data = pandas.DataFrame([[i, prob_array_room_list[i]] for i in range(len(prob_array_room_list))], columns = list(range(2)))
                    self.psl_model.get_predicate('RoomCooccur').add_data(Partition.OBSERVATIONS, data)
                
                if self.args.reasoning in ['both','obj']:
                    for predicate in self.psl_model.get_predicates().values():
                        if predicate.name() in ['OBJCOOCCUR']:
                            predicate.clear_data()
                    prob_array_obj_list = list(self.prob_array_obj)
                    data = pandas.DataFrame([[i, prob_array_obj_list[i]] for i in range(len(prob_array_obj_list))], columns = list(range(2)))
                    self.psl_model.get_predicate('ObjCooccur').add_data(Partition.OBSERVATIONS, data)

        observations["depth"][observations["depth"]==0.5] = 100 # don't construct unprecise map with distance less than 0.5 m
        self.depth = observations["depth"]
        self.rgb = observations["rgb"][:,:,[2,1,0]]
        observations["rgb_annotated"] = observations["rgb"]

        self.scenegraph.update_observation(observations)
        
        self.update_map(observations)
        self.update_free_map(observations)
        
        if self.args.visulize and False:  # Falsed by someone
            input_pose = np.zeros(7)
            traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)
            save_map = copy.deepcopy(torch.from_numpy(traversible))
            gray_map = torch.stack((save_map, save_map, save_map))
            save_image((gray_map / gray_map.max()), 'figures/map/img'+str(self.total_steps)+'.png')
            save_image(torch.from_numpy(observations["rgb"]/255).float().permute(2,0,1), 'figures/rgb/img'+str(self.total_steps)+'.png')
            save_image(torch.from_numpy(observations["depth"]/5).float().permute(2,0,1).float(), 'figures/dist/d'+str(self.navigate_steps)+'.png')
        # look down twice and look around at first to initialize map
        if self.total_steps == 1:
            # look down
            self.sem_map_module.set_view_angles(30)
            self.free_map_module.set_view_angles(30)
            # self.observed_map_module.set_view_angles(30)
            return {"action": 5}
        elif self.total_steps <= 7:
            return {"action": 6}
        elif self.total_steps == 8:
            # look down
            self.sem_map_module.set_view_angles(60)
            self.free_map_module.set_view_angles(60)
            # self.observed_map_module.set_view_angles(60)
            return {"action": 5}
        elif self.total_steps <= 14:
            return {"action": 6}
        elif self.total_steps <= 15:
            self.sem_map_module.set_view_angles(30)
            self.free_map_module.set_view_angles(30)
            # self.observed_map_module.set_view_angles(30)
            return {"action": 4}
        elif self.total_steps <= 16:
            self.sem_map_module.set_view_angles(0)
            self.free_map_module.set_view_angles(0)
            # self.observed_map_module.set_view_angles(0)
            return {"action": 4}
        # get panoramic view at first
        if self.total_steps <= 22 and not self.found_goal:
            self.panoramic.append(observations["rgb"][:,:,[2,1,0]])
            self.panoramic_depth.append(observations["depth"])
            self.detect_objects(observations)
            room_detection_result = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], rooms_captions)
            self.update_room_map(observations, room_detection_result)
            if not self.found_goal: # if found a goal, directly go to it
                return {"action": 6}
                    
        
        if not (observations["gps"] == self.last_gps).all():
            self.move_steps += 1
            self.not_move_steps = 0
            if self.using_random_goal:
                self.move_since_random += 1
        else:
            self.not_move_steps += 1
            
        self.last_gps = observations["gps"]
        
        # if not self.found_goal:
        if (self.goal_verification and self.found_goal_times < self.found_goal_times_threshold) or (not self.goal_verification and not self.found_goal):
            ## perform object and room detection if have't found a goal
            self.detect_objects(observations)
            if self.total_steps % 2 == 0 and self.args.reasoning in ['both','room']:
                room_detection_result = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], rooms_captions)
                self.update_room_map(observations, room_detection_result)
          
        ### ------ generate action using FMM ------ ###
        ## update pose and map
        self.history_pose.append(self.full_pose.cpu().detach().clone())
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)
        
        if self.found_goal: 
            ## directly go to goal
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[0]*100/self.resolution)))] = 1
        elif self.found_long_goal: 
            ## go to long goal
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.long_goal_temp_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.long_goal_temp_gps[0]*100/self.resolution)))] = 1
        elif not self.first_fbe: # first FBE process
            self.goal_loc = self.fbe(traversible, cur_start)  # , annotated by someone
            self.not_use_random_goal()
            self.first_fbe = True
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
        
        stg_y, stg_x, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        if self.found_long_goal and number_action == 0: # didn't detect goal when arrive at long goal, start over FBE. 
            self.found_long_goal = False
        
        if (not self.found_goal and not self.found_long_goal and number_action == 0) or (self.using_random_goal and self.move_since_random > 20): 
            # FBE if arrive at a selected frontier, or randomly explore for some steps
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
            stg_y, stg_x, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        self.loop_time = 0
        while (not self.found_goal and number_action == 0) or self.not_move_steps >= 7:
            # the agent is stuck, then random explore
            self.loop_time += 1
            self.random_this_ex += 1
            self.stuck_time += 1
            if self.loop_time > 20 or self.stuck_time == 5:
                return {"action": 0}
            self.not_move_steps = 0
            self.goal_map = self.set_random_goal()
            self.using_random_goal = True
            stg_y, stg_x, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        # ------------------------------
        if self.args.visulize:
            save_map = copy.deepcopy(torch.from_numpy(traversible))
            gray_map = torch.stack((save_map, save_map, save_map))
            paper_obstacle_map = copy.deepcopy(gray_map)[:,1:-1,1:-1]
            gray_map = self.visualize_scenegraph_map(gray_map)
            gray_map[:, int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)-2:int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)+2, int(self.full_pose[0]*100/self.resolution)-2:int(self.full_pose[0]*100/self.resolution)+2] = 0
            gray_map[0, int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)-2:int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)+2, int(self.full_pose[0]*100/self.resolution)-2:int(self.full_pose[0]*100/self.resolution)+2] = 1
            goal_size = 3 if self.found_goal else 2
            if not self.found_goal and self.goal_loc is not None:
                gray_map[:,int(self.map_size_cm/5)-self.goal_loc[0]-goal_size:int(self.map_size_cm/5)-self.goal_loc[0]+goal_size, self.goal_loc[1]-goal_size:self.goal_loc[1]+goal_size] = 0
                gray_map[1,int(self.map_size_cm/5)-self.goal_loc[0]-goal_size:int(self.map_size_cm/5)-self.goal_loc[0]+goal_size, self.goal_loc[1]-goal_size:self.goal_loc[1]+goal_size] = 1
            else:
                gray_map[:, int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)-goal_size:int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)+goal_size, int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)-goal_size:int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)+goal_size] = 0
                gray_map[1, int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)-goal_size:int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)+goal_size, int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)-goal_size:int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)+goal_size] = 1
            # gray_map[:, int(stg_y)-1:int(stg_y)+1, int(stg_x)-1:int(stg_x)+1] = 0
            # gray_map[2, int(stg_y)-1:int(stg_y)+1, int(stg_x)-1:int(stg_x)+1] = 1
            free_map = self.fbe_free_map.cpu().numpy()[0,0,::-1].copy() > 0.5
            
            paper_map = torch.zeros_like(paper_obstacle_map)
            paper_map_trans = paper_map.permute(1,2,0)
            # unknown_rgb = colors.to_rgb('lightcyan')
            unknown_rgb = colors.to_rgb('#FFFFFF')
            paper_map_trans[:,:,:] = torch.tensor( unknown_rgb)
            # free_rgb = colors.to_rgb('floralwhite')
            free_rgb = colors.to_rgb('#E7E7E7')
            paper_map_trans[self.fbe_free_map.cpu().numpy()[0,0,::-1]>0.5,:] = torch.tensor( free_rgb).double()
            frontier_rgb = colors.to_rgb('indianred')
            selem = skimage.morphology.disk(1)
            selem = skimage.morphology.disk(4)
            # free_map[skimage.morphology.binary_dilation(free_map, selem)] = 1
            # paper_map_trans[(free_map==1)*(paper_map_trans[:,:,0]==torch.tensor(unknown_rgb)[0]).numpy(),:] = torch.tensor(frontier_rgb).double()
            paper_map_trans = self.draw_frontier_score(paper_map_trans, frontier_rgb, unknown_rgb)
            # obstacle_rgb = colors.to_rgb('dimgrey')
            obstacle_rgb = colors.to_rgb('#A2A2A2')
            paper_map_trans[skimage.morphology.binary_dilation(self.full_map.cpu().numpy()[0,0,::-1]>0.5,skimage.morphology.disk(1)),:] = torch.tensor(obstacle_rgb).double()
            paper_map_trans = paper_map_trans.permute(2,0,1)
            # self.paper_map_trans = paper_map_trans



            # self.visualize_obj_goal(paper_map_trans)
            self.visualize_agent_and_goal(paper_map_trans)
            paper_map_trans = paper_map_trans / paper_map_trans.max()
            rgb = torch.from_numpy(observations["rgb_annotated"] / 255).float().permute(2, 0, 1)
            text_image = torch.ones(rgb.shape[0], paper_map_trans.shape[1] - rgb.shape[1], rgb.shape[2])
            metrics = self.benchmark._env.get_metrics()
            text = [
                'episode_id: {}'.format(self.benchmark._env.current_episode.episode_id),
                'episode: {}'.format(self.benchmark._env.current_episode.goals_key),
                'object_category: {}'.format(self.benchmark._env.current_episode.object_category),
                'geodesic_distance: {}'.format(self.benchmark._env.current_episode.info['geodesic_distance']),
                'euclidean_distance: {}'.format(self.benchmark._env.current_episode.info['euclidean_distance']),
            ]
            paper_map_trans = self.add_text(paper_map_trans, text)
            rgb = torch.cat([rgb, text_image], dim=1)
            self.agent_state_image = torch.cat([paper_map_trans, rgb], dim=2)

            try:
                pose_dict = json.load(open('figures/pose/pose.json', 'r'))
            except:
                pose_dict = {}
            pose_dict[str(self.navigate_steps)] = {'compass': str(observations['compass']), 'gps': str(observations['gps'])}
            json.dump(pose_dict, open('figures/pose/pose.json', 'w'))
            
        observations["pointgoal_with_gps_compass"] = self.get_relative_goal_gps(observations)

        ###-----------------------------------###

        self.last_loc = copy.deepcopy(self.full_pose)
        self.prev_action = number_action
        self.navigate_steps += 1
        torch.cuda.empty_cache()
        
        return {"action": number_action}
    
    def not_use_random_goal(self):
        self.move_since_random = 0
        self.using_random_goal = False
        
    def get_glip_real_label(self, prediction):
        labels = prediction.get_field("labels").tolist()
        new_labels = []
        if self.glip_demo.entities and self.glip_demo.plus:
            for i in labels:
                if i <= len(self.glip_demo.entities):
                    new_labels.append(self.glip_demo.entities[i - self.glip_demo.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        return new_labels
    
    def fbe(self, traversible, start):
        """
        fontier: unknown area and free area 
        unknown area: not free and not obstacle 
        select a frontier using commonsense and PSL and return a GPS
        """
        fbe_map = torch.zeros_like(self.full_map[0,0])
        fbe_map[self.fbe_free_map[0,0]>0] = 1  # first free 
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle

        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp == 0] = 4  # don't know space is 4
        fbe_cp[fbe_cp < 4] = 0  # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp # intersection between unknown area and free area 
        frontier_map = diff == 1
        frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
        num_frontiers = len(torch.where(frontier_map)[0])
        if num_frontiers == 0:
            return None
        
        # for each frontier, calculate the inverse of distance
        planner = FMMPlanner(traversible, None)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        
        ## use the threshold of 1.6 to filter close frontiers to encourage exploration
        idx_16 = np.where(distances>=1.6)
        distances_16 = distances[idx_16]
        distances_16_inverse = 1 - (np.clip(distances_16,0,11.6)-1.6) / (11.6-1.6)
        frontier_locations_16 = frontier_locations[idx_16]
        self.frontier_locations = frontier_locations  # added by someone
        self.frontier_locations_16 = frontier_locations_16  # added by someone
        if len(distances_16) == 0:
            return None
        num_16_frontiers = len(idx_16[0])  # 175
        scores = np.zeros((num_16_frontiers))
                
        if self.args.reasoning in ['both', 'room']:
            for i, loc in enumerate(frontier_locations_16):
                sub_room_map = self.room_map[0,:,max(0,loc[0]-12):min(self.map_size-1,loc[0]+13), max(0,loc[1]-12):min(self.map_size-1,loc[1]+13)].cpu().numpy() # sub_room_map.shape = [9, 25, 25], select the room map around the frontier
                whether_near_room = np.max(np.max(sub_room_map, 1),1) # whether_near_room.shape = [9], 1*9 wether the frontier is close to each room
                score_1 = np.clip(1-(1-self.prob_array_room)-(1-whether_near_room), 0, 10)  # score_1.shape = [9], prob_array_room.shape = [9]
                score_2 = 1- np.clip(self.prob_array_room+(1-whether_near_room), -10,1)  # score_2.shape = [9]
                scores[i] = np.sum(score_1) - np.sum(score_2)
        
        if self.args.reasoning in ['both', 'obj']:
            for i in range(21):
                num_obj = len(self.obj_locations[i])
                if num_obj <= 0:
                    continue
                frontier_location_mtx = np.tile(frontier_locations_16, (num_obj,1,1)) # k*m*2 k: num of objects, m: num of frontiers  (4, 175, 2) = (175, 2)
                obj_location_mtx = np.array(self.obj_locations[i])[:,1:] # k*2  (4, 2)
                obj_confidence_mtx = np.tile(np.array(self.obj_locations[i])[:,0],(num_16_frontiers,1)).transpose(1,0) # k*m  (4, 175)
                obj_location_mtx = np.tile(obj_location_mtx, (num_16_frontiers,1,1)).transpose(1,0,2) # k*m*2  (4, 175, 2)
                dist_frontier_obj = np.square(frontier_location_mtx - obj_location_mtx)
                dist_frontier_obj = np.sqrt(np.sum(dist_frontier_obj, axis=2)) / 20 # k*m  (4, 175)
                near_frontier_obj = dist_frontier_obj < 1.6 # k*m 
                obj_confidence_mtx[near_frontier_obj==False] = 0 # k*m 
                obj_confidence_max = np.max(obj_confidence_mtx, axis=0)  # (175)
                score_1 = np.clip(1-(1-self.prob_array_obj[i])-(1-obj_confidence_max), 0, 10)
                score_2 = 1- np.clip(self.prob_array_obj[i]+(1-obj_confidence_max), -10,1)
                scores += score_1 - score_2
                

        for node in self.scenegraph.nodes:
            xy = np.array(node.object['xy']).reshape(1, 2)
            distance = np.linalg.norm(xy - frontier_locations_16, axis=1)
            score = np.tile(np.array(node.score), (num_16_frontiers))
            score[distance > 1.6] = 0
            score = score / distance * 20
            scores += score

        if self.args.reasoning == 'both':  # True
            scores += 2 * distances_16_inverse
        else:
            scores += 1 * distances_16_inverse
        idx_16_max = idx_16[0][np.argmax(scores)]
        goal = frontier_locations[idx_16_max] - 1  # annotated by someone
        # goal = frontier_locations_16[idx_16_max] - 1  # added by someone
        # with open("output/FBE_PSL_oh_gpt_o/frontier_dist.txt", "a") as file_object:
        #     file_object.write(str(distances[idx_16_max]) + '\n')
        self.scores = scores
        return goal
        
    def get_goal_gps(self, observations, angle, distance):
        ### return goal gps in the original agent coordinates
        if type(angle) is torch.Tensor:
            angle = angle.cpu().numpy()
        agent_gps = observations['gps']
        agent_compass = observations['compass']
        goal_direction = agent_compass - angle/180*np.pi
        goal_gps = np.array([(agent_gps[0]+np.cos(goal_direction)*distance).item(),
         (agent_gps[1]-np.sin(goal_direction)*distance).item()])
        return goal_gps

    def get_relative_goal_gps(self, observations, goal_gps=None):
        if goal_gps is None:
            goal_gps = self.goal_gps
        direction_vector = goal_gps - np.array([observations['gps'][0].item(),observations['gps'][1].item()])
        rho = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
        phi_world = np.arctan2(direction_vector[1], direction_vector[0])
        agent_compass = observations['compass']
        phi = phi_world - agent_compass
        return np.array([rho, phi.item()], dtype=np.float32)
   
    def init_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = self.map_size, self.map_size
        self.full_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.room_map = torch.zeros(1,9 ,full_w, full_h).float().to(self.device)
        self.visited = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()
        self.fbe_free_map = copy.deepcopy(self.full_map).to(self.device) # 0 is unknown, 1 is free
        self.full_pose = torch.zeros(3).float().to(self.device)
        # Origin of local map
        self.origins = np.zeros((2))
        
        def init_map_and_pose():
            self.full_map.fill_(0.)
            self.full_pose.fill_(0.)
            # full_pose[:, 2] = 90
            self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map

        init_map_and_pose()

    def update_map(self, observations):
        """
        full pose: gps and angle in the initial coordinate system, where 0 is towards the x axis
        """
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.full_map = self.sem_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.full_map)
    
    def update_free_map(self, observations):
        """
        update free map using visual projection
        """
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.fbe_free_map = self.free_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.fbe_free_map)
        self.fbe_free_map[int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4, int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4] = 1
    
    def update_room_map(self, observations, room_prediction_result):
        new_room_labels = self.get_glip_real_label(room_prediction_result)
        observations["rgb_annotated"] = self.draw_bboxes_with_labels(rgb=observations["rgb_annotated"], bboxes=room_prediction_result.bbox.numpy(), labels=new_room_labels, color='yellow')
        type_mask = np.zeros((9,self.config.SIMULATOR.DEPTH_SENSOR.HEIGHT, self.config.SIMULATOR.DEPTH_SENSOR.WIDTH))
        bboxs = room_prediction_result.bbox
        score_vec = torch.zeros((9)).to(self.device)
        for i, box in enumerate(bboxs):
            box = box.to(torch.int64)
            idx = rooms.index(new_room_labels[i])
            type_mask[idx,box[1]:box[3],box[0]:box[2]] = 1
            score_vec[idx] = room_prediction_result.get_field("scores")[i]
        self.room_map = self.room_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.room_map, torch.from_numpy(type_mask).to(self.device).type(torch.float32), score_vec)
        # self.room_map_refine = copy.deepcopy(self.room_map)
        # other_room_map_sum = self.room_map_refine[0,0] + torch.sum(self.room_map_refine[0,2:],axis=0)
        # self.room_map_refine[0,1][other_room_map_sum>0] = 0
    
    def get_traversible(self, map_pred, pose_pred):
        """
        update traversible map
        """
        grid = np.rint(map_pred)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        # start = [int(start_x), int(start_y)]
        start = pu.threshold_poses(start, grid.shape)
        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1
        #Get traversible
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        def delete_boundary(mat):
            new_mat = copy.deepcopy(mat)
            return new_mat[1:-1,1:-1]
        
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        traversible = skimage.morphology.binary_dilation(
                    grid[y1:y2, x1:x2],
                    self.selem) != True

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.navigate_steps)

        # obstacle dilation do not dilate collision
        traversible = 1 - traversible
        selem = skimage.morphology.disk(4)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        traversible[int(start[0]-y1)-1:int(start[0]-y1)+2,
            int(start[1]-x1)-1:int(start[1]-x1)+2] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible[self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 0
        traversible = add_boundary(traversible)
        return traversible, start, start_o
    
    def _plan(self, traversible, goal_map, agent_pose, start, start_o, goal_found):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        # if newly_goal_set:
        #     self.action_5_count = 0

        if self.prev_action == 1:
            x1, y1, t1 = self.last_loc.cpu().numpy()
            x2, y2, t2 = self.full_pose.cpu()
            y1 = self.map_size_cm/100 - y1
            y2 = self.map_size_cm/100 - y2
            t1 = -t1
            t2 = -t2
            buf = 4
            length = 5

            if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 1
                self.col_width = min(self.col_width, 3)
            else:
                self.col_width = 1
            # self.col_width = 4
            dist = pu.get_l2_distance(x1, x2, y1, y2)
            col_threshold = self.collision_threshold

            if dist < col_threshold: # Collision
                self.former_collide += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                        (j-width//2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                        (j-width//2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(round(r*100/self.map_resolution)), \
                               int(round(c*100/self.map_resolution))
                        [r, c] = pu.threshold_poses([r, c],
                                    self.collision_map.shape)
                        self.collision_map[r,c] = 1
            else:
                self.former_collide = 0

        stg, stop, = self._get_stg(traversible, start, np.copy(goal_map), goal_found)

        # Deterministic Local Policy
        if stop:
            action = 0
            (stg_y, stg_x) = stg

        else:
            (stg_y, stg_x) = stg
            angle_st_goal = math.degrees(math.atan2(stg_y - start[0],
                                                stg_x - start[1]))
            angle_agent = (start_o)%360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_st_goal- angle_agent)%360.0
            if relative_angle > 180:
                relative_angle -= 360
            if self.former_collide < 10:
                if relative_angle > 16:
                    action = 3 # Right
                elif relative_angle < -16:
                    action = 2 # Left
                else:
                    action = 1
            elif self.prev_action == 1:
                if relative_angle > 0:
                    action = 3 # Right
                else:
                    action = 2 # Left
            else:
                action = 1
            if self.former_collide >= 10 and self.prev_action != 1:
                self.former_collide  = 0
            if stg_y == start[0] and stg_x == start[1]:
                action = 1

        return stg_y, stg_x, action
    
    def _get_stg(self, traversible, start, goal, goal_found):
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        def delete_boundary(mat):
            new_mat = copy.deepcopy(mat)
            return new_mat[1:-1,1:-1]
        
        goal = add_boundary(goal, value=0)
        original_goal = copy.deepcopy(goal)
        
            
        centers = []
        if len(np.where(goal !=0)[0]) > 1:
            goal, centers = CH._get_center_goal(goal)
        state = [start[0] + 1, start[1] + 1]
        self.planner = FMMPlanner(traversible, None)
            
        if self.dilation_deg!=0: 
            #if self.args.debug_local:
            #    self.print_log("dilation added")
            goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
            
        if goal_found:
            # if self.args.debug_local:
            #     self.print_log("goal found!")
            try:
                goal = CH._block_goal(centers, goal, original_goal, goal_found)
            except:
                goal = self.set_random_goal(goal)

        self.planner.set_multi_goal(goal, state) # time cosuming 

        decrease_stop_cond =0
        if self.dilation_deg >= 6:
            decrease_stop_cond = 0.2 #decrease to 0.2 (7 grids until closest goal)
        stg_y, stg_x, replan, stop = self.planner.get_short_term_goal(state, found_goal = goal_found, decrease_stop_cond=decrease_stop_cond)
        stg_x, stg_y = stg_x - 1, stg_y - 1
        if stop:
            a = 1
        
        # self.closest_goal = CH._get_closest_goal(start, goal)
        
        return (stg_y, stg_x), stop
    
    def set_random_goal(self):
        """
        return a random goal in the map
        """
        obstacle_map = self.full_map.cpu().numpy()[0,0,::-1]
        goal = np.zeros_like(obstacle_map)
        goal_index = np.where((obstacle_map<1))
        np.random.seed(self.total_steps)
        if len(goal_index[0]) != 0:
            i = np.random.choice(len(goal_index[0]), 1)[0]
            h_goal = goal_index[0][i]
            w_goal = goal_index[1][i]
        else:
            h_goal = np.random.choice(goal.shape[0], 1)[0]
            w_goal = np.random.choice(goal.shape[1], 1)[0]
        goal[h_goal, w_goal] = 1
        return goal
    
    def visualize_agent_and_goal(self, map):
        # map = map.permute(1, 2, 0)
        def draw_agent(pose, agent_size, color_index, alpha=1):
            # map[:, int((self.map_size_cm/100-pose[1])*100/self.resolution)-agent_size:int((self.map_size_cm/100-pose[1])*100/self.resolution)+agent_size, int(pose[0]*100/self.resolution)-agent_size:int(pose[0]*100/self.resolution)+agent_size] = 0
            # map[color_index, int((self.map_size_cm/100-pose[1])*100/self.resolution)-agent_size:int((self.map_size_cm/100-pose[1])*100/self.resolution)+agent_size, int(pose[0]*100/self.resolution)-agent_size:int(pose[0]*100/self.resolution)+agent_size] = 1
            color_ori = map[:, int((self.map_size_cm/100-pose[1])*100/self.resolution)-agent_size:int((self.map_size_cm/100-pose[1])*100/self.resolution)+agent_size, int(pose[0]*100/self.resolution)-agent_size:int(pose[0]*100/self.resolution)+agent_size]
            color_new = torch.zeros_like(color_ori)
            color_new[color_index] = 1
            color_new = alpha * color_new + (1 - alpha) * color_ori
            map[:, int((self.map_size_cm/100-pose[1])*100/self.resolution)-agent_size:int((self.map_size_cm/100-pose[1])*100/self.resolution)+agent_size, int(pose[0]*100/self.resolution)-agent_size:int(pose[0]*100/self.resolution)+agent_size] = color_new

        def draw_goal(goal_size, color_index):
            skimage.morphology.disk(goal_size)
            if not self.found_goal and self.goal_loc is not None:
                map[:,int(self.map_size_cm/5)-self.goal_loc[0]-goal_size:int(self.map_size_cm/5)-self.goal_loc[0]+goal_size, self.goal_loc[1]-goal_size:self.goal_loc[1]+goal_size] = 0
                map[color_index,int(self.map_size_cm/5)-self.goal_loc[0]-goal_size:int(self.map_size_cm/5)-self.goal_loc[0]+goal_size, self.goal_loc[1]-goal_size:self.goal_loc[1]+goal_size] = 1
            else:
                map[:, int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)-goal_size:int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)+goal_size, int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)-goal_size:int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)+goal_size] = 0
                map[color_index, int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)-goal_size:int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)+goal_size, int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)-goal_size:int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)+goal_size] = 1

        for idx, pose in enumerate(self.history_pose):
            draw_step_num = 30
            alpha = max(0, 1 - (len(self.history_pose) - idx) / draw_step_num)
            agent_size = 1
            if idx == len(self.history_pose) - 1:
                agent_size = 2
            draw_agent(pose=pose, agent_size=agent_size, color_index=0, alpha=alpha)
        # return map  # added by someone for paper visualization
        # draw_agent(pose=self.full_pose, agent_size=2, color_index=0)
        # if self.found_goal:  # annotated by someone for paper visualization
        #     draw_goal(goal_size=4, color_index=0)
        draw_goal(goal_size=2, color_index=1)
        # map = map.permute(2, 0, 1)
        return map

    
    def visualize_obj_goal(self, map):
        agent_state = self.benchmark._env.sim.get_agent_state()

        origin = np.array(self.benchmark._env.current_episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(self.benchmark._env.current_episode.start_rotation)

        agent_position = agent_state.position
        goal_position_list = [goal.position for goal in self.benchmark._env.current_episode.goals]

        agent_position = quaternion_rotate_vector(rotation_world_start.inverse(), agent_position - origin)
        goal_position_list = [quaternion_rotate_vector(rotation_world_start.inverse(), goal_position - origin) for goal_position in goal_position_list]

        np.array([-agent_position[2], agent_position[0]], dtype=np.float32)
        goal_position_list = [np.array([-goal_position[2], goal_position[0]], dtype=np.float32) for goal_position in goal_position_list]

        goal_position_list = [[self.map_size_cm / 100.0 / 2.0 + goal_position[0], self.map_size_cm / 100.0 / 2.0 - goal_position[1]] for goal_position in goal_position_list]

        for goal_position in goal_position_list:
            if 0 <= int((self.map_size_cm/100-goal_position[1])*100/self.resolution) < map.shape[1] and 0 <= int(goal_position[0]*100/self.resolution) < map.shape[2]:
                map[:, int((self.map_size_cm/100-goal_position[1])*100/self.resolution)-2:int((self.map_size_cm/100-goal_position[1])*100/self.resolution)+2, int(goal_position[0]*100/self.resolution)-2:int(goal_position[0]*100/self.resolution)+2] = 0
                map[2, int((self.map_size_cm/100-goal_position[1])*100/self.resolution)-2:int((self.map_size_cm/100-goal_position[1])*100/self.resolution)+2, int(goal_position[0]*100/self.resolution)-2:int(goal_position[0]*100/self.resolution)+2] = 1
                # map[:, int((self.map_size_cm/100-goal_position[1])*100/self.resolution), int(goal_position[0]*100/self.resolution)] = 0
                # map[2, int((self.map_size_cm/100-goal_position[1])*100/self.resolution), int(goal_position[0]*100/self.resolution)] = 1

        return map

    def add_text(self, image: torch.Tensor, text_list: list, coordinate=(5, 5)):
        image = image.permute(1, 2, 0)
        image = (image * 255).byte().numpy()
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        # for i, t in enumerate(text):
        #     # font = ImageFont.truetype('arial.ttf', 36)
        #     text_position = (0, 40 * i)
        #     # text_color = (0, 0, 0)
        #     draw.text(text_position, t, fill='black')
        # font = ImageFont.truetype('arial.ttf', 36)
        text = ''
        for t in text_list:
            text = text + t + '\n'
        draw.text(coordinate, text, fill='black')
        image = np.array(image)
        image = torch.tensor(image)
        image = image / 255
        image = image.permute(2, 0, 1)
        return image
    
    def draw_bboxes_with_labels(self, rgb, bboxes, labels, color='red'):  
        # img = Image.open(image_path)  
        # rgb = np.transpose(rgb, (1, 2, 0))
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)  
        
        for bbox, label in zip(bboxes, labels):  
            x1, y1, x2, y2 = bbox  
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)  
            
            # text_width, text_height = draw.textsize(label)  
            font = ImageFont.load_default()  
            
            bbox = draw.textbbox((0, 0), label, font=font)  
            left, upper, right, lower = bbox  
            
            text_width = right - left  
            text_height = lower - upper  
            # margin = 10  
            # draw.rectangle([x1, y1 - text_height - margin, x1 + text_width, y1 - margin], fill=color)  
            # draw.text((x1, y1 - text_height - margin), label, fill='white')  
            draw.rectangle([x1, y1, x1 + text_width, y1 + text_height], fill=color)  
            draw.text((x1, y1), label, fill='white')  
        
        rgb = np.array(img)
        return rgb

    def draw_frontier_score(self, paper_map_trans, frontier_rgb, unknown_rgb):
        # frontier_map = torch.zeros_like(paper_map_trans)[:, :, 0]
        # color_map = torch.zeros_like(paper_map_trans)
        index_map = torch.zeros_like(paper_map_trans)[:, :, 0]
        scores = torch.tensor(self.scores)
        value, indices = torch.sort(scores)
        scores = scores[indices]
        frontier_locations_16 = self.frontier_locations_16[indices]
        wight_rgb = colors.to_rgb('white')
        wight_rgb = torch.tensor(wight_rgb).double()
        frontier_rgb = torch.tensor(frontier_rgb).double()
        thickness = 2
        selem = skimage.morphology.disk(thickness)
        def hsv_to_rgb(h, s, v):  
            h = h / 360.0  
            r, g, b = colorsys.hsv_to_rgb(h, s, v)  
            # r, g, b = int(r * 255), int(g * 255), int(b * 255)  
            return r, g, b  
        for i, frontier in enumerate(frontier_locations_16):
            # frontier[0] = frontier[0] + 10
            frontier[0] = self.map_size - 1 - frontier[0]
            # frontier_map[frontier[0], frontier[1]] = 1
            # v = 0.5 - 0.5 * i / len(frontier_locations_16)
            # combine_rgb = v * wight_rgb + (1 - v) * frontier_rgb
            h = 240 * (1 - i / len(frontier_locations_16))
            r, g, b = hsv_to_rgb(h, 1, 1)  
            combine_rgb = torch.tensor([r, g, b]).double()
            # color_map[color_map[0] - 4:color_map[0] + 4, color_map[1] - 4:color_map[1] + 4, :] = torch.tensor(combine_rgb).double()
            # index_map[color_map[0] - 4:color_map[0] + 4, color_map[1] - 4:color_map[1] + 4, :] = 1
            index_map = torch.zeros_like(paper_map_trans)[:, :, 0].numpy()
            index_map[frontier[0] - thickness:frontier[0] + thickness + 1, frontier[1] - thickness:frontier[1] + thickness + 1] = selem
            paper_map_trans[(index_map==1)*(paper_map_trans[:,:,0]==torch.tensor(unknown_rgb)[0]).numpy(), :] = combine_rgb
        # selem = skimage.morphology.disk(4)
        # frontier_map[skimage.morphology.binary_dilation(frontier_map, selem)] = 1
        # paper_map_trans[(frontier_map==1)*(paper_map_trans[:,:,0]==torch.tensor(unknown_rgb)[0]).numpy(),:] = color_map[(frontier_map==1)*(paper_map_trans[:,:,0]==torch.tensor(unknown_rgb)[0]).numpy(),:]
        return paper_map_trans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, choices=["local", "remote"]
    )
    parser.add_argument(
        "--PSL_infer", default="one_hot", type=str, choices=["optim", "one_hot"]
    )
    parser.add_argument(
        "--reasoning", default="obj", type=str, choices=["both", "room", "obj"]
    )
    parser.add_argument(
        "--llm", default="deberta", type=str, choices=["deberta", "chatgpt"]
    )
    parser.add_argument(
        "--error_analysis", default=False, type=bool, choices=[False, True]
    )
    parser.add_argument(
        "--visulize", action='store_true'
    )
    parser.add_argument(
        "--split_l", default=0, type=int
    )
    parser.add_argument(
        "--split_r", default=11, type=int
    )
    args = parser.parse_args()
    if args.error_analysis:
        os.environ["CHALLENGE_CONFIG_FILE"] = "configs/error_analysis_config.yaml"
    else:
        os.environ["CHALLENGE_CONFIG_FILE"] = "configs/challenge_objectnav2021.local.rgbd.yaml"
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    agent = CLIP_LLM_FMMAgent_NonPano(task_config=config, args=args)

    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False, split_l=args.split_l, split_r=args.split_r)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
