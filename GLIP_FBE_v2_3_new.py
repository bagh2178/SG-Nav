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
from PIL import Image
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete

import habitat
from habitat.config import Config
from habitat.core.agent import Agent
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from GLIP.maskrcnn_benchmark.config import cfg as glip_cfg
from utils_glip import *

from utils_fmm.fmm_planner import FMMPlanner
from utils_fmm.mapping import Semantic_Mapping
import utils_fmm.control_helper as CH
import utils_fmm.pose_utils as pu


class CLIP_LLM_FMMAgent_NonPano(Agent):
    """
    New in this version: 
    1. frontier distance change from eular distance to fmm distance
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
        self.context = 'room' # 'obj', 'room', ''

        # init glip model
        config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml"
        weight_file = "GLIP/MODEL/glip_large_model.pth"
        # config_file = "GLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
        # weight_file = "GLIP/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"
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

        self.map_size_cm = 4000
        self.resolution = self.map_resolution = 5
        self.camera_horizon = 0
        self.dilation_deg = 0
        self.collision_threshold = 0.08
        self.col_width = 5
        self.selem = skimage.morphology.square(1)
        self.init_map()
        self.sem_map_module = Semantic_Mapping(self).to(self.device) 
        self.free_map_module = Semantic_Mapping(self, max_height=10,min_height=-150).to(self.device)
        # self.observed_map_module = Semantic_Mapping(self, max_height=120,min_height=-150).to(self.device)
        self.free_map_module.eval()
        self.free_map_module.set_view_angles(self.camera_horizon)
        # self.observed_map_module.eval()
        # self.observed_map_module.set_view_angles(self.camera_horizon)
        self.sem_map_module.eval()
        self.sem_map_module.set_view_angles(self.camera_horizon)
        

        print('FMM navigate init finish!!!')

        #####
        self.frontiers_total = 0
        self.random_total = 0

    def reset(self):
        self.navigate_steps = 0
        self.turn_angles = 0
        self.move_steps = 0
        self.total_steps = 0
        self.current_room_search_step = 0
        self.found_goal = False
        self.goal_loc = None
        self.correct_room = False
        self.changing_room = False
        self.changing_room_steps = 0
        self.ever_long_goal = False
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
        ###########
        self.current_obj_predictions = []
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
    
    def detect_objects(self, observations):
        self.current_obj_predictions = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], object_captions) # self.glip_panorama(object_captions) # time cosuming
        new_labels = self.get_glip_real_label(self.current_obj_predictions)
        self.current_obj_predictions.add_field("labels", new_labels)
        shortest_distance = 120 # TODO: shortest distance or most confident?
        shortest_distance_angle = 0
        goal_prediction = copy.deepcopy(self.current_obj_predictions)
        obj_labels = self.current_obj_predictions.get_field("labels")
        goal_bbox = []
        for j, label in enumerate(obj_labels):
            if self.obj_goal in label:
                goal_bbox.append(self.current_obj_predictions.bbox[j])
            elif self.obj_goal == 'gym_equipment' and (label in ['treadmill', 'exercise machine']):
                goal_bbox.append(self.current_obj_predictions.bbox[j])
        if len(goal_bbox) > 0:
            long_goal_detected_before = copy.deepcopy(self.found_long_goal)
            goal_prediction.bbox = torch.stack(goal_bbox)
            for box in goal_prediction.bbox:
                box = box.to(torch.int64)
                center_point = (box[:2] + box[2:]) // 2
                temp_direction = (center_point[0] - 320) * 79 / 640
                temp_distance = self.depth[center_point[1],center_point[0],0]
                k = 0
                pos_neg = 1
                while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                    pos_neg *= -1
                    k += 0.5
                    temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                    self.depth[center_point[1],center_point[0]+int(pos_neg*k),0])
                if temp_distance >= 4.999:
                    self.found_long_goal = True
                    self.ever_long_goal = True
                # new_goal_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                else:
                    self.found_goal = True
                    self.found_long_goal = False

                direction = temp_direction
                distance = temp_distance
                if distance < shortest_distance:
                    shortest_distance = distance
                    shortest_distance_angle = direction
                    box_shortest = copy.deepcopy(box)
        
            if self.found_goal:
                self.goal_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            elif not long_goal_detected_before:
                self.long_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            if self.args.error_analysis and self.found_goal:
                if (observations['semantic'][box_shortest[0]:box_shortest[2],box_shortest[1]:box_shortest[3]] == self.goal_mp3d_idx).sum() > min(300, 0.2 * (box_shortest[2]-box_shortest[0])*(box_shortest[3]-box_shortest[1])):
                     self.detect_true = True
        
        
    def act(self, observations):
        """ 
        observations: 
        """ 
        if self.total_steps >= 482:
            return {"action": 0}
        self.total_steps += 1
        if self.navigate_steps == 0:
            self.obj_goal = projection[int(observations["objectgoal"])]

        observations["depth"][observations["depth"]==0.5] = 100 # don't construct unprecise map 
        self.depth = observations["depth"]
        self.rgb = observations["rgb"][:,:,[2,1,0]]
        
        self.update_map(observations)
        self.update_free_map(observations)
        # self.update_observed_map(observations)
        # look down twice and look around at first
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
            if not self.found_goal:
                return {"action": 6}
                    
        
        if not (observations["gps"] == self.last_gps).all():
            self.move_steps += 1
            self.not_move_steps = 0
            if self.using_random_goal:
                self.move_since_random += 1
        else:
            self.not_move_steps += 1
        
        self.last_gps = observations["gps"]
        
        if not self.found_goal:
            self.detect_objects(observations)
          
        # generate action using FMM
        # print(observations["gps"], observations["compass"], self.goal_gps)
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)
        if self.found_goal: 
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(799,int(self.map_size_cm/10+self.goal_gps[1]*100/self.resolution))), max(0,min(799,int(self.map_size_cm/10+self.goal_gps[0]*100/self.resolution)))] = 1
        elif self.found_long_goal: 
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(799,int(self.map_size_cm/10+self.long_goal_temp_gps[1]*100/self.resolution))), max(0,min(799,int(self.map_size_cm/10+self.long_goal_temp_gps[0]*100/self.resolution)))] = 1
        elif not self.first_fbe:
            self.goal_loc = self.fbe(traversible, cur_start)
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.not_use_random_goal()
            self.first_fbe = True
            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
        
        stg_y, stg_x, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        if self.found_long_goal and number_action == 0: # didn't detect goal when going to long goal, start over FBE. 
            self.found_long_goal = False
          
        if (not self.found_goal and not self.found_long_goal and number_action == 0) or (self.using_random_goal and self.move_since_random > 20):
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
            
        loop_time = 0
        while (not self.found_goal and number_action == 0) or self.not_move_steps >= 7:
            loop_time += 1
            self.random_this_ex += 1
            if loop_time > 20:
                return {"action": 0}
            self.not_move_steps = 0
            self.goal_map = self.set_random_goal()
            self.using_random_goal = True
            self.random_total += 1
            stg_y, stg_x, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        # ------------------------------
        
        # save_map = copy.deepcopy(torch.from_numpy(traversible))
        # gray_map = torch.stack((save_map, save_map, save_map))
        # gray_map[:, int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)-2:int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)+2, int(self.full_pose[0]*100/self.resolution)-2:int(self.full_pose[0]*100/self.resolution)+2] = 0
        # gray_map[0, int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)-2:int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)+2, int(self.full_pose[0]*100/self.resolution)-2:int(self.full_pose[0]*100/self.resolution)+2] = 1
        # if self.using_random_goal:
        #     goal_loc = np.where(self.goal_map > 0)
        #     gray_map[:, goal_loc] = 0
        #     gray_map[1, goal_loc] = 1
        # if not self.found_goal and self.goal_loc is not None:
        #     gray_map[:,int(self.map_size_cm/5)-self.goal_loc[0]-2:int(self.map_size_cm/5)-self.goal_loc[0]+2, self.goal_loc[1]-2:self.goal_loc[1]+2] = 0
        #     gray_map[1,int(self.map_size_cm/5)-self.goal_loc[0]-2:int(self.map_size_cm/5)-self.goal_loc[0]+2, self.goal_loc[1]-2:self.goal_loc[1]+2] = 1
        # else:
        #     gray_map[:, int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)-2:int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)+2, int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)-2:int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)+2] = 0
        #     gray_map[1, int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)-2:int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)+2, int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)-2:int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)+2] = 1
        # gray_map[:, int(stg_y)-2:int(stg_y)+2, int(stg_x)-2:int(stg_x)+2] = 0
        # gray_map[2, int(stg_y)-2:int(stg_y)+2, int(stg_x)-2:int(stg_x)+2] = 1
       
        # save_image(torch.from_numpy(observations["rgb"]/255).float().permute(2,0,1), 'figures/rgb/img'+str(self.navigate_steps)+'.png')
        # dist= torch.stack((torch.from_numpy(self.planner.fmm_dist), torch.from_numpy(self.planner.fmm_dist), torch.from_numpy(self.planner.fmm_dist)))
        # save_image((dist / dist.max()), 'figures/dist/img'+str(self.navigate_steps)+'.png')
        # save_image((gray_map / gray_map.max()), 'figures/map/img'+str(self.navigate_steps)+'.png')
        # free_map = self.fbe_free_map.cpu().numpy()[0,0,::-1].copy()
        # save_image((torch.from_numpy(free_map) / free_map.max()), 'figures/free_map/img'+str(self.navigate_steps)+'.png')
        
        
        observations["pointgoal_with_gps_compass"] = self.get_relative_goal_gps(observations)

        ###-----------------------------------###

        self.last_loc = copy.deepcopy(self.full_pose)
        self.prev_action = number_action
        if number_action == 0:
            a = 1
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
            # labels = [self.entities[i - self.plus] for i in labels ]
        else:
            new_labels = ['object' for i in labels]
        return new_labels
    
    
    def fbe(self, traversible, start):
        # fontier: unknown area and free area
        # unknown area: not free and not obstacle 
        # find nearest frontier and return the gps
        fbe_map = torch.zeros_like(self.full_map[0,0])
        fbe_map[self.fbe_free_map[0,0]>0] = 1 # first free 
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle
    
        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp
        frontier_map = diff == 1
        #dist= torch.stack((frontier_map, frontier_map, frontier_map)).cpu().numpy()[:,::-1].copy()
        #save_image((torch.from_numpy(dist) / dist.max()), 'figures/frontiers/img'+str(self.navigate_steps)+'.png')
        # get clostest frontier from the agent
        frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
        if len(torch.where(frontier_map)[0]) == 0:
            return None
        # calculate the distance from every locations to agent
        planner = FMMPlanner(traversible, None)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        idx_16 = np.where(distances>=1.6)
        distances_16 = distances[idx_16]
        
        if len(distances_16) == 0:
            return None
        #sample = np.random.exponential(scale=2) + 1.6
        
        #idx_16_min = idx_16[0][np.argmin(np.abs(distances_16-sample))]
        idx_16_min = idx_16[0][np.argmin(distances_16)]
        #idx_16_min = idx_16[0][random.randint(0,len(distances_16)-1)]
        goal = frontier_locations[idx_16_min] -1
        return goal
        
        
    def get_goal_gps(self, observations, angle, distance):
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
        self.visited = self.full_map[0,0].cpu().numpy()
        self.explored_map = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()
        self.fbe_observed_map = copy.deepcopy(self.full_map).to(self.device) 
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
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.fbe_free_map = self.free_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.fbe_free_map)
        self.fbe_free_map[int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4, int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4] = 1
    
    def get_traversible(self, map_pred, pose_pred):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, choices=["local", "remote"]
    )
    parser.add_argument(
        "--error_analysis", default=False, type=bool, choices=[False, True]
    )
    args = parser.parse_args()
    if args.error_analysis:
        os.environ["CHALLENGE_CONFIG_FILE"] = "configs/error_analysis_config.yaml"
    else:
        os.environ["CHALLENGE_CONFIG_FILE"] = "configs/challenge_objectnav2021.local.rgbd_self_setting.yaml"
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    for dir in os.listdir('figures'):
        if os.path.isfile(os.path.join('figures', dir)):
            os.remove(os.path.join('figures', dir))
        else:
            for file in os.listdir(os.path.join('figures', dir)):
                os.remove(os.path.join('figures', dir, file))
    agent = CLIP_LLM_FMMAgent_NonPano(task_config=config, args=args)

    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()