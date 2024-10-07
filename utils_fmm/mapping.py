import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np

from utils_fmm.model import get_grid, ChannelPool
import utils_fmm.depth_utils as du

import cv2
import time



class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, agent, max_height=None, min_height=None, num_cats=1):
        super(Semantic_Mapping, self).__init__()

        self.device = agent.device
        self.screen_h = agent.config.SIMULATOR.DEPTH_SENSOR.HEIGHT
        self.screen_w = agent.config.SIMULATOR.DEPTH_SENSOR.WIDTH
        self.agent_height = agent.config.SIMULATOR.AGENT_0.HEIGHT*100.
        self.resolution = 5
        self.z_resolution = 5
        self.map_size_cm = agent.map_size_cm
        self.vision_range = 100
        self.dropout = 0.5
        self.fov = agent.config.SIMULATOR.DEPTH_SENSOR.HFOV
        self.du_scale = 1
        self.exp_pred_threshold = 10
        self.map_pred_threshold = 10
        self.max_z_consider = self.agent_height + 1 + 50
        self.min_z_consider = 20
        if max_height is not None:
            self.max_z_consider = max_height
            self.min_z_consider = min_height
            
        self.view_angles = [0.0] #TODO: what is the view angle? 
        
        self.max_height = int(250 / self.z_resolution)
        self.min_height = int(-150 / self.z_resolution)
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi/2.0]
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, self.fov)

        vr = 100

        if num_cats is None:
            num_cats = 1
        self.init_grid = torch.zeros(1, num_cats, vr, vr,
                                self.max_height - self.min_height).float().to(self.device)
        self.feat = torch.ones(1,num_cats,
                          self.screen_h//self.du_scale * self.screen_w//self.du_scale
                         ).float().to(self.device)
        
    def set_view_angles(self, view_angle):
        self.view_angles[0] = -view_angle


    def forward(self, depth, pose_obs, maps_last, type_mask=None, type_prob=None):
        if type_mask is not None:
            return self.forward_(depth, pose_obs, maps_last, type_mask, type_prob)
        depth = torch.unsqueeze(depth, 0) * 100
        pose_obs = torch.unsqueeze(pose_obs, 0)

        point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)
        #Multiprocessing
        agent_view_t = du.transform_camera_view_t_multiple(point_cloud_t, self.agent_height, self.view_angles, self.device)

        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)

        def pose_transform(pose_obs, agent_view_centered_t):
            pose_obs = pose_obs.clone()
            agent_view_centered_t = agent_view_centered_t.clone()
            pose_obs[0, 0] = pose_obs[0, 0] * 100
            pose_obs[0, 1] = pose_obs[0, 1] * 100
            pose_obs[0, 2] = pose_obs[0, 2] - 90
            pose_obs[0, 2] = pose_obs[0, 2] / 57.29577951308232
            x = pose_obs[0, 0].item()
            y = pose_obs[0, 1].item()
            t = pose_obs[0, 2].item()
            pose_matrix = torch.tensor([
                [np.cos(t), -np.sin(t), 0, x],
                [np.sin(t), np.cos(t), 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=agent_view_centered_t.dtype, device=agent_view_centered_t.device)
            agent_view_centered_t = agent_view_centered_t.permute(3, 0, 1, 2)
            ones = torch.ones(agent_view_centered_t.shape[1:], device=agent_view_centered_t.device).unsqueeze(0)
            agent_view_centered_t = torch.cat([agent_view_centered_t, ones], dim=0)
            shape = agent_view_centered_t.shape
            agent_view_centered_t = agent_view_centered_t.reshape(4, -1)
            point_cloud_world = pose_matrix @ agent_view_centered_t
            point_cloud_world = point_cloud_world.reshape(shape)
            point_cloud_world = point_cloud_world[:3]
            point_cloud_world = point_cloud_world.permute(1, 2, 3, 0)
            return point_cloud_world
        
        point_cloud_world = pose_transform(pose_obs, agent_view_centered_t)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] / xy_resolution)
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] - vision_range//2.)/vision_range*2.
        XYZ_cm_std[...,2] = XYZ_cm_std[...,2] / z_resolution
        XYZ_cm_std[...,2] = (XYZ_cm_std[...,2] -
                             (max_h+min_h)//2.)/(max_h-min_h)*2.


        XYZ_cm_std = XYZ_cm_std.permute(0,3,1,2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(self.init_grid*0., self.feat, XYZ_cm_std).transpose(2,3)

        min_z = int(self.min_z_consider/z_resolution - min_h)
        max_z = int(self.max_z_consider/z_resolution - min_h) 

        agent_height_proj = voxels[...,min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:,0:1,:,:]
        fp_exp_pred = all_height_proj[:,0:1,:,:]
        fp_map_pred = fp_map_pred/self.map_pred_threshold
        fp_exp_pred = fp_exp_pred/self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        agent_view = torch.zeros(1, 1,
                                 self.map_size_cm//self.resolution,
                                 self.map_size_cm//self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm//(self.resolution * 2) - self.vision_range//2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm//(self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:,0, y1:y2, x1:x2] = fp_map_pred
        # agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:,1] += rel_pose_change[:,0] * \
                            torch.sin(pose[:,2]/57.29577951308232) \
                        + rel_pose_change[:,1] * \
                            torch.cos(pose[:,2]/57.29577951308232)
            pose[:,0] += rel_pose_change[:,0] * \
                            torch.cos(pose[:,2]/57.29577951308232) \
                        - rel_pose_change[:,1] * \
                            torch.sin(pose[:,2]/57.29577951308232)
            pose[:,2] += rel_pose_change[:,2]*57.29577951308232

            pose[:,2] = torch.fmod(pose[:,2]-180.0, 360.0)+180.0
            pose[:,2] = torch.fmod(pose[:,2]+180.0, 360.0)-180.0

            return pose

        current_poses = pose_obs  # get_new_pose_batch(poses_last, pose_obs)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                                * 100.0/self.resolution
                                - self.map_size_cm//(self.resolution*2)) /\
                                (self.map_size_cm//(self.resolution*2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                        self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)
        map_pred, _ = torch.max(maps2,1)

        map_pred[map_pred > 0.5] = 1

        return map_pred

    def forward_(self, depth, pose_obs, maps_last, type_mask=None, type_prob=None):
        depth = torch.unsqueeze(depth, 0) * 100
        type_mask = torch.unsqueeze(type_mask, 0)
        pose_obs = torch.unsqueeze(pose_obs, 0)

        bs, c, h, w = type_mask.size()
        point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)
        #Multiprocessing
        agent_view_t = du.transform_camera_view_t_multiple(point_cloud_t, self.agent_height, self.view_angles, self.device)

        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] / xy_resolution)
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] - vision_range//2.)/vision_range*2.
        XYZ_cm_std[...,2] = XYZ_cm_std[...,2] / z_resolution
        XYZ_cm_std[...,2] = (XYZ_cm_std[...,2] -
                             (max_h+min_h)//2.)/(max_h-min_h)*2.

        self.feat = nn.AvgPool2d(self.du_scale)(type_mask[:,:,:,:]
                        ).view(bs, c, h//self.du_scale * w//self.du_scale) # scaled semantic segmentation

        XYZ_cm_std = XYZ_cm_std.permute(0,3,1,2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(self.init_grid*0., self.feat, XYZ_cm_std).transpose(2,3)

        min_z = int(self.min_z_consider/z_resolution - min_h)
        max_z = int(self.max_z_consider/z_resolution - min_h) 

        agent_height_proj = voxels[...,min_z:max_z].sum(4)

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm//self.resolution,
                                 self.map_size_cm//self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm//(self.resolution * 2) - self.vision_range//2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm//(self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:,:, y1:y2, x1:x2] = torch.clamp(
                        agent_height_proj,
                        min=0.0, max=1.0)

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:,1] += rel_pose_change[:,0] * \
                            torch.sin(pose[:,2]/57.29577951308232) \
                        + rel_pose_change[:,1] * \
                            torch.cos(pose[:,2]/57.29577951308232)
            pose[:,0] += rel_pose_change[:,0] * \
                            torch.cos(pose[:,2]/57.29577951308232) \
                        - rel_pose_change[:,1] * \
                            torch.sin(pose[:,2]/57.29577951308232)
            pose[:,2] += rel_pose_change[:,2]*57.29577951308232

            pose[:,2] = torch.fmod(pose[:,2]-180.0, 360.0)+180.0
            pose[:,2] = torch.fmod(pose[:,2]+180.0, 360.0)-180.0

            return pose

        current_poses = pose_obs  # get_new_pose_batch(poses_last, pose_obs)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                                * 100.0/self.resolution
                                - self.map_size_cm//(self.resolution*2)) /\
                                (self.map_size_cm//(self.resolution*2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                        self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        for i in range(c):
            translated[0,i][translated[0,i]>0] = type_prob[i]
        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1) # n*2*w*h
        map_pred, _ = torch.max(maps2,1)

        return map_pred
