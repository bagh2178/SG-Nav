import cv2
import numpy as np
import torch
import skimage

def line_list(text, line_length=80):
    text_list = []
    for i in range(0, len(text), line_length):
        text_list.append(text[i:(i + line_length)])
    return text_list

def add_text(image: np.ndarray, text: str, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def add_text_list(image: np.ndarray, text_list: list, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    for i, text in enumerate(text_list):
        position_i = (position[0], position[1] + i * 15)
        cv2.putText(image, text, position_i, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def add_rectangle(image: np.ndarray, top_left: tuple, bottom_right: tuple, color=(0, 255, 0), thickness=2):
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image

def add_resized_image(base_image: np.ndarray, overlay_image: np.ndarray, position: tuple, size: tuple):
    resized_overlay = cv2.resize(overlay_image, size)

    h, w = resized_overlay.shape[:2]

    x, y = position

    if x + w > base_image.shape[1] or y + h > base_image.shape[0]:
        raise ValueError("Overlay image goes out of the bounds of the base image.")

    base_image[y:y+h, x:x+w] = resized_overlay
    return base_image

def crop_around_point(image: np.ndarray, point: tuple, size: tuple):
    img_height, img_width = image.shape[:2]
    
    crop_width, crop_height = size
    
    left = max(point[0] - crop_width // 2, 0)
    top = max(point[1] - crop_height // 2, 0)
    right = min(point[0] + (crop_width - crop_width // 2), img_width)
    bottom = min(point[1] + (crop_height - crop_height // 2), img_height)
    
    if right - left < crop_width:
        if left == 0:
            right = left + crop_width
        else:
            left = right - crop_width
    if bottom - top < crop_height:
        if top == 0:
            bottom = top + crop_height
        else:
            top = bottom - crop_height
    
    cropped_image = image[top:bottom, left:right]
    
    return cropped_image

def draw_agent(agent, map, pose, agent_size, color_index, alpha=1):
    color_ori = map[:, int((agent.map_size_cm/100-pose[1])*100/agent.resolution)-agent_size:int((agent.map_size_cm/100-pose[1])*100/agent.resolution)+agent_size, int(pose[0]*100/agent.resolution)-agent_size:int(pose[0]*100/agent.resolution)+agent_size]
    color_new = torch.zeros_like(color_ori)
    color_new[color_index] = 1
    color_new = alpha * color_new + (1 - alpha) * color_ori
    map[:, int((agent.map_size_cm/100-pose[1])*100/agent.resolution)-agent_size:int((agent.map_size_cm/100-pose[1])*100/agent.resolution)+agent_size, int(pose[0]*100/agent.resolution)-agent_size:int(pose[0]*100/agent.resolution)+agent_size] = color_new

def draw_goal(agent, map, goal_size, color_index):
    skimage.morphology.disk(goal_size)
    if not agent.found_goal and agent.goal_loc is not None:
        map[:,int(agent.map_size_cm/5)-agent.goal_loc[0]-goal_size:int(agent.map_size_cm/5)-agent.goal_loc[0]+goal_size, agent.goal_loc[1]-goal_size:agent.goal_loc[1]+goal_size] = 0
        map[color_index,int(agent.map_size_cm/5)-agent.goal_loc[0]-goal_size:int(agent.map_size_cm/5)-agent.goal_loc[0]+goal_size, agent.goal_loc[1]-goal_size:agent.goal_loc[1]+goal_size] = 1
    else:
        map[:, int((agent.map_size_cm/200+agent.goal_gps[1])*100/agent.resolution)-goal_size:int((agent.map_size_cm/200+agent.goal_gps[1])*100/agent.resolution)+goal_size, int((agent.map_size_cm/200+agent.goal_gps[0])*100/agent.resolution)-goal_size:int((agent.map_size_cm/200+agent.goal_gps[0])*100/agent.resolution)+goal_size] = 0
        map[color_index, int((agent.map_size_cm/200+agent.goal_gps[1])*100/agent.resolution)-goal_size:int((agent.map_size_cm/200+agent.goal_gps[1])*100/agent.resolution)+goal_size, int((agent.map_size_cm/200+agent.goal_gps[0])*100/agent.resolution)-goal_size:int((agent.map_size_cm/200+agent.goal_gps[0])*100/agent.resolution)+goal_size] = 1
