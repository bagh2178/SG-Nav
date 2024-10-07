import skimage.morphology
import numpy as np
import copy

# def _which_direction(start_o):
#     xy = [0,0] #move forward to which direction
#     remainder =  start_o % (2*180) # Becomes positive number 
#     if remainder <= 1e-3 or abs(remainder - (2*180)) <=1: #no rotation
#         xy=[0,1]
#     elif abs(remainder - 180/2) <= 1 : #rotated once to the left #1.507
#         xy = [1,0]
#     elif abs(remainder + 180/2 - 2*180) <= 1:#rotated once to the right  #4.712
#         xy = [-1,0]
#     elif abs(remainder - 180)<= 1:
#         xy = [0,-1]
#     else:
#         raise Exception("start_o falls into nowhere!, start_o is " + str(start_o))
#     return xy

def _which_direction(y):
    """
    Given degrees y in [0, 359], return the closest cardinal direction as a tuple (x_dir, z_dir) in {-1, 0, 1}^2
    :param y: Input angle in [0, 359]
    """
    dir_degrees = [270, 180, 90, 0]
    closest_degree = dir_degrees[min(range(len(dir_degrees)), key=lambda i: abs(dir_degrees[i] - y))]
    if closest_degree == 270:  # facing x negative, z neutral
        s_rot = [-1, 0]
    elif closest_degree == 180:  # facing x neutral, z negative
        s_rot = [0, -1]
    elif closest_degree == 90:  # facing x positive, z neutral
        s_rot = [1, 0]
    else:  # facing x neutral, z positive
        s_rot = [0, 1]
    return s_rot

def _which_direction_angle(start_o):
    xy = [0,0] #move forward to which direction
    remainder =  start_o % (2*180) # Becomes positive number 
    if remainder <= 1e-3 or abs(remainder - (2*180)) <=1e-3: #no rotation
        xy=[0,1]
        angle = 90
    elif abs(remainder - 180/2) <= 1e-3 : #rotated once to the left #1.507
        xy = [1,0]
        angle = 0
    elif abs(remainder + 180/2 - 2*180) <= 1e-3:#rotated once to the right  #4.712
        xy = [-1,0]
        angle = 180
    elif abs(remainder - 180)<= 1e-3:
        xy = [0,-1]
        angle = 270
    else:
        raise Exception("start_o falls into nowhere!, start_o is " + str(start_o))
    return angle

def get_valid_directions(goal_loc):
    xys = [[0,1], [0,-1], [1,0], [-1,0]]
    crosses = [[1,0], [1,0], [0,1], [0,1]]
    goal_loci, goal_locj = goal_loc[0], goal_loc[1]
    rel_vec = np.array([121,121]) - np.array(goal_loc)
    rel_vec = rel_vec / math.sqrt(rel_vec[0]**2 + rel_vec[1]**2)
    xy_returns = []
    for xyi, xy in enumerate(xys):
        xy= np.array(xy) 
        cross = crosses[xyi]
        dot_p = rel_vec[0] * xy[0] + rel_vec[1] * xy[1]
        if dot_p >= 0:
            xy_returns.append(xy)
        else:
            pass
    return xy_returns

def _check_five_pixels_ahead_map_pred_for_moving(args, grid, start, xy):
    grid_copy = copy.deepcopy(grid)
    xy = np.array(xy)
    truefalse = True
    range_val = 8
    if args.side_step_step_size:
        range_val =args.side_step_step_size

    if xy[0] ==0:
        cross = [1,0]
    else:
        cross = [0,1]

    for i in range(1, range_val): #(1,6)?
        xy_new = xy * i
        if hasattr(args, 'sidestep_width'):
            for j in range(-args.sidestep_width, args.sidestep_width+1):
                grid_copy[j*cross[0] + start[0] + 1+ xy_new[0], j*cross[1] +start[1]+ 1+ xy_new[1]] = 0.5
                if grid[j*cross[0] + start[0] + 1+ xy_new[0], j*cross[1] +start[1]+ 1+ xy_new[1]] == 0:

                    truefalse = False
        else: 
            for j in range(-4, 5):
                grid_copy[j*cross[0] + start[0] + 1+ xy_new[0], j*cross[1] +start[1]+ 1+ xy_new[1]] = 0.5
                if grid[j*cross[0] + start[0] + 1+ xy_new[0], j*cross[1] +start[1]+ 1+ xy_new[1]] == 0:

                    truefalse = False
    return truefalse
    

def _add_cross_dilation_one_center(goal,  goal_loc, magnitude, additional_thickness): #goal_loc is (i,j)
    i = goal_loc[0]; j = goal_loc[1]
    wheres_i = [a for a in range(i-magnitude, i+magnitude+1)] + [i] * (2*magnitude+1)
    wheres_j = [j] * (2*magnitude+1) + [a for a in range(j-magnitude, j+magnitude+1)]
    for th in range(additional_thickness):
        i_new = i+th; j_new = j+th
        wheres_i += [a for a in range(i-magnitude, i+magnitude+1)] + [i_new] * (2*magnitude+1)
        wheres_j += [j_new] * (2*magnitude+1) + [a for a in range(j-magnitude, j+magnitude+1)]
        i_new = i-th; j_new = j-th
        wheres_i += [a for a in range(i-magnitude, i+magnitude+1)] + [i_new] * (2*magnitude+1)
        wheres_j += [j_new] * (2*magnitude+1) + [a for a in range(j-magnitude, j+magnitude+1)]
        
    wheres = (np.array(wheres_i), np.array(wheres_j))
    #remove those that exceed 0, 242
    wheres_i_new = []; wheres_j_new = []
    for i, j in zip(wheres[0], wheres[1]):
        if i >=0 and i < goal.shape[0] and j >=0 and j < goal.shape[1]:
            wheres_i_new.append(i); wheres_j_new.append(j)
    wheres = (np.array(wheres_i_new), np.array(wheres_j_new))
    goal[wheres] = 1
    return goal
            

def _add_cross_dilation(goal, magnitude, additional_thickness):
    goal_locs = np.where(goal!=0)
    for a in zip(goal_locs[0], goal_locs[1]):
        g = (a[0], a[1])
        goal = _add_cross_dilation_one_center(goal,  g, magnitude, additional_thickness)
    return goal


def _where_connected_to_curr_pose(start, traversible, seed, visited):
    non_traversible = 1- traversible*1
    if traversible[start[0]+1, start[1]+1] == 0:
        count = 0
        visited_place = np.where(visited ==1)
        while traversible[start[0]+1, start[1]+1] == 0 and count<100:
            np.random.seed(seed + count)
            start_idx = np.random.choice(len(visited_place[0]))
            start = (visited_place[0][start_idx], visited_place[1][start_idx])
            count +=1
        
    connected_regions = skimage.morphology.label(traversible, connectivity=2)
    where_start_connected = np.where(connected_regions == connected_regions[start[0]+1, start[1]+1])
    wc_wrong = len(where_start_connected[0]) < len(np.where(visited)[0]) or np.sum(traversible[where_start_connected]) < np.sum(non_traversible[where_start_connected])
    
    if wc_wrong: #error in connected region
        count = 0
        while wc_wrong and count <min(len(np.where(visited ==1)[0]), 100):
            start = (np.where(visited ==1)[0][count], np.where(visited ==1)[1][count])
            where_start_connected = np.where(connected_regions == connected_regions[start[0]+1, start[1]+1])
            wc_wrong = len(where_start_connected[0]) < len(np.where(visited)[0]) or np.sum(traversible[where_start_connected]) < np.sum(non_traversible[where_start_connected])
            count +=1
    return where_start_connected

def _planner_broken(fmm_dist, goal, traversible, start,  seed, visited):
    where_connected = _where_connected_to_curr_pose(start, traversible,  seed, visited)
    if fmm_dist[start[0]+1, start[1]+1] == fmm_dist.max():
        return True, where_connected
    else:
        return False, where_connected      
    
def _get_closest_goal(start, goal):
    real_start = [start[0]+1, start[1]+1]
    goal_locs = np.where(goal==1)
    dists = [(i-real_start[0])**2 + (j-real_start[1])**2 for i, j in zip(goal_locs[0], goal_locs[1])]
    min_loc = np.argmin(dists)
    return (goal_locs[0][min_loc], goal_locs[1][min_loc])

def _block_goal(centers, goal, original_goal, goal_found):
    goal[np.where(original_goal==1)] = 1
    return goal

def _get_center_goal(goal, pointer=None):
    connected_regions = skimage.morphology.label(goal, connectivity=2)
    unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
    new_goal = np.zeros(goal.shape)
    centers = []
    for lab in unique_labels:
        wheres = np.where(connected_regions == lab)
        wheres_center = (int(np.rint(np.mean(wheres[0]))), int(np.rint(np.mean(wheres[1]))))
        centers.append(wheres_center)
              
    for i, c in enumerate(centers):
        new_goal[c[0], c[1]] = 1
        
    return new_goal, centers

def _get_approximate_success(prev_rgb, frame, action):
    wheres = np.where(prev_rgb != frame)
    wheres_ar = np.zeros(prev_rgb.shape)
    wheres_ar[wheres] = 1
    wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
    connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
    unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
    max_area = -1
    for lab in unique_labels:
        wheres_lab = np.where(connected_regions == lab)
        max_area = max(len(wheres_lab[0]), max_area)
    if (action in ['OpenObject', 'CloseObject']) and max_area > 500:
        success = True
    elif max_area > 100:
        success = True
    else:
        success = False
    return success
        
def _append_to_actseq(success, actions, api_action):
     exception_actions = ["LookUp_30", "LookUp_0", "LookDown_30", "LookDown_0"]
     if not(api_action is None) :
         action_received = api_action['action']
         if not(action_received in exception_actions):
             actions.append(api_action)
         else:
             if action_received == "LookUp_30":
                 if success:
                     actions.append(dict(action="LookUp_15", forceAction=True))
                     actions.append(dict(action="LookUp_15", forceAction=True))
             elif action_received == "LookDown_30":
                 if success:
                     actions.append(dict(action="LookDown_15", forceAction=True))
                     actions.append(dict(action="LookDown_15", forceAction=True))
             else: #pass for lookup0, lookdown 0
                 pass
     return actions