import cv2
import numpy as np
import skfmm
import skimage
from numpy import ma
import pickle
import os
import copy

def get_mask(sx, sy, scale, step_size):
    """
    using all the points on the edges as mask.
    """
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    """
    mask[0, size//2] = 1
    mask[-1, size//2] = 1
    mask[size//2, 0] = 1
    mask[size//2, -1] = 1

    mask[0, size//4] = 1
    mask[-1, size//4] = 1
    mask[size//4, 0] = 1
    mask[size//4, -1] = 1

    mask[0, 3*size//4] = 1
    mask[-1, 3*size//4] = 1
    mask[3*size//4, 0] = 1
    mask[3*size//4, -1] = 1

    mask[0, 0] = 1
    mask[-1, -1] = 1
    mask[0, -1] = 1
    mask[-1, 0] = 1
    """
    mask[0,:] = mask[-1,:] = 1
    mask[:,0] = mask[:,-1] = 1
    mask[size//2, size//2] = 1
    return mask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
                mask[i, j] = max(5, (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                 ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


def moving_avg(a, n=2):
    h = a.shape[0]
    w = a.shape[1]
    b = copy.deepcopy(a[n:h-n,n:w-n])
    for i in range(1,n+1):
        for j in range(1,i+1):
            b += a[n+j:h-n+j,n+(i-j):w-n+(i-j)] / (800*i)
            b += a[n-(i-j):h-n-(i-j),n+j:w-n+j]/ (800*i)
            b += a[n-j:h-n-j,n-(i-j):w-n-(i-j)]/ (800*i)
            b += a[n+(i-j):h-n+(i-j),n-j:w-n-j]/ (800*i)
    b /= 1+n/200

    return b

class FMMPlanner():
    def __init__(self, traversible, args, scale=1, step_size=5):
        self.scale = scale
        self.args = args
        self.step_size = step_size
        self.visualize = False
        self.stop_cond = 0.5
        self.save = False
        self.save_t = 0
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale,
                                           traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.))

    def set_goal(self, goal, auto_improve=False):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.)), \
                         int(goal[1] / (self.scale * 1.))

        if self.traversible[goal_x, goal_y] == 0.:
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1) # time consuming
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return

    def set_multi_goal(self, goal_map, state):
        traversible_ma = ma.masked_values(self.traversible * 1, 0) # travrsible 1 and false
        goal_x, goal_y = np.where(goal_map==1)
        if self.traversible[goal_x, goal_y] == 0.: 
            ## if goal is not traversible, find a traversible place nearby as goal
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y], state)
            
        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        
        dd = ma.filled(dd, np.max(dd) + 1) # fill untraversible place as max distance
        if dd[state[0],state[1]] == np.max(dd): # agent is in a untraversible place (not supposed to happen)
            goal_map_ma = np.zeros_like(goal_map) == 0
            goal_map_ma[goal_map == 1] = 0
            dd += skfmm.distance(goal_map_ma, dx=1)
        self.fmm_dist = dd
   
        return

    def get_short_term_goal(self, state, found_goal = 0, decrease_stop_cond=0):
        scale = self.scale * 1.
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = get_mask(dx, dy, scale, self.step_size)
        dist_mask = get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]
        n = 2
        dist = np.pad(self.fmm_dist, self.du + n,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        subset = dist[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]
        subset_large = dist[state[0]:state[0] + 2 * self.du + 1+2*n,
                      state[1]:state[1] + 2 * self.du + 1+2*n]
        subset = moving_avg(subset_large, n=n)
        assert subset.shape[0] == 2 * self.du + 1 and \
               subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2
        
        stop_condition = max((self.stop_cond - decrease_stop_cond)*100/5., 0.2)

        if self.visualize:
            print("dist until goal is ", subset[self.du, self.du])
        if subset[self.du, self.du] < stop_condition:
            stop = True
        else:
            stop = False

        subset -= subset[self.du, self.du] # dis change wrt agent point
        #ratio1 = subset / dist_mask
        #subset[ratio1 < -1.5] = 1

        ## see which direction has the fastest distance decrease to the goal
        mid = self.du
        for i in range(len(subset)-1):
            subset[0,i] /= np.sqrt(np.abs(mid-i)**2+mid**2) / mid
        for i in range(len(subset)-1):
            subset[i,-1] /= np.sqrt(np.abs(mid-i)**2+mid**2) / mid
        for i in range(len(subset)-1):
            subset[-1,-i-1] /= np.sqrt(np.abs(mid-i)**2+mid**2) / mid
        for i in range(len(subset)-1):
            subset[-i-1,0] /= np.sqrt(np.abs(mid-i)**2+mid**2) / mid
        
        # do not move accross the obstacles
        
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        if subset[stg_x, stg_y] > -0.0001:
            replan = True
        else:
            replan = False


        return (stg_x + state[0] - self.du) * scale, \
               (stg_y + state[1] - self.du) * scale, replan, stop

    def _find_nearest_goal(self, goal, state=None):
        """
        find the nearest traversible place
        """
        max_x, max_y = self.traversible.shape
        top_left_selected = (max(0,goal[0]-80), max(0,goal[1]-80))
        down_right_selected = (min(max_x-1, goal[0]+80), min(max_y-1, goal[1]+80))
        
        traversible = np.ones((int(down_right_selected[0]-top_left_selected[0]),int(down_right_selected[1]-top_left_selected[1]))) * 1.0
        planner = FMMPlanner(traversible, self.args)
        goal = (goal[0]-top_left_selected[0],goal[1]-top_left_selected[1])
        planner.set_goal(goal)
        mask = self.traversible[int(top_left_selected[0]):int(down_right_selected[0]), int(top_left_selected[1]):int(down_right_selected[1])]

        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max() * 2
        dist_sort_idx = np.argsort(dist_map, axis=None) # a little time cosuming 
        i = 0
        goal = np.unravel_index(dist_sort_idx[i], dist_map.shape)
        goal = (top_left_selected[0]+goal[0],top_left_selected[1]+goal[1])
        return goal
        