import numpy as np
import torch


def expand_3d_box(bbox: torch.Tensor, eps=0.02) -> torch.Tensor:
    '''
    Expand the side of 3D boxes such that each side has at least eps length.
    Assumes the bbox cornder order in open3d convention. 
    
    bbox: (N, 8, D)
    
    returns: (N, 8, D)
    '''
    center = bbox.mean(dim=1)  # shape: (N, D)

    va = bbox[:, 1, :] - bbox[:, 0, :]  # shape: (N, D)
    vb = bbox[:, 2, :] - bbox[:, 0, :]  # shape: (N, D)
    vc = bbox[:, 3, :] - bbox[:, 0, :]  # shape: (N, D)
    
    a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    
    va = torch.where(a < eps, va / a * eps, va)  # shape: (N, D)
    vb = torch.where(b < eps, vb / b * eps, vb)  # shape: (N, D)
    vc = torch.where(c < eps, vc / c * eps, vc)  # shape: (N, D)
    
    new_bbox = torch.stack([
        center - va/2.0 - vb/2.0 - vc/2.0,
        center + va/2.0 - vb/2.0 - vc/2.0,
        center - va/2.0 + vb/2.0 - vc/2.0,
        center - va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 + vc/2.0,
        center - va/2.0 + vb/2.0 + vc/2.0,
        center + va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 - vc/2.0,
    ], dim=1) # shape: (N, 8, D)
    
    new_bbox = new_bbox.to(bbox.device)
    new_bbox = new_bbox.type(bbox.dtype)
    
    return new_bbox


def compute_3d_iou_accuracte_batch(bbox1, bbox2):
    '''
    Compute IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.
    
    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)
    
    import pytorch3d.ops as ops

    bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    
    inter_vol, iou = ops.box3d_overlap(bbox1.float(), bbox2.float())
    
    return iou


def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute IoU between two sets of axis-aligned 3D bounding boxes.
    
    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1) # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1) # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1) # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1) # Shape: (N, 3)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

    # Compute volume of intersection box
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return iou


def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    '''
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.
     
    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2
        
    Returns:
        mask_sub: (N, H, W), binary mask
    '''
    N = xyxy.shape[0] # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)
    
    inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1] # (N, N)
    
    inter_over_box1 = inter_areas / areas[:, None] # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T # (N, N)
    
    # if the intersection area is smaller than th2 of the area of box1, 
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1) # (N, N)
    contained_idx = contained.nonzero() # (num_contained, 2)

    mask_sub = mask.copy() # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

    return mask_sub