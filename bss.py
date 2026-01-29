import random
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import random
def select_random_point_pairs(H, W, num_points, num_points_per_pair=2, boundary_margin=35, point_margin=40, max_attempts=1000):
    """
    Randomly select point pairs, returning a list of length num_points, where each element is a tuple containing num_points_per_pair points.
    All points must be at least boundary_margin pixels away from the image boundaries.
    Any two points within each pair satisfy |x1 - x2| > point_margin and |y1 - y2| > point_margin.
    Points can be repeated across pairs, but are unique within each pair.
    If unable to find enough points (after max_attempts attempts), return the found ones (may be fewer).
    Parameters:
        H (int): Image height
        W (int): Image width
        num_points (int): Length of the returned point pairs list
        num_points_per_pair (int): Number of points in each pair (default: 2)
        boundary_margin (int): Boundary margin (default: 35)
        point_margin (int): Minimum distance margin between points (default: 40)
        max_attempts (int): Maximum attempts to avoid infinite loop (default: 1000)
    Returns:
        list: List containing num_points point pairs, each pair is a tuple with num_points_per_pair (y, x) points
    """
    # Handle special cases
    if num_points_per_pair == 0:
        return [()] * num_points
    point_pairs = []
    for _ in range(num_points):
        pair_points = []
        attempts = 0
        while len(pair_points) < num_points_per_pair and attempts < max_attempts:
            # Randomly generate points, respecting boundaries
            y = np.random.randint(boundary_margin, H - boundary_margin)
            x = np.random.randint(boundary_margin, W - boundary_margin)
            # Check distance constraints with all existing points in the pair
            is_valid = True
            for py, px in pair_points:
                if not (abs(y - py) > point_margin and abs(x - px) > point_margin):
                    is_valid = False
                    break
            if is_valid:
                pair_points.append((y, x))
            attempts += 1
        # If reached the required number, add; otherwise add partial (or skip, depending on needs)
        if pair_points:
            point_pairs.append(tuple(pair_points))
    # If fewer than num_points pairs, return the available ones
    return point_pairs[:num_points]
class BSS(MIFGSM):
    """
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of scaled copies in each iteration.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model.
        s_d (float): 
            The minimum scale ratio for scaling the image during the attack. It is used to control the lower bound of scaling for the transformation.
        s_u (float): 
            The maximum scale ratio for scaling the image during the attack. It is used to control the upper bound of scaling for the transformation.
        boundary_margin (int): 
            The margin (in pixels) around the image that will not be used for selecting points for the attack. This ensures that points are selected away from the edges of the image.
        point_margin (int): 
            The minimum distance margin between points in each pair of points that are selected for transformation. This helps ensure that the selected points are sufficiently far apart from each other.
        num_points_per_pair (int): 
            The number of points in each pair (default: 2). The function allows selecting pairs of points, and this parameter controls how many points are in each selected pair.
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, But rounded to 2/255, epoch=10, decay=1., s_d=0.25, s_u=0.75,boundary_margin=35,num_points_per_pair = 2,point_margin = 40
    """
    def __init__(self, model_name, epsilon=16 / 255, alpha=2 / 255, epoch=10, decay=1., targeted=False,
                 random_start=False, norm='linfty', loss='crossentropy', device=None, attack='BSS', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay=decay, targeted=targeted, random_start=random_start,
                         norm=norm, loss=loss, device=device, attack=attack, **kwargs)
        self.num_scale = 10 # ie number scale
        self.s_d = 0.25 # ie r=1
        self.s_u = 0.75 # ie r=1
        self.boundary_margin=35 #ie db=35
        self.point_margin = 40 #ie dp=40
        self.num_points_per_pair = 2 #ie M=2
        self.point_pairs=[]
    def scale_single_dim_point_pair(self, x, dim, points):
        """
        Image block cutting + scaling along the specified dimension + interpolation filling + block-level Admix fusion
        Keep blocks in place, only scale, no rearrangement
        Parameters:
            x (torch.Tensor): Input image [B, C, H, W]
            dim (int): Dimension to cut and scale (2=H, 3=W)
            points (list/tuple): Split point coordinates (y,x)
        Returns:
            list: List of scaled image blocks (in original order)
        """
        length = x.size(dim)
        # Extract and sort split coordinates, remove duplicates
        coords = sorted(list(set(p[dim - 2] for p in points if 0 <= p[dim - 2] < length)))
        split_points = [0] + coords + [length]
        lengths = [split_points[i + 1] - split_points[i] for i in range(len(split_points) - 1)]
        # Filter zero-length splits
        valid_lengths = [l for l in lengths if l > 0]
        valid_indices = [i for i, l in enumerate(lengths) if l > 0]
        if not valid_lengths:
            # If no valid splits, return original
            return [x]
        num_blocks = len(valid_lengths) # Dynamically use valid block count
        # 1. Cut blocks
        x_strips = list(x.split(lengths, dim=dim))
        valid_strips = [x_strips[i] for i in valid_indices]
        # Generate random weights to add variability and avoid uniform splits, using s_d and s_u
        rand = torch.empty(num_blocks, device=self.device).uniform_(self.s_d, self.s_u)
        rand_norm = rand / rand.sum()
        # Keep original order, no sorting weights
        scaled_floats = length * rand_norm # Apply weights directly, no rearrangement
        # Round and correct sum to length
        scaled_lengths = torch.round(scaled_floats).int()
        scaled_lengths = torch.clamp(scaled_lengths, min=1)
        total = scaled_lengths.sum().item()
        diff = length - total
        if diff > 0:
            for _ in range(diff):
                max_idx = scaled_lengths.argmax().item()
                scaled_lengths[max_idx] += 1
        elif diff < 0:
            for _ in range(-diff):
                valid_mask = scaled_lengths > 1
                if not valid_mask.any():
                    break
                min_idx = (scaled_lengths + (~valid_mask).int() * 1e9).argmin().item()
                scaled_lengths[min_idx] -= 1
        # Assign scaled lengths to strips, process in original order
        x_scaled = []
        for idx in valid_indices: # Use original index order
            target_len = scaled_lengths[valid_indices.index(idx)].item()
            strip = valid_strips[valid_indices.index(idx)]
            if strip.size(dim) == 0:
                shape = list(strip.shape)
                shape[dim] = 1
                strip = torch.zeros(shape, dtype=strip.dtype, device=strip.device)
            if dim == 2: # Height scaling
                new_h, new_w = target_len, strip.size(3)
            else: # Width scaling
                new_h, new_w = strip.size(2), target_len
            strip_scaled = F.interpolate(strip, size=(new_h, new_w), mode='bilinear', align_corners=False)
            x_scaled.append(strip_scaled)
        return x_scaled
    def scale(self, x, point_pair,):
        """
        Split the input tensor into a 3x3 grid based on coordinate pairs and randomly scale.
        Parameters:
            x (torch.Tensor): Input tensor (B, C, H, W)
            point_pair (list): List of coordinate pairs, like [((x1, y1), (x2, y2)), ...]
        Returns:
            torch.Tensor: Scaled tensor
        """
        dims = [2, 3]
        random.shuffle(dims)
        # Process along the first dimension
        strips1 = self.scale_single_dim_point_pair(x, dims[0], point_pair)
        result_strips = []
        for strip in strips1:
            # Process along the second dimension
            strips2 = self.scale_single_dim_point_pair(strip, dims[1], point_pair)
            # Concat along dims[1]
            result_strips.append(torch.cat(strips2, dim=dims[1]))
        # Concat along dims[0]
        return torch.cat(result_strips, dim=dims[0])
    def transform(self, x, **kwargs):
        """
        Transform the input image by scaling based on attention center and edge points,
        then apply patch masking.
        Arguments:
            x (torch.Tensor): Input image tensor (B, C, H, W)
        """
        if x.shape[0] != 1:
            raise ValueError("Expected batch size of 1")
        _, _, h, w = x.shape
        # Use random point pairs to increase split variability
        self.point_pairs = select_random_point_pairs(h, w, num_points=self.num_scale, num_points_per_pair=self.num_points_per_pair,
                                                     point_margin=self.point_margin,boundary_margin=self.boundary_margin)
        x_t = torch.cat([self.scale(x, point_pair) for point_pair in self.point_pairs])
        return x_t
    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits,
                                                                                                           label.repeat(
                                                                                                               self.num_scale))