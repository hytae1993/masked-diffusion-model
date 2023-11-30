import torch
from torchvision import transforms
import torch.nn as nn

import numpy as np
import math
import random
from typing import Sequence


class Scheduler:
    def __init__(self, args):
        
        self.args                   = args
        self.height                 = args.data_size
        self.width                  = args.data_size
        self.image_size             = self.height * self.width
        
        self.updated_ddpm_num_steps = None
        self.ratio_list             = None
        self.black_area_pixels      = None
        
    def update_ddpm_num_steps(self, max_time):
        """
        Update new time list according to time scheduler.

        Parameters:
        - time: Max time fo ddpm steps
        
        Returns:
        - Updated ddpm time step.
        """
        time_list                   = list(range(1, self.image_size+1))
        
        if self.args.ddpm_schedule == 'linear':
            black_area_num_pixel        = self.get_extract_linear_random_sublist(time_list, self.args.ddpm_num_steps)
        
        elif self.args.ddpm_schedule == 'log_scale':
            black_area_num_pixel        = self.get_extract_log_scale_random_sublist(time_list, self.args.ddpm_num_steps)
            
        else:
            raise ValueError("Invalid mask ratio scheduler")
        
        black_area_num_pixel[-1]    = self.image_size # make sure the last T is remove all pixels  
        
        ddpm_num_steps          = len(black_area_num_pixel)
        self.ratio_list         = torch.tensor(black_area_num_pixel / self.image_size)
        self.black_area_pixels  = black_area_num_pixel
        
        self.updated_ddpm_num_steps = ddpm_num_steps
        
        return ddpm_num_steps
    
    
    def get_black_area_num_pixels_all(self):
        
        return self.black_area_pixels
    
    
    def get_updated_ddpm_num_steps(self):
        
        return self.updated_ddpm_num_steps
    
    
    def get_black_area_num_pixels_time(self, time):
        
        time    = (time-1).int()
        black_area_num_pixles_time = torch.index_select(torch.tensor(self.black_area_pixels, device=time.device), 0, time)
    
        return black_area_num_pixles_time
    
    
    def get_extract_linear_random_sublist(self, time_list, n):
        if n > len(time_list):
            raise ValueError("Desired to remove number of pixels is greater than the size of input image.")
        
        # Generate linear scale indices
        max_index       = len(time_list) - 1
        # linear_indices  = [int(i / self.image_size) for i in range(1, n+1)]
        linear_indices  = [int(self.image_size/n) * i for i in range(1, n+1)]
        
        unique_linear_indices = list(set(linear_indices))
        
        black_area_num_pixel = np.array([time_list[i] for i in sorted(unique_linear_indices)[:n]])
        return black_area_num_pixel
    
    def get_extract_log_scale_random_sublist(self, time_list, n):
        if n > len(time_list):
            raise ValueError("Desired to remove number of pixels is greater than the size of input image.")

        # Generate logarithmic scale indices
        max_index   = len(time_list) - 1
        log_indices = [int(round(10**random.uniform(0, math.log10(max_index)))) for _ in range(n)]

        # Ensure unique indices
        unique_log_indices = list(set(log_indices))
        
        # Take the first n unique log-scale indices
        black_area_num_pixel = np.array([time_list[i] for i in sorted(unique_log_indices)[:n]])
        return black_area_num_pixel
    
    
    def get_mask(self, black_area_num):
        """
        Generate random masks with black areas for each channel in the batch.

        Parameters:
        - black_area_ratios: List of ratios for the mask area to be black for each channel.

        Returns:
        - masks: Binary masks with black areas, shape (batch_size, 1, height, width).
        """
        masks = torch.ones((len(black_area_num), 1, self.height, self.width))

        for i in range(len(black_area_num)):
            num_black_pixels = black_area_num[i].int()

            black_pixels = random.sample(range(self.height * self.width), num_black_pixels)

            black_pixels = [(idx // self.width, idx % self.width) for idx in black_pixels]

            for j, k in black_pixels:
                masks[i, 0, j, k] = 0.0

        return masks
    
    
    def get_schedule_shift_time(self, timesteps: torch.IntTensor) -> torch.FloatTensor:
        random      = torch.FloatTensor(len(timesteps)).uniform_(-1.0, +1.0)
        random      = random.to(timesteps.device)
        timesteps   = timesteps.int()
        ratio       = torch.index_select(self.ratio_list.to(timesteps.device), 0, timesteps-1)
        shift_time  = random * ratio
        shift_time  = shift_time.to(self.args.weight_dtype)
        return shift_time
    
    
    def perturb_shift(self, data: torch.FloatTensor, shift: torch.FloatTensor):
        shift   = shift[:,None,None,None]
        shift   = shift.to(data.device)
        data    = data + shift 
        return data
    
    
    def perturb_shift_inverse(self, data: torch.FloatTensor, shift: torch.FloatTensor):
        shift   = shift[:,None,None,None]
        shift   = shift.to(data.device)
        data    = data - shift 
        return data