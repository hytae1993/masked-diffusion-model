import torch
from torchvision import transforms
import torch.nn as nn

import numpy as np
import math
import random
from typing import Sequence

import matplotlib.pyplot as plt


class Scheduler:
    def __init__(self, args):
        
        self.args                   = args
        self.height                 = args.data_size
        self.width                  = args.data_size
        self.image_size             = self.height * self.width
        
        self.updated_ddpm_num_steps = None
        self.ratio_list             = None
        self.black_area_pixels      = None
        
        # self.random      = torch.FloatTensor(1,3,self.height,self.width).uniform_(-1.0, +1.0)
        
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
            black_area_num_pixel        = self.get_extract_linear_random_sublist(self.args.ddpm_num_steps)
        
        elif self.args.ddpm_schedule == 'log':
            black_area_num_pixel        = self.get_extract_log_random_sublist(time_list, self.args.ddpm_num_steps)
            
        elif self.args.ddpm_schedule == 'exponential':
            black_area_num_pixel        = self.get_extract_exponential_random_sublist(self.args.ddpm_num_steps)
            
        elif self.args.ddpm_schedule == 'sigmoid':
            black_area_num_pixel        = self.get_extract_sigmoid_random_sublist(time_list, self.args.ddpm_num_steps)
            
        else:
            raise ValueError("Invalid mask ratio scheduler")
        
        if self.args.ddpm_schedule == 'log':
            black_area_num_pixel[-1]    = self.image_size # make sure the last T is remove all pixels  
            self.ratio_list         = torch.tensor(black_area_num_pixel / self.image_size)
        else:
            self.ratio_list         = black_area_num_pixel
            
        ddpm_num_steps          = len(black_area_num_pixel)
        self.reverse_ratio      = torch.flip(self.ratio_list, dims=(0,))
        self.black_area_pixels  = black_area_num_pixel        
        self.updated_ddpm_num_steps = ddpm_num_steps
        
        return ddpm_num_steps
    
    
    def get_black_area_num_pixels_all(self):
        
        return self.black_area_pixels
    
    
    def get_updated_ddpm_num_steps(self):
        
        return self.updated_ddpm_num_steps
    
    
    def get_ratio_list(self):
        
        return self.ratio_list
    
    
    def get_reverse_ratio_list(self):
        
        return self.reverse_ratio
    
    
    def get_black_area_num_pixels_time(self, time):
        
        try:
            time    = (time-1).int()
        except AttributeError:
            time    = time - 1
            
        if self.args.select_degrade_pixel == 'indexing':
            black_area_num_pixles_time  = torch.index_select(torch.tensor(self.black_area_pixels, device=time.device), 0, time)
        elif self.args.select_degrade_pixel == 'thresholding':
            black_area_num_pixles_time  = torch.index_select(torch.tensor(self.ratio_list, device=time.device), 0, time)
            
        return black_area_num_pixles_time
    
    
    def get_extract_linear_random_sublist(self, ddpm_num_steps):
        
        linear_schedule = np.linspace(1e-3, 1, ddpm_num_steps)
        
        normalized_schedule = torch.tensor(linear_schedule)
        
        return normalized_schedule
    
    
    def get_extract_log_random_sublist(self, time_list, ddpm_num_steps):
        if ddpm_num_steps > len(time_list):
            raise ValueError("Desired to remove number of pixels is greater than the size of input image.")

        x       = np.linspace(1, self.image_size, ddpm_num_steps)
        values  = np.log(x)
        
        values  = values - min(values) + 1
        values  = values * (self.image_size / max(values))
        values  = np.asarray(values, dtype=int)
    
        unique_values = list(sorted(set(values)))
        
        black_area_num_pixel    = np.array(unique_values)
        
        return black_area_num_pixel
    
    
    def get_extract_exponential_random_sublist(self, ddpm_num_steps):
        '''
        https://colab.research.google.com/drive/17Xwe4D50gJcfeyvD7LEbpspFWmNzIani?authuser=1#scrollTo=hXIq9ucgPkku
        '''
        linear_schedule = np.linspace(0, 1, ddpm_num_steps)
    
        exponential_schedule = self.args.ddpm_schedule_base ** linear_schedule # base: 10, 100, 1000
        
        # Normalize the schedule to end at 1, not start from 0
        normalized_schedule = exponential_schedule / exponential_schedule[-1]
        normalized_schedule = torch.tensor(normalized_schedule)
        
        return normalized_schedule

    def get_extract_sigmoid_random_sublist(self, time_list, ddpm_num_steps):
        if ddpm_num_steps > len(time_list):
            raise ValueError("Desired to remove number of pixels is greater than the size of input image.")
        
        # steepness_factor = 1.5
        base    = self.args.ddpm_schedule_base # default 1.5, define steepness of sigmoid
        result = []
        for i in range(ddpm_num_steps):
            x = 1 + (self.image_size - 1) * (1 / (1 + math.exp(-0.1 * base * (i - ddpm_num_steps / 2))))
            result.append(int(x))
        
        # Normalize the list to start from 1 and have a smooth increase
        min_val = min(result)
        result = [val - min_val + 1 for val in result]
        
        # Scale the values to end at n
        max_val = max(result)
        result = [val * self.image_size // max_val for val in result]
        
        # Ensure the first and last elements are exactly 1 and n
        result[0] = 1
        result[-1] = self.image_size
        
        black_area_num_pixel    = list(sorted(set(result)))
        black_area_num_pixel    = np.array(black_area_num_pixel)
        
        return black_area_num_pixel
    
    
    def get_timesteps_epoch(self, epoch, epoch_length):
        """
        Calculate the timesteps that is used for epoch
        Used for hierachically increasing timesteps
        If set scale to 1, use every timesteps at every epochs
        """
        scale       = self.args.scheduler_num_scale_timesteps
        timeindex   = [i for i in range(1,self.updated_ddpm_num_steps+1)]

        section     = math.ceil((epoch+1) / (epoch_length / scale))

        # num_trained_timesteps       = self.updated_ddpm_num_steps // np.power(2, scale-section)
        try:
            timesteps_used_epoch        = [timeindex[i-1] for i in range(1, self.updated_ddpm_num_steps+1) if i % np.power(2, scale-section) == 0]
        except ValueError:
            timesteps_used_epoch        = [timeindex[i-1] for i in range(1, self.updated_ddpm_num_steps+1) if i % np.power(2, 0) == 0]
            
        timesteps_used_epoch[-1]    = self.updated_ddpm_num_steps    # force last t matches T
        
        return timesteps_used_epoch

    
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
    
    
    def get_mean_mask(self, black_area_num, img, mean_option=None, index=None, mean_value=None):
        """
        Generate masks with mean of selected areas for each channel in the batch.

        Parameters:
        - black_area_num: number of pixels to degrade
        - img: input image
        - mean_option: how to fill the degraded pixels [mean, value]
        - idx: (optional)

        Returns:
        - noisy_img: Input image in which areas to be removed are filled with their mean value
        - mean_masks: Masks filled with average values for areas to be removed, shape (batch_size, 1, height, width).
        - black_idx: 
        """
        masks = torch.ones((len(black_area_num), img.shape[1], self.height, self.width)).to(img.device)
        
        black_idx   = []
        for i in range(len(black_area_num)):
            num_black_pixels = black_area_num[i].int()
            
            if index != None:
                black_pixels    = random.sample(index[i], num_black_pixels)
            
            else:
                black_pixels    = random.sample(range(self.height * self.width), num_black_pixels)
                black_pixels    = [(idx // self.width, idx % self.width) for idx in black_pixels]

            black_idx.append(black_pixels)

            for j, k in black_pixels:
                for l in range(img.shape[1]):
                    masks[i, l, j, k] = 0.0
                        
        sum_pixel   = (img * (1-masks)).sum(dim=(1,2,3), keepdim=True)
        if mean_value == None:
            mean_pixel  = sum_pixel / (1-masks).sum(dim=(1,2,3), keepdim=True)
        else: 
            mean_pixel  = mean_value
        
        noisy_img   = ((1-masks) * mean_pixel) + masks * img
        mean_masks  = ((1-masks) * mean_pixel) + masks
        
        return noisy_img, mean_masks, black_idx, mean_pixel
    
    
    def degrade_training(self, black_area_num, img, mean_option=None, mean_area=None):
        """
        Degrade input image with mask for 'training'

        Parameters:
        - black_area_num: number of pixels to degrade
        - img: input image
        - mean_option: how to fill the degraded pixels [mean, value]

        Returns:
        - noisy_img: Input image in which areas to be removed are filled with some value
        """
        if self.args.select_degrade_pixel == 'indexing':
            masks = torch.ones((len(black_area_num), self.height*self.width)).to(img.device)
            
            for i, num in enumerate(black_area_num):
                masks[i, torch.randperm(self.height*self.width)[:num]] = 0.0
            masks   = masks.reshape(len(black_area_num), 1, self.height, self.width)
            masks   = masks.expand_as(img)
        
        elif self.args.select_degrade_pixel == 'thresholding':
            if self.args.degrade_channel == '1-channel':
                masks   = torch.FloatTensor(img.shape[0], self.height*self.width).uniform_(0.0, +1.0).to(img.device)
                masks   = (masks > black_area_num.unsqueeze(dim=1)).float() * 1
                masks   = masks.reshape(len(black_area_num), 1, self.height, self.width)
                masks   = masks.expand_as(img)
                    
            elif self.args.degrade_channel == '3-channel':
                masks   = torch.FloatTensor(img.shape[0], 3*self.height*self.width).uniform_(0.0, +1.0).to(img.device)
                masks   = (masks > black_area_num.unsqueeze(dim=1)).float() * 1
                masks   = masks.reshape(len(black_area_num), 3, self.height, self.width)
                
        try:
            # mean_pixel = float(mean_option)
            mean_pixel  = torch.ones(len(black_area_num), img.shape[1], 1, 1).to(img.device) * float(mean_option)
        except ValueError:
            if mean_option == 'degraded_area':  # calculate with degraded pixels
                if mean_area == 'image-wise':
                    sum_pixel   = (img * (1-masks)).sum(dim=(1,2,3), keepdim=True)
                    mean_pixel  = sum_pixel / (1-masks).sum(dim=(1,2,3), keepdim=True)
                    
                elif mean_area == 'channel-wise':
                    sum_pixel   = (img * (1-masks)).sum(dim=(2,3), keepdim=True)
                    mean_pixel  = sum_pixel / (1-masks).sum(dim=(2,3), keepdim=True)
                
            elif mean_option == 'non_degraded_area':    # calculate with non-degraded area
                sum_pixel   = (img * masks).sum(dim=(2,3), keepdim=True)
                mean_pixel  = sum_pixel / (1-masks).sum(dim=(2,3), keepdim=True) * -1
                mean_pixel[torch.isnan(mean_pixel)] = 0.0
                
            elif mean_option == "0":
                mean_pixel  = torch.ones(len(black_area_num), img.shape[1], 1, 1).to(img.device) * float(mean_option)

        degrade_img     = ((1-masks) * mean_pixel) + masks * img
        degrade_mask    = ((1-masks) * mean_pixel) + masks
        mean_mask       = torch.ones((len(black_area_num), img.shape[1], self.height, self.width)).to(img.device) * mean_pixel
        
        return degrade_img, masks, degrade_mask, mean_mask
    
    
    def degrade_dependent_momentum_sampling(self, sample_t, sample_0, mean_option, index_start, index_end, index_list):
        """
        Degrade input image with single mask or multi masks for 'sampling'.
        Single mask is used for difference of two masks.
        Multi masks are used for multi degradation.

        Parameters:

        Returns:
        
        """
        masks_t   = torch.zeros(self.args.sample_num, self.height*self.width).to(sample_t.device)
        masks_0   = torch.zeros(self.args.sample_num, self.height*self.width).to(sample_0.device)
        mask      = torch.zeros(self.args.sample_num, self.height*self.width).to(sample_0.device)
        
        index_used          = index_list[:, :index_start]
        index_using         = index_list[:, index_start:index_end]
        index_total_used    = index_list[:, :index_end]
        
        masks_t.scatter_(1, index_used, 1)
        masks_0.scatter_(1, index_using, 1)
        mask.scatter_(1, index_total_used, 1)
        
        masks_t = masks_t.view(self.args.sample_num, -1, self.height, self.width)
        masks_0 = masks_0.view(self.args.sample_num, -1, self.height, self.width)
        mask    = mask.view(self.args.sample_num, -1, self.height, self.width)
        
        pixels_t    = sample_t * masks_t
        pixels_0    = sample_0 * masks_0
        preserved   = pixels_t + pixels_0
        
        try:
            mean_pixel = float(mean_option)
        except ValueError:
            if mean_option == 'degraded_area':  # calculate with degraded pixels
                # sum_pixel   = (img * (1-masks)).sum(dim=(1,2,3), keepdim=True)
                # mean_pixel  = sum_pixel / (1-masks).sum(dim=(1,2,3), keepdim=True)
                pass
            elif mean_option == 'non_degraded_area':    # calculate with non-degraded area
                sum_pixel   = (preserved * mask).sum(dim=(1,2,3), keepdim=True)
                mean_pixel  = sum_pixel / (1-mask).sum(dim=(1,2,3), keepdim=True) / sample_t.shape[1] * -1
                mean_pixel[torch.isnan(mean_pixel)] = 0.0
            elif mean_option == 0:
                mean_pixel  = torch.ones(self.args.sample_num, mask.shape[1], 1, 1).to(mask.device) * float(mean_option)
            elif mean_option == 'difference':
                pass
            
        noisy_img   = ((1-mask) * mean_pixel) + preserved
        mean_masks  = ((1-mask) * mean_pixel)
        
        return noisy_img, mean_masks, mean_pixel
    
    
    def degrade_index_sampling(self, index, black_area_num_t, img, mean_option=None, mean_area=None):
        masks   = torch.ones((self.args.sample_num), self.height*self.width).to(img.device)
        
        index_using = index[:, :black_area_num_t[0]]
        masks.scatter_(1, index_using, 0)
        
        masks   = masks.reshape(len(black_area_num_t), 1, self.height, self.width)
        masks   = masks.expand_as(img)
        
        try:
            mean_pixel  = torch.ones(len(black_area_num_t), img.shape[1], 1, 1).to(img.device) * float(mean_option)
        except ValueError:
            if mean_option == 'degraded_area':  # calculate with degraded pixels
                if mean_area == 'image-wise':
                    sum_pixel   = (img * (1-masks)).sum(dim=(1,2,3), keepdim=True)
                    mean_pixel  = sum_pixel / (1-masks).sum(dim=(1,2,3), keepdim=True)
                    
                elif mean_area == 'channel-wise':
                    sum_pixel   = (img * (1-masks)).sum(dim=(2,3), keepdim=True)
                    mean_pixel  = sum_pixel / (1-masks).sum(dim=(2,3), keepdim=True)
                
            elif mean_option == 'non_degraded_area':    # calculate with non-degraded area
                sum_pixel   = (img * masks).sum(dim=(2,3), keepdim=True)
                mean_pixel  = sum_pixel / (1-masks).sum(dim=(2,3), keepdim=True) * -1
                mean_pixel[torch.isnan(mean_pixel)] = 0.0
                
            elif mean_option == 0:
                mean_pixel  = torch.ones(len(black_area_num_t), img.shape[1], 1, 1).to(img.device) * float(mean_option)
                
            elif mean_option == 'difference':
                pass
            
        degrade_img     = ((1-masks) * mean_pixel) + masks * img
        degrade_mask    = masks
        mean_mask       = mean_pixel * torch.ones((len(black_area_num_t), masks.shape[1], self.height, self.width)).to(img.device)
        
        return degrade_img, degrade_mask, mean_mask
        
    
    def degrade_independent_base_sampling(self, black_area_num_t, img, mean_option=None, mean_area=None):
        """
        Degrade input image with single mask for base 'sampling'.
        Same as original ddpm sampling.
        Randomly select degraded area.

        Parameters:

        Returns:
        
        """
        
        if self.args.select_degrade_pixel == 'indexing':
            masks = torch.ones((len(black_area_num_t), self.height*self.width)).to(img.device)
            
            for i, num in enumerate(black_area_num_t):
                masks[i, torch.randperm(self.height*self.width)[:num]] = 0.0
            masks   = masks.reshape(len(black_area_num_t), 1, self.height, self.width)
            masks   = masks.expand_as(img)
        
        elif self.args.select_degrade_pixel == 'thresholding':
            if self.args.degrade_channel == '1-channel':
                masks   = torch.FloatTensor(img.shape[0], self.height*self.width).uniform_(0.0, +1.0).to(img.device)
                masks   = (masks > black_area_num_t.unsqueeze(dim=1)).float() * 1
                masks   = masks.reshape(len(black_area_num_t), 1, self.height, self.width)
                masks   = masks.expand_as(img)
                    
            elif self.args.degrade_channel == '3-channel':
                masks   = torch.FloatTensor(img.shape[0], 3*self.height*self.width).uniform_(0.0, +1.0).to(img.device)
                masks   = (masks > black_area_num_t.unsqueeze(dim=1)).float() * 1
                masks   = masks.reshape(len(black_area_num_t), 3, self.height, self.width)
                
        try:
            mean_pixel  = torch.ones(len(black_area_num_t), img.shape[1], 1, 1).to(img.device) * float(mean_option)
        except ValueError:
            if mean_option == 'degraded_area':  # calculate with degraded pixels
                if mean_area == 'image-wise':
                    sum_pixel   = (img * (1-masks)).sum(dim=(1,2,3), keepdim=True)
                    mean_pixel  = sum_pixel / (1-masks).sum(dim=(1,2,3), keepdim=True)
                    
                elif mean_area == 'channel-wise':
                    sum_pixel   = (img * (1-masks)).sum(dim=(2,3), keepdim=True)
                    mean_pixel  = sum_pixel / (1-masks).sum(dim=(2,3), keepdim=True)
                
            elif mean_option == 'non_degraded_area':    # calculate with non-degraded area
                sum_pixel   = (img * masks).sum(dim=(2,3), keepdim=True)
                mean_pixel  = sum_pixel / (1-masks).sum(dim=(2,3), keepdim=True) * -1
                mean_pixel[torch.isnan(mean_pixel)] = 0.0
                
            elif mean_option == 0:
                mean_pixel  = torch.ones(len(black_area_num_t), img.shape[1], 1, 1).to(img.device) * float(mean_option)
                
            elif mean_option == 'difference':
                pass
            
        degrade_img     = ((1-masks) * mean_pixel) + masks * img
        degrade_mask    = masks
        mean_mask       = mean_pixel * torch.ones((len(black_area_num_t), masks.shape[1], self.height, self.width)).to(img.device)
        
        return degrade_img, degrade_mask, mean_mask
    
    
    def degrade_dependent_base_sampling(self, black_area_num_t, black_area_num_next_t, img, mean_option, mean_area):
        """
        Degrade input image with single mask for base 'sampling'.
        Same as original ddpm sampling
        Select degraded area respect to index_list for dependent between all time.

        Parameters:

        Returns:
        
        """
        if self.args.select_degrade_pixel == 'indexing':
            pass
        
        elif self.args.select_degrade_pixel == 'thresholding':
            if self.args.degrade_channel == '1-channel':
                mask            = torch.FloatTensor(img.shape[0], self.height*self.width).uniform_(0.0, +1.0).to(img.device)
                
                masks_t         = (mask > black_area_num_t.unsqueeze(dim=1)).float() * 1
                masks_t         = masks_t.reshape(len(black_area_num_t), 1, self.height, self.width)
                masks_t         = masks_t.expand_as(img)
                
                masks_next_t    = (mask > black_area_num_next_t.unsqueeze(dim=1)).float() * 1
                masks_next_t    = masks_next_t.reshape(len(black_area_num_next_t), 1, self.height, self.width)
                masks_next_t    = masks_next_t.expand_as(img)
                    
            elif self.args.degrade_channel == '3-channel':
                mask            = torch.FloatTensor(img.shape[0], 3*self.height*self.width).uniform_(0.0, +1.0).to(img.device)
                
                masks_t         = (mask > black_area_num_t.unsqueeze(dim=1)).float() * 1
                masks_t         = masks_t.reshape(len(black_area_num_t), 3, self.height, self.width)
                
                masks_next_t    = (mask > black_area_num_next_t.unsqueeze(dim=1)).float() * 1
                masks_next_t    = masks_next_t.reshape(len(black_area_num_next_t), 3, self.height, self.width)
        
        
        if mean_option == 'degraded_area':  # calculate with degraded pixels
            if mean_area == 'image-wise':
                sum_pixel_t         = (img * (1-masks_t)).sum(dim=(1,2,3), keepdim=True)
                mean_pixel_t        = sum_pixel_t / (1-masks_t).sum(dim=(1,2,3), keepdim=True)
                
                sum_pixel_next_t    = (img * (1-masks_next_t)).sum(dim=(1,2,3), keepdim=True)
                mean_pixel_next_t   = sum_pixel_next_t / (1-masks_next_t).sum(dim=(1,2,3), keepdim=True)
                
            elif mean_area == 'channel-wise':
                sum_pixel_t         = (img * (1-masks_t)).sum(dim=(2,3), keepdim=True)
                mean_pixel_t        = sum_pixel_t / (1-masks_t).sum(dim=(2,3), keepdim=True)
                
                sum_pixel_next_t    = (img * (1-masks_next_t)).sum(dim=(2,3), keepdim=True)
                mean_pixel_next_t   = sum_pixel_next_t / (1-masks_next_t).sum(dim=(2,3), keepdim=True)
            
        elif mean_option == 'non_degraded_area':    # calculate with non-degraded area
            pass
            
        elif mean_option == "0":
            mean_pixel_t  = torch.ones(len(black_area_num_t), img.shape[1], 1, 1).to(img.device) * float(mean_option)
            mean_pixel_next_t  = torch.ones(len(black_area_num_t), img.shape[1], 1, 1).to(img.device) * float(mean_option)
            
        elif mean_option == 'difference':
            pass
        
        degrade_img_t           = ((1-masks_t) * mean_pixel_t) + masks_t * img
        degrade_mask_t          = masks_t
        mean_mask_t             = mean_pixel_t * torch.ones((len(black_area_num_t), masks_t.shape[1], self.height, self.width)).to(img.device)
        
        degrade_img_next_t      = ((1-masks_next_t) * mean_pixel_next_t) + masks_next_t * img
        degrade_mask_next_t     = masks_next_t
        mean_mask_next_t        = mean_pixel_next_t * torch.ones((len(black_area_num_t), masks_t.shape[1], self.height, self.width)).to(img.device)
    
        return degrade_img_t, degrade_mask_t, mean_mask_t, degrade_img_next_t, degrade_mask_next_t, mean_mask_next_t
    
    
    def degrade_interpolation_sampling(self, black_area_num_t, img, mean_option=None, mean_area=None):
        masks   = torch.FloatTensor(1, self.height*self.width).uniform_(0.0, +1.0).to(img.device)
        # masks   = torch.FloatTensor(len(black_area_num_t), self.height*self.width).uniform_(0.0, +1.0).to(img.device)
        masks   = (masks > black_area_num_t.unsqueeze(dim=1)).float() * 1
        masks   = masks.reshape(len(black_area_num_t), 1, self.height, self.width)
        masks   = masks.expand_as(img)
        
        try:
            mean_pixel  = torch.ones(len(black_area_num_t), img.shape[1], 1, 1).to(img.device) * float(mean_option)
        except ValueError:
            sum_pixel   = (img * (1-masks)).sum(dim=(1,2,3), keepdim=True)
            mean_pixel  = sum_pixel / (1-masks).sum(dim=(1,2,3), keepdim=True)
            
        degrade_img     = ((1-masks) * mean_pixel) + masks * img
        degrade_mask    = masks
        mean_mask       = mean_pixel * torch.ones((len(black_area_num_t), masks.shape[1], self.height, self.width)).to(img.device)
        
        return degrade_img, degrade_mask, mean_mask
    
    
    def degrade_with_mask(self, img, masks, mean_option, mean_area):
        try:
            mean_pixel  = torch.ones(len(masks), img.shape[1], 1, 1).to(img.device) * float(mean_option)
        except ValueError:
            if mean_option == 'degraded_area':  # calculate with degraded pixels
                if mean_area == 'image-wise':
                    sum_pixel   = (img * (1-masks)).sum(dim=(1,2,3), keepdim=True)
                    mean_pixel  = sum_pixel / (1-masks).sum(dim=(1,2,3), keepdim=True)
                    
                elif mean_area == 'channel-wise':
                    sum_pixel   = (img * (1-masks)).sum(dim=(2,3), keepdim=True)
                    mean_pixel  = sum_pixel / (1-masks).sum(dim=(2,3), keepdim=True)
                
            elif mean_option == 'non_degraded_area':    # calculate with non-degraded area
                sum_pixel   = (img * masks).sum(dim=(2,3), keepdim=True)
                mean_pixel  = sum_pixel / (1-masks).sum(dim=(2,3), keepdim=True) * -1
                mean_pixel[torch.isnan(mean_pixel)] = 0.0
                
            elif mean_option == 0:
                mean_pixel  = torch.ones(len(masks), img.shape[1], 1, 1).to(img.device) * float(mean_option)
                
            elif mean_option == 'difference':
                pass
        
        degrade_img     = ((1-masks) * mean_pixel) + masks * img
        
        return degrade_img
    
    # def get_schedule_shift_time(self, timesteps, binarymasks) -> torch. FloatTensor:
    #     random      = torch.FloatTensor(len(timesteps)).uniform_(-1.0, +1.0)
    #     random      = random.to(timesteps.device)
    #     timesteps   = timesteps.int()
    #     ratio       = torch.index_select(self.ratio_list.to(timesteps.device), 0, timesteps-1)
        
    #     shift_time  = random * ratio
    #     # reverse_ratio   = torch.index_select(torch.flip(self.ratio_list, [0]).to(timesteps.device), 0, timesteps-1)
    #     # shift_time  = random * reverse_ratio
    #     shift_time  = shift_time.to(self.args.weight_dtype)
    #     return shift_time
    
    def get_schedule_shift_time(self, timesteps: torch.IntTensor, binarymasks: torch.Tensor) -> torch.FloatTensor:
        
        timesteps   = timesteps.int()
        
        if self.args.shift_type == '1-d_constant':
            """
            1-d shift
            """
            random      = torch.FloatTensor(len(timesteps)).uniform_(-1.0, +1.0)
            random      = random.to(timesteps.device)
            timesteps   = timesteps.int()
            ratio       = torch.index_select(self.ratio_list.to(timesteps.device), 0, timesteps-1)
            
            shift_time  = random * ratio
            shift_time  = shift_time.to(self.args.weight_dtype)
            shift_time  = shift_time[:,None,None,None]
            
            ### using when the data is not normalized
            # clamp_min   = -1 * mu - ratio
            # clamp_max   = -1 * mu + ratio
            # shift_time  = torch.clamp(shift_time, min=clamp_min, max=clamp_max)
            # shift_time  = shift_time[:,None,None,None]
            
            # """
            # 1-d shift -> not using broadcasting
            # # """
            # random      = torch.ones(len(timesteps),3,self.height,self.width).to(timesteps.device)
            # ratio       = torch.index_select(self.ratio_list.to(timesteps.device), 0, timesteps-1)
            
            # if random.shape[0] == 1:
            #     random      = random * torch.FloatTensor(len(timesteps)).uniform_(-1.0, +1.0).to(timesteps.device)
            # else:
            #     random      = random * torch.FloatTensor(len(timesteps),1,1,1).uniform_(-1.0, +1.0).to(timesteps.device)
                
            # try:
            #     shift_time  = random * ratio
            # except RuntimeError:
            #     ratio       = ratio[:,None,None,None]
            #     ratio       = ratio.expand_as(random)
            #     shift_time  = random * ratio
            
        elif self.args.shift_type == '3-d_constant':
            """
            3-d shift
            """
            random      = torch.ones(len(timesteps),3,1,1).to(timesteps.device)
            random      = random * torch.FloatTensor(len(timesteps),3,1,1).uniform_(-1.0, +1.0).to(timesteps.device)
            random      = random.to(timesteps.device)
            timesteps   = timesteps.int()
            
            ratio       = torch.index_select(self.ratio_list.to(timesteps.device), 0, timesteps-1)
            ratio       = ratio[:,None,None,None]
            ratio       = ratio.expand_as(random)
            shift_time  = random * ratio
            shift_time  = shift_time.to(self.args.weight_dtype)
                
        
        elif self.args.shift_type == 'noise_reduction':
            """
            image-dimension noise shift
            reduce both mean and std according to timestep 
            """
            # random      = torch.FloatTensor(len(timesteps),1,self.height,self.width).uniform_(-1.0, +1.0) 
            random      = torch.FloatTensor(len(timesteps),1,self.height,self.width).normal_(mean=self.args.noise_mean, std=1).to(timesteps.device)
            ratio       = torch.index_select(self.ratio_list.to(timesteps.device), 0, timesteps-1)
            # ratio       = 1
            
            try:
                shift_time  = random * ratio
            except RuntimeError:
                ratio       = ratio[:,None,None,None]
                ratio       = ratio.expand_as(random)
                shift_time  = random * ratio
            
        elif self.args.shift_type == 'noise_std_reduction':
            """
            image-dimension noise shift
            reduce both std according to timestep 
            """
            ratio       = torch.index_select(self.ratio_list.to(timesteps.device), 0, timesteps-1)
            shift_time  = torch.zeros(len(timesteps), 3, self.height, self.width).to(timesteps.device)
            for i in range(len(timesteps)):
                shift_time[i]  = torch.FloatTensor(1,3,self.height,self.width).normal_(mean=self.args.noise_mean, std=1*ratio[i])
                
        elif self.args.shift_type == 'noise_with_perturbation':
            """
            image-dimension noise shift(gaussian distribution) and constant shift(uniform distribution)
            """
            perturbation    = torch.ones(len(timesteps),3,self.height,self.width).to(timesteps.device)
            
            if perturbation.shape[0] == 1:
                perturbation    = perturbation * torch.FloatTensor(len(timesteps)).uniform_(-1.0, +1.0).to(timesteps.device)
            else:
                perturbation    = perturbation * torch.FloatTensor(len(timesteps),1,1,1).uniform_(-1.0, +1.0).to(timesteps.device)
                
            random      = torch.FloatTensor(len(timesteps),3,self.height,self.width).normal_(mean=self.args.noise_mean, std=1).to(timesteps.device)
            shift_time  = perturbation + random
            
            ratio       = torch.index_select(self.ratio_list.to(timesteps.device), 0, timesteps-1)
            
            try:
                shift_time  = random * ratio
            except RuntimeError:
                ratio       = ratio[:,None,None,None]
                ratio       = ratio.expand_as(random)
                shift_time  = random * ratio
                
        elif self.args.shift_type == 'non_shift':
            shift_time  = torch.zeros(len(timesteps),3,self.height,self.width).to(timesteps.device)
            
                
        # reverse_ratio   = torch.index_select(torch.flip(self.ratio_list, [0]).to(timesteps.device), 0, timesteps-1)
        # shift_time  = random * reverse_ratio
        shift_time  = shift_time.to(self.args.weight_dtype)
        shift_time  = shift_time.expand_as(binarymasks)
        # shift_time  = shift_time * (1-binarymasks)  # shift only degraed area
        # shift_time  = shift_time * binarymasks    # shift only non-degraded area
        
        # shift_time  = shift_time - shift_time.mean() # make mask mean to zero
        
        return shift_time
    
    
    def get_schedule_shift_time_interpolation(self, timesteps: torch.IntTensor, mu: torch.Tensor, interpolation_shift) -> torch.FloatTensor:
        
        timesteps   = timesteps.int()
        
        shift       = torch.ones(len(timesteps)) * interpolation_shift
        shift       = shift.to(timesteps.device)
        timesteps   = timesteps.int()
        ratio       = torch.index_select(self.ratio_list.to(timesteps.device), 0, timesteps-1)
        
        shift_time  = shift * ratio
        shift_time  = shift_time.to(self.args.weight_dtype)
        
        clamp_min   = -1 * mu - ratio
        clamp_max   = -1 * mu + ratio
        shift_time  = torch.clamp(shift_time, min=clamp_min, max=clamp_max)
        
        shift_time  = shift_time[:,None,None,None]
        shift_time  = shift_time.to(self.args.weight_dtype)
        
        return shift_time
    
    
    def perturb_shift(self, data: torch.FloatTensor, shift: torch.FloatTensor):
        try: 
            shift   = shift.to(data.device)
            data    = data + shift 
            
        except RuntimeError:
            shift   = shift[:,None,None,None]
            shift   = shift.to(data.device)
            data    = data + shift 
        return data
    
    
    def perturb_shift_inverse(self, data: torch.FloatTensor, shift: torch.FloatTensor):
        try:
            shift   = shift.to(data.device)
            data    = data - shift 
        except RuntimeError:
            shift   = shift[:,None,None,None]
            shift   = shift.to(data.device)
            data    = data - shift 
        return data
    
    
    def get_weight_timesteps(self, timesteps: torch.IntTensor, power_base: torch.FloatTensor=2.0):
        alpha   = torch.linspace(start=1, end=0, steps=self.updated_ddpm_num_steps)
        # alpha   = torch.linspace(start=0, end=1, steps=self.updated_ddpm_num_steps)
        power   = torch.pow(power_base, alpha)  # power_base ^ alpha
        # plt.subplot(121)
        # plt.plot(self.black_area_pixels, 'r', label='time threshold')
        # plt.legend()
        # plt.subplot(122)
        # plt.plot(power, 'b', label='loss weight')
        # plt.legend()
        # plt.savefig('loss-weight.png')
        # exit(1)
        power   = power.to(timesteps.device)
        weight  = power[timesteps]
        return weight