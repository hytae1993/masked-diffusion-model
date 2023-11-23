import torch
from torchvision import transforms
import torch.nn as nn

import numpy as np
import math
import random
from typing import Sequence


class Scheduler:
    def __init__(self, args):
        # self.batch_size     = batch_size
        # self.dim_channel    = dim_channel
        # self.height     = height
        # self.width      = width
        
        # self.alpha_shift        = alpha_shift
        # self.alpha_scale        = alpha_scale
        # self.alpha_shift_param  = alpha_shift_param
        # self.alpha_scale_param  = alpha_scale_param
        # self.distribution_shift = distribution_shift
        # self.distribution_scale = distribution_scale
        # self.channelwise_shift  = channelwise_shift
        # self.channelwise_scale  = channelwise_scale
        
        self.args                   = args
        # self.ddpm_schedule          = ddpm_schedule
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
        
        
        
        if self.args.ddpm_schedule == 'linear':
            time_list   = list(range(1, max_time+1))
            time_list   = list(range(1, self.image_size+1))
            ratio       = time_list / max_time
            
        elif self.args.ddpm_schedule == 'simple_log_scale':
            time_list   = list(range(1, max_time+1))
            base        = 1.1
            ratio       = np.power(base, time_list)
            ratio       = ratio / ratio.max()
            
            black_area_ratio_pixel  = self.height * self.width * ratio
            black_area_ratio_pixel  = np.ceil(black_area_ratio_pixel)
            black_area_ratio_pixel  = np.unique(black_area_ratio_pixel)
            
            ddpm_num_steps          = len(black_area_ratio_pixel)
            
            self.ratio_list         = black_area_ratio_pixel / self.image_size
            self.black_area_pixels  = black_area_ratio_pixel
        
        elif self.args.ddpm_schedule == 'log_scale':
            time_list                   = list(range(1, self.image_size+1))
            black_area_num_pixel        = self.get_extract_log_scale_random_sublist(time_list, self.args.ddpm_num_steps)
            black_area_num_pixel[-1]    = self.image_size # make sure the last T is remove all pixels           
            
            ddpm_num_steps          = len(black_area_num_pixel)
            self.ratio_list         = black_area_num_pixel / self.image_size
            self.black_area_pixels  = black_area_num_pixel
            
        else:
            raise ValueError("Invalid mask ratio scheduler")
        
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
    
    
    def get_extract_log_scale_random_sublist(self, time_list, n):
        if n > len(time_list):
            raise ValueError("Desired to remove number of pixels is greater than the size of input image.")

        # Generate logarithmic scale indices
        max_index = len(time_list) - 1
        log_indices = [int(round(10**random.uniform(0, math.log10(max_index)))) for _ in range(n)]

        # Ensure unique indices
        unique_log_indices = list(set(log_indices))
        
        # Take the first n unique log-scale indices
        black_area_num_pixel = np.array([time_list[i] for i in sorted(unique_log_indices)[:n]])
        return black_area_num_pixel
    
    
    # def get_black_area_ratio_func(self, time):
        
    #     if self.args.ddpm_schedule == 'linear':
    #         # black_area_ratio    = list(map(int, timesteps / (self.args.ddpm_num_steps+1)))
    #         ratio    = time / self.updated_ddpm_num_steps
            
    #     elif self.args.ddpm_schedule == 'log_scale':
    #         base     = 1.1
    #         ratio    = np.power(base, time.cpu())
    #         ratio    = ratio / np.power(base, self.updated_ddpm_num_steps)
    #     else:
    #         raise ValueError("Invalid mask ratio scheduler")
        
    #     black_area_ratio    = self.height * self.width * ratio
    #     black_area_ratio    = np.ceil(black_area_ratio) 
    
    
    # def get_list_black_area_ratios(self, time):
    #     """
    #     Generate a scheduled time list. Each time means the ratio to be removed from original input image.

    #     Parameters:
    #     - time: Max time fo ddpm steps
        
    #     Returns:
    #     - black_area_ratio: scheduled time list.
    #     """
        
    #     if self.args.ddpm_schedule == 'linear':
    #         # black_area_ratio    = list(map(int, timesteps / (self.args.ddpm_num_steps+1)))
    #         ratio    = time / self.updated_ddpm_num_steps
            
    #     elif self.args.ddpm_schedule == 'log_scale':
    #         base     = 1.1
    #         ratio    = np.power(base, time.cpu())
    #         ratio    = ratio / np.power(base, self.updated_ddpm_num_steps)
    #     else:
    #         raise ValueError("Invalid mask ratio scheduler")
        
    #     black_area_ratio    = self.height * self.width * ratio
    #     black_area_ratio    = np.ceil(black_area_ratio)
        
    #     return black_area_ratio
    
    
    

    

    def get_alpha_scalar(self, time_index: torch.Tensor, time_length: torch.Tensor, option: str='linear', option_param: float=0.0):
        if option == 'linear':
            alpha = time_index / (time_length-1)
            alpha = alpha.sqrt()
        elif option == 'linear_inverse':
            alpha = time_index / (time_length-1)
            alpha = 1.0 - alpha
            alpha = alpha.sqrt()
        elif option == 'sigmoid':
            k = option_param * torch.ones_like(time_index)
            x = time_index / (time_length-1)
            x = 2.0 * x - 1 # [-1, +1]
            y = -x * k      # [+k, -k]
            alpha = 1.0 / (1.0 + y.exp())
            alpha_min = 1.0 / (1.0 + torch.exp(k))
            alpha_max = 1.0 / (1.0 + torch.exp(-k))
            alpha = (alpha - alpha_min) / (alpha_max - alpha_min)
        elif option == 'constant':
            alpha = option_param * torch.ones_like(time_index)
        elif option == 'one':
            alpha = torch.ones_like(time_index)
        elif option == 'zero':
            alpha = torch.zeros_like(time_index)
        else:
            print('error: not implemented')
        return alpha

    
    def get_alpha(self, batch_size: int, dim_channel: int, dim_height: int, dim_width: int, time_index: torch.Tensor, time_length: torch.Tensor, option: str='linear', option_param: float=0.0):
        alpha = self.get_alpha_scalar(time_index, time_length, option, option_param)
        alpha = alpha.reshape(batch_size, 1, 1, 1)
        alpha = alpha.repeat(1, dim_channel, dim_height, dim_width)
        return alpha


    def get_latent(self, batch_size: int, dim_channel: int, dim_height: int, dim_width: int, channelwise: bool=False, option: str='uniform', value_min: float=-1.0, value_max: float=+1.0):
        if option == 'normal':
            if channelwise:
                latent = torch.randn(size=(batch_size, dim_channel, 1, 1))
                latent = latent.repeat(1, 1, dim_height, dim_width)
            else:
                latent = torch.randn(size=(batch_size, 1, 1, 1))
                latent = latent.repeat(1, dim_channel, dim_height, dim_width)
            latent = latent.expand(batch_size, dim_channel, dim_height, dim_width)
        elif option == 'uniform':
            if channelwise:
                latent = torch.FloatTensor(batch_size, dim_channel, 1, 1).uniform_(value_min, value_max)
                latent = latent.repeat(1, 1, dim_height, dim_width)
            else:
                latent = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(value_min, value_max)
                latent = latent.repeat(1, dim_channel, dim_height, dim_width)
            latent = latent.expand(batch_size, dim_channel, dim_height, dim_width)
        elif option == 'linear':
            latent = torch.linspace(value_min, value_max, batch_size)
            latent = latent.reshape(batch_size, 1, 1, 1)
            latent = latent.repeat(1, dim_channel, dim_height, dim_width)
            latent = latent.expand(batch_size, dim_channel, dim_height, dim_width)
        elif option == 'zero':
            latent = torch.zeros(batch_size, dim_channel, dim_height, dim_width)
        elif option == 'one':
            latent = torch.ones(batch_size, dim_channel, dim_height, dim_width)
        else:
            print('error: not implemented')
        return latent


    def get_latent_interpolate(self, batch_size: int, dim_channel: int, dim_height: int, dim_width: int, channelwise: bool=False, option: str='uniform', value_min: float=-1.0, value_max: float=+1.0):
        if option == 'normal':
            ''' 
            if channelwise:
                latent = torch.zeros(size=(batch_size, dim_channel))
                for i in range(dim_channel):
                    (value1, value2) = torch.randn(2)
                    latent[:,i] = torch.linspace(value1, value2, batch_size)
                latent = latent.reshape(batch_size, dim_channel, 1, 1)
                latent = latent.repeat(1, 1, dim_height, dim_width)
            else:
                (value1, value2) = torch.randn(2)
                latent = torch.linspace(value1, value2, batch_size)
                latent = latent.reshape(batch_size, 1, 1, 1)
                latent = latent.repeat(1, dim_channel, dim_height, dim_width)
            latent = latent.expand(batch_size, dim_channel, dim_height, dim_width)
            '''
            latent = torch.linspace(value_min, value_max, batch_size)
            latent = latent.reshape(batch_size, 1, 1, 1)
            latent = latent.repeat(1, dim_channel, dim_height, dim_width)
            latent = latent.expand(batch_size, dim_channel, dim_height, dim_width)   
        elif option == 'uniform':
            '''
            if channelwise:
                latent = torch.zeros(size=(batch_size, dim_channel))
                for i in range(dim_channel):
                    (value1, value2) = torch.FloatTensor(2).uniform_(value_min, value_max)
                    latent[:,i] = torch.linspace(value1, value2, batch_size)
                latent = latent.reshape(batch_size, dim_channel, 1, 1)
                latent = latent.repeat(1, 1, dim_height, dim_width)
            else:
                (value1, value2) = torch.FloatTensor(2).uniform_(value_min, value_max)
                latent = torch.linspace(value1, value2, batch_size)
                latent = latent.reshape(batch_size, 1, 1, 1)
                latent = latent.repeat(1, dim_channel, dim_height, dim_width)
            latent = latent.expand(batch_size, dim_channel, dim_height, dim_width)
            '''
            latent = torch.linspace(value_min, value_max, batch_size)
            latent = latent.reshape(batch_size, 1, 1, 1)
            latent = latent.repeat(1, dim_channel, dim_height, dim_width)
            latent = latent.expand(batch_size, dim_channel, dim_height, dim_width)   
        elif option == 'linear':
            latent = torch.linspace(value_min, value_max, batch_size)
            latent = latent.reshape(batch_size, 1, 1, 1)
            latent = latent.repeat(1, dim_channel, dim_height, dim_width)
            latent = latent.expand(batch_size, dim_channel, dim_height, dim_width)   
        elif option == 'zero':
            latent = torch.zeros(batch_size, dim_channel, dim_height, dim_width)
        elif option == 'one':
            latent = torch.ones(batch_size, dim_channel, dim_height, dim_width)
        else:
            print('error: not implemented')
        return latent


    def get_latent_interpolate_constant(self, batch_size: int, dim_channel: int, dim_height: int, dim_width: int, channelwise: bool=False, option: str='uniform', value_min: float=-1.0, value_max: float=+1.0):
        if option == 'zero':
            latent = torch.zeros(batch_size, dim_channel, dim_height, dim_width)
        elif option == 'one':
            latent = torch.ones(batch_size, dim_channel, dim_height, dim_width)
        elif option == 'uniform':
            if channelwise:
                latent = torch.FloatTensor(1, dim_channel, 1, 1).uniform_(value_min, value_max)
            else:
                latent = torch.FloatTensor(1, 1, 1, 1).uniform_(value_min, value_max)
            latent = latent.repeat(batch_size, dim_channel, dim_height, dim_width) 
        elif option == 'normal':
            if channelwise:
                latent = torch.randn(size=(1, dim_channel, 1, 1))
            else:
                latent = torch.randn(size=(1, 1, 1, 1))
            latent = latent.repeat(batch_size, dim_channel, dim_height, dim_width) 
        else:
            print('error: not implemented')
        return latent
    

    def get_latent_shift(self, time_index: torch.Tensor, time_length: torch.Tensor, batch_size: int=None, dim_channel: int=None, dim_height: int=None, dim_width: int=None, channelwise: bool=None, distribution: str=None):
        if batch_size is None:
            batch_size = self.batch_size
        if dim_channel is None:
            dim_channel = self.dim_channel
        if dim_height is None:
            dim_height = self.dim_height
        if dim_width is None:
            dim_width = self.dim_width
        if channelwise is None:
            channelwise = self.channelwise_shift
        if distribution is None:
            distribution = self.distribution_shift 

        alpha   = self.get_alpha(batch_size, dim_channel, dim_height, dim_width, time_index, time_length, self.alpha_shift, self.alpha_shift_param)
        latent  = self.get_latent(batch_size, dim_channel, dim_height, dim_width, channelwise, distribution)
        latent  = alpha * latent
        return latent


    def get_latent_shift_interpolate(self, time_index: torch.Tensor, time_length: torch.Tensor, value_min: float=-1.0, value_max: float=1.0, batch_size: int=None, dim_channel: int=None, dim_height: int=None, dim_width: int=None, channelwise: bool=None, distribution: str=None):
        if batch_size is None:
            batch_size = self.batch_size
        if dim_channel is None:
            dim_channel = self.dim_channel
        if dim_height is None:
            dim_height = self.dim_height
        if dim_width is None:
            dim_width = self.dim_width
        if channelwise is None:
            channelwise = self.channelwise_shift
        if distribution is None:
            distribution = self.distribution_shift  
        
        alpha   = self.get_alpha(batch_size, dim_channel, dim_height, dim_width, time_index, time_length, self.alpha_shift, self.alpha_shift_param)
        latent  = self.get_latent_interpolate(batch_size, dim_channel, dim_height, dim_width, channelwise, distribution, value_min, value_max)
        latent  = alpha * latent
        return latent
    

    def get_latent_shift_interpolate_constant(self, time_index: torch.Tensor, time_length: torch.Tensor, value_min: float=-1.0, value_max: float=1.0, batch_size: int=None, dim_channel: int=None, dim_height: int=None, dim_width: int=None, channelwise: bool=None, distribution: str=None):
        if batch_size is None:
            batch_size = self.batch_size
        if dim_channel is None:
            dim_channel = self.dim_channel
        if dim_height is None:
            dim_height = self.dim_height
        if dim_width is None:
            dim_width = self.dim_width
        if channelwise is None:
            channelwise = self.channelwise_shift
        if distribution is None:
            distribution = self.distribution_shift 

        alpha   = self.get_alpha(batch_size, dim_channel, dim_height, dim_width, time_index, time_length, self.alpha_shift, self.alpha_shift_param)
        latent  = self.get_latent_interpolate_constant(batch_size, dim_channel, dim_height, dim_width, channelwise, distribution, value_min, value_max)
        latent  = alpha * latent
        return latent
    
    
    # option: uniform   => value_min = 0.5, value_max = 1.5
    # option: normal    => mean 0, std 1
    def get_latent_scale(self, time_index: torch.Tensor, time_length: torch.Tensor, batch_size: int=None, dim_channel: int=None, dim_height: int=None, dim_width: int=None, channelwise: bool=None, distribution: str=None):
        if batch_size is None:
            batch_size = self.batch_size
        if dim_channel is None:
            dim_channel = self.dim_channel
        if dim_height is None:
            dim_height = self.dim_height
        if dim_width is None:
            dim_width = self.dim_width
        if channelwise is None:
            channelwise = self.channelwise_scale
        if distribution is None:
            distribution = self.distribution_scale

        alpha   = self.get_alpha(batch_size, dim_channel, dim_height, dim_width, time_index, time_length, self.alpha_scale, self.alpha_scale_param)
        latent  = self.get_latent(batch_size, dim_channel, dim_height, dim_width, channelwise, distribution)
        latent  = torch.pow(2.0, alpha * latent)
        return latent


    # option: uniform   => value_min = 0.5, value_max = 1.5
    # option: normal    => mean 0, std 1
    def get_latent_scale_interpolate(self, time_index: torch.Tensor, time_length: torch.Tensor, value_min: float=-1.0, value_max: float=1.0, batch_size: int=None, dim_channel: int=None, dim_height: int=None, dim_width: int=None, channelwise: bool=None, distribution: str=None):
        if batch_size is None:
            batch_size = self.batch_size
        if dim_channel is None:
            dim_channel = self.dim_channel
        if dim_height is None:
            dim_height = self.dim_height
        if dim_width is None:
            dim_width = self.dim_width
        if channelwise is None:
            channelwise = self.channelwise_scale
        if distribution is None:
            distribution = self.distribution_scale
 
        alpha   = self.get_alpha(batch_size, dim_channel, dim_height, dim_width, time_index, time_length, self.alpha_scale, self.alpha_scale_param)
        latent  = self.get_latent_interpolate(batch_size, dim_channel, dim_height, dim_width, channelwise, distribution, value_min, value_max)
        latent  = torch.pow(2.0, alpha * latent)
        return latent


    def get_latent_scale_interpolate_constant(self, time_index: torch.Tensor, time_length: torch.Tensor, value_min: float=-1.0, value_max: float=1.0, batch_size: int=None, dim_channel: int=None, dim_height: int=None, dim_width: int=None, channelwise: bool=None, distribution: str=None):
        if batch_size is None:
            batch_size = self.batch_size
        if dim_channel is None:
            dim_channel = self.dim_channel
        if dim_height is None:
            dim_height = self.dim_height
        if dim_width is None:
            dim_width = self.dim_width
        if channelwise is None:
            channelwise = self.channelwise_scale
        if distribution is None:
            distribution = self.distribution_scale
                
        alpha   = self.get_alpha(batch_size, dim_channel, dim_height, dim_width, time_index, time_length, self.alpha_scale, self.alpha_scale_param)
        latent  = self.get_latent_interpolate_constant(batch_size, dim_channel, dim_height, dim_width, channelwise, distribution, value_min, value_max)
        latent  = torch.pow(2.0, alpha * latent)
        return latent


class RandomTransform():
    def __init__(self, options: Sequence[int]=[1,2,3,4,5], channelwise: bool=True):
        self.options        = options
        self.channelwise    = channelwise
    # x: batch_size x dim_channel x dim_height x dim_width
    # alpha: batch_size x 1
    def __call__(self, x: torch.Tensor, alpha: Sequence[float]):
        dim_channel = x.shape[1]
        options = self._get_options(alpha, dim_channel)
        y = self._apply_transform_batch(x, options)
        return y, options 
       
       
    def _get_options(self, alpha: Sequence[float], dim_channel: int):
        batch_size = len(alpha)
        
        if self.channelwise:
            dim_channel = dim_channel
        else:
            dim_channel = 1
            
        options = np.zeros(shape=(batch_size, dim_channel), dtype=np.int8) 
            
        for i in range(batch_size):
            for j in range(dim_channel): 
                threshold = alpha[i]
                if random.random() > threshold:
                    option = 0
                else:
                    option = random.choice(self.options)
                options[i,j] = option
        
        options = options.astype(int)
        return options 
    
   
    def _apply_transform_batch(self, x, options):
        batch_size = x.shape[0]
        y = torch.zeros_like(x)
        for i in range(batch_size):
            if self.channelwise:
                for j in range(x.shape[1]):
                    x_channel = x[i,j].unsqueeze(0)
                    y[i,j] = self._apply_transform(x_channel, options[i,j])
            else:
                y[i] = self._apply_transform(x[i], options[i])
        return y
    
           
    def _apply_transform_inverse_batch(self, x, options):
        batch_size = x.shape[0]
        y = torch.zeros_like(x)
        for i in range(batch_size):
            if self.channelwise:
                for j in range(x.shape[1]):
                    x_channel = x[i,j].unsqueeze(0)
                    y[i,j] = self._apply_transform_inverse(x_channel, options[i,j])
            else:
                y[i] = self._apply_transform_inverse(x[i], options[i])
        return y
    
    
    def _apply_transform(self, x, option):
        if option == 0:
            y = x
        elif option == 1:
            y = transforms.functional.hflip(x)
        elif option == 2:
            y = transforms.functional.vflip(x)
        elif option == 3:
            y = transforms.functional.rotate(x, 90)
        elif option == 4:
            y = transforms.functional.rotate(x, 180)
        elif option == 5:
            y = transforms.functional.rotate(x, 270)
        return y


    def _apply_transform_inverse(self, x, option):
        if option == 0:
            y = x
        elif option == 1:
            y = transforms.functional.hflip(x)
        elif option == 2:
            y = transforms.functional.vflip(x)
        elif option == 3:
            y = transforms.functional.rotate(x, -90)
        elif option == 4:
            y = transforms.functional.rotate(x, -180)
        elif option == 5:
            y = transforms.functional.rotate(x, -270)
        return y