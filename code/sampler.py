import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms.functional import rotate
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

import math
import numpy as np
import csv
import statistics
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os
import random
from tqdm.auto import tqdm

from utils.datautils import normalize01, normalize01_global

class Sampler:
    def __init__(self, 
                dataset: Dataset, 
                args,
                Scheduler,
                dataset_hist,
                ):

        self.dataset        = dataset
        self.args           = args 
        self.Scheduler      = Scheduler
        self.dataset_hist   = dataset_hist
        
        '''
        if evaluate: 
            self.metric_fid = FrechetInceptionDistance(feature=64).to(device)
        ''' 

    def _get_latent_initial(self, model: Module):
        
        histogram_shape         = self.dataset_hist[0]
        histogram_bin_edges     = self.dataset_hist[1]
        histogram_mean_cum_sum  = self.dataset_hist[2]
        
        if self.args.mean_area == 'image-wise':
            dim_channel_sample = 1
        elif self.args.mean_area == 'channel-wise':
            dim_channel_sample = 3
            
        if self.args.sample_latent_shape.lower() == 'data':
            val_random  = torch.rand(self.args.sample_num)
            index_bin   = torch.searchsorted(histogram_mean_cum_sum, val_random)
            index_bin   = np.unravel_index(index_bin, histogram_shape)
            sample_mean = torch.empty(self.args.sample_num, 0)
             
            for c in range(dim_channel_sample):
                val_rand    = torch.rand(self.args.sample_num)
                bin_low     = histogram_bin_edges[c][index_bin[c]]
                bin_high    = histogram_bin_edges[c][index_bin[c]+1] 
                val_sample  = (bin_high - bin_low) * val_rand + bin_low
                val_sample  = val_sample.unsqueeze(-1)
                sample_mean = torch.cat((sample_mean, val_sample), 1)

        elif self.args.sample_latent_shape.lower() == 'zero':
            sample_mean = torch.zeros(self.args.sample_num, dim_channel_sample)
        elif self.args.sample_latent_shape.lower() == 'normal':
            sample_mean = torch.randn(self.args.sample_num, dim_channel_sample)
        elif self.args.sample_latent_shape.lower() == 'uniform':
            sample_mean = torch.FloatTensor(self.args.sample_num, dim_channel_sample).uniform_(-1, 1)
        elif self.args.sample_latent_shape.lower() == 'grid':
            sample_mean = torch.linspace(-1, 1, self.args.sample_num)

        sample = sample_mean[:,:,None,None]
        sample = sample.expand(self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        
        return sample
        
    
    def _get_latent_initial_interpolation(self, interpolation_shift):
        latent      = torch.zeros(self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        
        if interpolation_shift > 0:
            grid        = torch.linspace(-1, 1-interpolation_shift, self.args.sample_num)
        elif interpolation_shift < 0:
            grid        = torch.linspace(-1-interpolation_shift, 1, self.args.sample_num)
        else:
            grid        = torch.linspace(-1, 1, self.args.sample_num)
            
        grid        = grid[:,None,None,None]
        latent      = latent + grid
        
        return latent, grid
        
    
    def sample(self, model: Module, timesteps_used_epoch, interpolation_shift=None):
        
        sample, visual_list  = self._sample_mean_shift_momentum(model, timesteps_used_epoch)
    
        return sample, visual_list
    
    
    def _sample_mean_shift_momentum(self, model: Module, timesteps_used_epoch):
        latent                      = self._get_latent_initial(model)
        sample_t                    = latent.to(model.device)
        degrade_mask_t              = torch.zeros(self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size).to(model.device)
        degrade_mask_next_t         = torch.zeros(self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size).to(model.device)
        
        
        sample_t_list               = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        degraded_mask_list          = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        degraded_mask_next_list     = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        shift_list                  = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        shifted_list                = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        mask_list                   = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        shifted_result_list         = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        sample_0_list               = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        difference_list             = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        degraded_t_list             = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        degraded_next_t_list        = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        
        
        # print(self.Scheduler.get_black_area_num_pixels_all())
        # print(timesteps_used_epoch)
        # exit(1)
        
        with torch.no_grad():
            sample_progress_bar = tqdm(total=len(timesteps_used_epoch), leave=False)
            sample_progress_bar.set_description(f"Sampling(momentum sampling)")
            
            for i in range(len(timesteps_used_epoch)-1, -1, -1):
                t       = timesteps_used_epoch[i]
                time    = torch.Tensor([t])
                time    = time.expand(self.args.sample_num).to(model.device)
                
                shift               = self.Scheduler.get_schedule_shift_time(time, degrade_mask_t) 
                shifted_sample_t    = self.Scheduler.perturb_shift(sample_t, shift)
                
                mask                = model(shifted_sample_t, time).sample
                shifted_sample_0    = shifted_sample_t + mask # x`_0
                # shifted_sample_0    = mask
                
                
                # # adjust mean
                # shifted_sample_0    = shifted_sample_0 + (shifted_sample_t.mean(dim=(1,2,3)) - shifted_sample_0.mean(dim=(1,2,3)))
                sample_0            = self.Scheduler.perturb_shift_inverse(shifted_sample_0, shift)
                
                
                # print("===============================================================")
                # print(sample_t.min(), sample_t.max(), sample_t.mean())
                # print(mask.min(), mask.max(), mask.mean())
                # print(sample_0.min(), sample_0.max(), sample_0.mean())
                
                sample_t_list[len(timesteps_used_epoch) - i]            = sample_t
                shift_list[len(timesteps_used_epoch) - i]               = shift
                shifted_list[len(timesteps_used_epoch) - i]             = shifted_sample_t
                mask_list[len(timesteps_used_epoch) - i]                = mask
                shifted_result_list[len(timesteps_used_epoch) - i]      = shifted_sample_0
                sample_0_list[len(timesteps_used_epoch) - i]            = sample_0
                
                if i > 0:
                    next_t  = time - 1
                elif i == 0: # for last step of momentum sampling
                    next_t  = time
                black_area_num_t            = self.Scheduler.get_black_area_num_pixels_time(time)
                black_area_num_next_t       = self.Scheduler.get_black_area_num_pixels_time(next_t)
                
                
                if self.args.sampling_mask_dependency == 'independent':
                    # degraded_t  = self.Scheduler.degrade_with_mask(sample_0, degrade_mask_next_t, mean_option=self.args.mean_option, mean_area=self.args.mean_area)
                    degraded_t, degrade_mask_t, mean_mask_t                 = self.Scheduler.degrade_independent_base_sampling(black_area_num_t, sample_0, mean_option=self.args.mean_option, mean_area=self.args.mean_area)
                    degraded_next_t, degrade_mask_next_t, mean_mask_next_t  = self.Scheduler.degrade_independent_base_sampling(black_area_num_next_t, sample_0, mean_option=self.args.mean_option, mean_area=self.args.mean_area)
                    
                    degraded_mask_list[len(timesteps_used_epoch) - i]       = degrade_mask_t
                    degraded_mask_next_list[len(timesteps_used_epoch) - i]  = degrade_mask_next_t
                    
                    
                elif self.args.sampling_mask_dependency == 'dependent_prev':
                    degraded_t  = self.Scheduler.degrade_with_mask(sample_0, degrade_mask_next_t, mean_option=self.args.mean_option, mean_area=self.args.mean_area)
                    degraded_next_t, degrade_mask_next_t, mean_mask_next_t  = self.Scheduler.degrade_independent_base_sampling(black_area_num_next_t, sample_0, mean_option=self.args.mean_option, mean_area=self.args.mean_area)
                    
                    degraded_mask_list[len(timesteps_used_epoch) - i]  = degrade_mask_next_t
                    
                    
                elif self.args.sampling_mask_dependency == 'dependent_t':
                    degraded_t, degrade_mask_t, mean_mask_t, degraded_next_t, degrade_mask_next_t, mean_mask_next_t \
                        = self.Scheduler.degrade_dependent_base_sampling(black_area_num_t, black_area_num_next_t, sample_0, mean_option=self.args.mean_option, mean_area=self.args.mean_area)
                        
                    degraded_mask_list[len(timesteps_used_epoch) - i]       = degrade_mask_t
                    degraded_mask_next_list[len(timesteps_used_epoch) - i]  = degrade_mask_next_t
                
                    
                if self.args.momentum_adaptive == 'base_sampling':
                    """
                    base momenutm sampling: cold diffusion
                    x_{t-1} = D(x_0, t-1)
                    """
                    if i == 0:
                        break
                    difference  = degraded_next_t - degraded_t
                    sample_t    = degraded_next_t
                
                elif self.args.momentum_adaptive == 'base_momentum':
                    """
                    base momenutm sampling: cold diffusion
                    x_{t-1} = x_t - D(x_0, t) + D(x_0, t-1)
                    """
                    if i > 0:
                        difference  = degraded_next_t - degraded_t
                        sample_t    = sample_t + difference
                        
                        # sample_t    = normalize01(sample_t)
                        # plt.imshow(sample_t[0].float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
                        # plt.savefig('test.png')
                    # elif i == 0:
                    #     sample_t    = sample_t - degraded_t + sample_0 # in momentum sampling algorithm, last step does not degrade
                    #     sample_0    = sample_t
                        
                    #     sample_0    = normalize01(sample_0)
                    #     plt.imshow(sample_0[0].float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
                    #     plt.savefig('test.png')
                    
                elif self.args.momentum_adaptive == 'momentum':
                    '''
                    new momentum sampling
                    difference: x_t - D(x_0, t)
                    weight: a, 1-a
                    '''
                    difference  = sample_t - degraded_t
                    momentum    = (1-self.args.adaptive_momentum_rate) * momentum + self.args.adaptive_momentum_rate * difference
                    sample_t    = momentum + degraded_next_t
                    
                elif self.args.momentum_adaptive == 'boosting':
                    '''
                    weight: a^2 + b^2 = 1, a is decaying parameter
                    '''
                    ratio       = self.Scheduler.get_ratio_list()
                    a           = ratio[i-1]
                    b           = math.sqrt(1-(a**2))
                    
                    difference  = sample_t - degraded_t
                    momentum    = (a**2) * momentum + (b**2) * difference
                    momentum    = difference
                    sample_t    = momentum + degraded_next_t
                                    
                degraded_next_t_list[len(timesteps_used_epoch) - i] = degraded_next_t    
                degraded_t_list[len(timesteps_used_epoch) - i]      = degraded_t    
                difference_list[len(timesteps_used_epoch) - i]      = difference    
                # if i != 0:
                #     sample_t_list[len(timesteps_used_epoch) - i]        = sample_t
                    
                sample_progress_bar.update(1)
        sample_progress_bar.close()
        
        return sample_0, [sample_t_list, shift_list, shifted_list, mask_list, shifted_result_list, sample_0_list, degraded_mask_list, degraded_mask_next_list, degraded_t_list, difference_list, degraded_next_t_list]
    
    
    def _sample_interpolation(self, model: Module, timesteps_used_epoch, interpolation_shift):
        latent, mu                  = self._get_latent_initial_interpolation(interpolation_shift)
        sample_t                    = latent.to(model.device)
        degrade_mask_t              = torch.zeros(1, self.args.out_channel, self.args.data_size, self.args.data_size).to(model.device)
        degrade_mask_next_t         = torch.zeros(1, self.args.out_channel, self.args.data_size, self.args.data_size).to(model.device)
        
        
        sample_t_list               = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        degraded_mask_list          = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        shift_list                  = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        shifted_list                = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        mask_list                   = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        shifted_result_list         = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        sample_0_list               = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        difference_list             = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        degraded_t_list             = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        degraded_next_t_list        = torch.zeros(len(timesteps_used_epoch)+1, self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        
        
        # print(self.Scheduler.get_black_area_num_pixels_all())
        # print(timesteps_used_epoch)
        # exit(1)
        
        with torch.no_grad():
            sample_progress_bar = tqdm(total=len(timesteps_used_epoch), leave=False)
            sample_progress_bar.set_description(f"Sampling(momentum sampling)")
            
            if self.args.sampling_mask_dependency == 'independent':
                index_list  = None
            elif self.args.sampling_mask_dependency == 'dependent':
                # make random list: list size = [sample_num, img_size]
                index_list = torch.stack([torch.randperm(self.args.data_size * self.args.data_size) for _ in range(self.args.sample_num)]).to(model.device)
            
            for i in range(len(timesteps_used_epoch)-1, -1, -1):
                t       = timesteps_used_epoch[i]
                time    = torch.Tensor([t])
                time    = time.expand(self.args.sample_num).to(model.device)
                
                shift               = self.Scheduler.get_schedule_shift_time_interpolation(time, mu.squeeze().to(model.device), interpolation_shift) 
                shifted_sample_t    = self.Scheduler.perturb_shift(sample_t, shift)
                
                mask                = model(shifted_sample_t, time).sample
                shifted_sample_0    = shifted_sample_t + mask # x`_0
                sample_0            = self.Scheduler.perturb_shift_inverse(shifted_sample_0, shift)
                
                
                shift_list[len(timesteps_used_epoch) - i]               = shift
                shifted_list[len(timesteps_used_epoch) - i]             = shifted_sample_t
                mask_list[len(timesteps_used_epoch) - i]                = mask
                shifted_result_list[len(timesteps_used_epoch) - i]      = shifted_sample_0
                sample_0_list[len(timesteps_used_epoch) - i]            = sample_0
                
                if i > 0:
                    next_t                      = time - 1
                    black_area_num_t            = self.Scheduler.get_black_area_num_pixels_time(time)
                    black_area_num_next_t       = self.Scheduler.get_black_area_num_pixels_time(next_t)
                    
                    degraded_t  = self.Scheduler.degrade_with_mask(sample_0, degrade_mask_next_t, mean_option=self.args.mean_option, mean_area=self.args.mean_area)
                    
                    degraded_next_t, degrade_mask_next_t, mean_mask_next_t  = self.Scheduler.degrade_interpolation_sampling(black_area_num_next_t, sample_0, mean_option=self.args.mean_option, mean_area=self.args.mean_area)
                    
                        
                    difference  = sample_t - degraded_t
                    
                    if self.args.momentum_adaptive == 'base_momentum':
                        """
                        base momenutm sampling: cold diffusion
                        x_{t-1} = x_t - D(x_0, t) + D(x_0, t-1)
                        """
                        sample_t    = degraded_next_t + difference
                        
                    elif self.args.momentum_adaptive == 'momentum':
                        '''
                        new momentum sampling
                        difference: x_t - D(x_0, t)
                        weight: a, 1-a
                        '''
                        momentum    = (1-self.args.adaptive_momentum_rate) * momentum + self.args.adaptive_momentum_rate * difference
                        sample_t    = momentum + degraded_next_t
                        
                    elif self.args.momentum_adaptive == 'boosting':
                        '''
                        weight: a^2 + b^2 = 1, a is decaying parameter
                        '''
                        ratio       = self.Scheduler.get_ratio_list()
                        a           = ratio[i-1]
                        b           = math.sqrt(1-(a**2))
                        
                        momentum    = (a**2) * momentum + (b**2) * difference
                        momentum    = difference
                        sample_t    = momentum + degraded_next_t
                        
                        
                    degraded_next_t_list[len(timesteps_used_epoch) - i] = degraded_next_t    
                    degraded_t_list[len(timesteps_used_epoch) - i]      = degraded_t    
                    difference_list[len(timesteps_used_epoch) - i]      = difference    
                    sample_t_list[len(timesteps_used_epoch) - i]        = sample_t
                    degraded_mask_list[len(timesteps_used_epoch) - i]   = degrade_mask_next_t
                
                sample_progress_bar.update(1)
        sample_progress_bar.close()
        
        return sample_0, mu, [sample_t_list, shift_list, shifted_list, mask_list, shifted_result_list, sample_0_list, degraded_mask_list, degraded_t_list, difference_list, degraded_next_t_list]
    
        
    def _save_image_grid(self, sample: torch.Tensor, normalization='global', dir_save=None, file_sample=None):
        batch_size  = sample.shape[0]
        nrow        = int(np.ceil(np.sqrt(batch_size)))
        if normalization == 'global':
            sample      = normalize01_global(sample)
        elif normalization == 'image':
            sample      = normalize01(sample)
            
        try:
            grid_sample = make_grid(sample, nrow=nrow, normalize=False, scale_each=False)
            
            if dir_save is not None and file_sample is not None:
                file_sample = os.path.join(dir_save, file_sample)
                save_image(grid_sample, file_sample)
        
        except ZeroDivisionError:
            grid_sample = None
        
        return grid_sample
    
    
    def _save_multi_index_image_grid(self, sample: torch.Tensor, nrow=None, normalization='global', option=None):
        # sample.shape = batch_size, timesteps, channel, height, width
        num_timesteps   = sample.shape[1]
        if nrow == None:
            nrow            = int(np.ceil(np.sqrt(num_timesteps))) 
        grids           = []
        for i in range(sample.shape[0]):
            if option == 'skip_first':
                if normalization == 'global':
                    sample_i   = normalize01_global(sample[i][1:])
                elif normalization == 'image':
                    sample_i   = normalize01(sample[i][1:])
                elif normalization == None:
                    sample_i    = sample[i][1:]
            
            else:
                if normalization == 'global':
                    sample_i   = normalize01_global(sample[i])
                elif normalization == 'image':
                    sample_i   = normalize01(sample[i])
                elif normalization == None:
                    sample_i    = sample[i]
            
            grid    = make_grid(sample_i, nrow=nrow, normalize=False, scale_each=False)
            
            grids.append(grid)
            
        return grids
    
    
    def _save_image_multi_grid(self, sample: list, sample_t: list, dir_save: str, file_sample: str):
        batch_size  = sample[0].shape[0]
        nrow        = int(np.ceil(np.sqrt(batch_size)))
        grid_name   = os.path.join(dir_save, file_sample)
        
        sample[0]   = normalize01(sample[0])
        sample[1]   = normalize01(sample[1])
        sample[2]   = normalize01(sample[2])
        sample[3]   = normalize01(sample[3])
        sample[4]   = normalize01(sample[4])
        sample[5]   = normalize01(sample[5])
        
        grid1       = make_grid(sample[0], nrow=nrow, normalize=True)
        grid2       = make_grid(sample[1], nrow=nrow, normalize=True)
        grid3       = make_grid(sample[2], nrow=nrow, normalize=True)
        grid4       = make_grid(sample[3], nrow=nrow, normalize=True)
        grid5       = make_grid(sample[4], nrow=nrow, normalize=True)
        grid6       = make_grid(sample[5], nrow=nrow, normalize=True)
        
        grid1       = grid1.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid2       = grid2.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid3       = grid3.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid4       = grid4.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid5       = grid5.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid6       = grid6.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        
        fig, axarr = plt.subplots(2,3, figsize=(15,10)) 
        axarr[0][0].imshow(grid1)
        axarr[0][1].imshow(grid2)
        axarr[0][2].imshow(grid3)
        axarr[1][0].imshow(grid4)
        axarr[1][1].imshow(grid5)
        axarr[1][2].imshow(grid6)
        
        axarr[0][0].set_title("input")
        axarr[0][1].set_title("T->0->..->{}->0".format(sample_t[0]))
        axarr[0][2].set_title("T->0->..->{}->0".format(sample_t[1]))
        axarr[1][0].set_title("T->0->..->{}->0".format(sample_t[2]))
        axarr[1][1].set_title("T->0->..->{}->0".format(sample_t[3]))
        axarr[1][2].set_title("T->0->..->{}->0".format(sample_t[4]))
        
        axarr[0][0].axis("off")
        axarr[0][1].axis("off")
        axarr[0][2].axis("off")
        axarr[1][0].axis("off")
        axarr[1][1].axis("off")
        axarr[1][2].axis("off")
        
        plt.tight_layout()
        # fig.suptitle('T ------> 0',fontweight ="bold") 
        fig.savefig(grid_name)
        plt.close(fig)
       
        
    def _save_image_pair_grid(self, data1: torch.Tensor, data2: torch.Tensor, dir_save: str, file_save: str):
        batch_size  = data1.shape[0]
        data = torch.zeros(batch_size*2, data1.shape[1], data1.shape[2], data1.shape[3])
        for i in range(batch_size):
            data[2*i]   = data1[i]
            data[2*i+1] = data2[i]
        nrow_batch  = int(np.ceil(np.sqrt(batch_size))) * 2
        data        = normalize01(data)
        grid_data   = make_grid(data, nrow=nrow_batch, normalize=True)
        file_save   = os.path.join(dir_save, file_save)
        save_image(grid_data, file_save)

    
    def get_nearest_neighbor(self, source: torch.Tensor, augment: bool=False, metric: str='cosine'):
        batch_size  = source.shape[0]
        score       = torch.Tensor()
        score       = score.to(source.device)
        dataloader  = DataLoader(self.dataset, batch_size=batch_size, drop_last=False, shuffle=False)
        transform   = transforms.Compose([transforms.RandomHorizontalFlip()]) 
        transform_resize    = transforms.Compose([transforms.Resize([32, 32])]) 
        source_small        = transform_resize(source)
        
        for i, (data, label) in enumerate(dataloader):
            data = normalize01(data)
            data = data.to(source.device)
            data_small = transform_resize(data)
            sim = self._compute_similarity(source_small, data_small, metric)
            sim = sim.to(source.device)

            if augment:
                data_aug = transform(data)
                data_aug_small = transform_resize(data_aug)
                sim_aug = self._compute_similarity(source_small, data_aug_small, metric)
                sim_aug = sim_aug.to(source.device)
                sim = torch.max(sim, sim_aug) 
            
            score = torch.cat((score, sim), dim=0)

        max_val, max_idx = score.max(dim=0)
        nearest_neighbor = torch.zeros_like(source) 

        for i in range(len(max_idx)):
            nearest = self.dataset[max_idx[i]][0]
            nearest_neighbor[i] = nearest
        return nearest_neighbor
    
    
    def _compute_similarity(self, source: torch.Tensor, target: torch.Tensor, metric: str='cosine'):
            vec_source = nn.Flatten()(source)
            vec_target = nn.Flatten()(target)
            if metric.lower() == 'cosine':
                score = nn.functional.cosine_similarity(vec_source[None,:,:], vec_target[:,None,:], dim=2) 
            return score