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
                ):

        self.dataset        = dataset
        self.args           = args 
        self.Scheduler      = Scheduler
        
        '''
        if evaluate: 
            self.metric_fid = FrechetInceptionDistance(feature=64).to(device)
        ''' 

    def _get_latent_initial(self, model: Module):
        
        latent      = torch.zeros(self.args.sample_num, self.args.out_channel, self.args.data_size, self.args.data_size)
        
        return latent
    
    
    def sample(self, model: Module, timesteps_used_epoch):
        '''
        Generate the sampling result about time t only used for training 
        '''
        if self.args.sampling == 'base':
            if self.args.method == 'base':
                sample, t_list, t_mask_list, sample_list, mask_list, mean_values = self._sample(model, timesteps_used_epoch) 
                return sample, t_list, t_mask_list, sample_list, mask_list, mean_values
            elif self.args.method == 'mean_shift':
                sample, t_list, t_mask_list, sample_list, mean_values, sample_t_shift_list, sample_shift_list = self._sample_mean_shift(model, timesteps_used_epoch) 
                return sample, t_list, t_mask_list, sample_list, mean_values, sample_t_shift_list, sample_shift_list
                
        elif self.args.sampling == 'momentum':
            if self.args.method == 'base':
                # sample, sample_list, t_list, t_mask, next_t_mask, t_mask_list = self._sample_momentum(model, timesteps_used_epoch)
                sample, t_list, t_mask_list, sample_list, mean_values = self._sample_momentum(model, timesteps_used_epoch)
                return sample, t_list, t_mask_list, sample_list, mean_values
            elif self.args.method == 'mean_shift':
                # sample, sample_list, t_list, t_mask, next_t_mask, t_mask_list = self._sample_mean_shift_momentum(model, timesteps_used_epoch)
                sample, t_list, t_mask_list, sample_list = self._sample_mean_shift_momentum(model, timesteps_used_epoch)
                return sample, t_list, t_mask_list, sample_list, mean_values
                
        


    def _sample(self, model: Module, timesteps_used_epoch):
        latent      = self._get_latent_initial(model)
        sample_t    = latent.to(model.device)
        sample_list = sample_t.unsqueeze(dim=1).cpu()
        t_list      = sample_t.unsqueeze(dim=1).cpu()
        t_mask_list = sample_t.unsqueeze(dim=1).cpu()
        mask_list   = sample_t.unsqueeze(dim=1).cpu()
        
        with torch.no_grad():
            sample_progress_bar = tqdm(total=len(timesteps_used_epoch), leave=False)
            sample_progress_bar.set_description(f"Sampling(base sampling)")
            
            if self.args.sampling_mask_dependency == 'independent':
                index_list  = None
            elif self.args.sampling_mask_dependency == 'dependent':
                # index_list  = torch.randperm(len(timesteps_used_epoch), dtype=torch.int64)
                index_list = torch.stack([torch.randperm(self.args.data_size * self.args.data_size) for _ in range(self.args.sample_num)]).to(model.device)
                
            index_start = 0
            mean_values = torch.zeros(self.args.sample_num, len(timesteps_used_epoch))
            
            for i in range(len(timesteps_used_epoch)-1, -1, -1):
                t       = timesteps_used_epoch[i]
                time    = torch.Tensor([t])
                time    = time.expand(self.args.sample_num).to(model.device)
                
                mask            = model(sample_t, time).sample
                sample_0        = sample_t + mask # x`_0
                
                mask_list       = torch.cat((mask_list, mask.unsqueeze(dim=1).cpu()), dim=1)
                sample_list     = torch.cat((sample_list, sample_0.unsqueeze(dim=1).cpu()), dim=1)
                if i > 0:
                    black_area_num_t            = self.Scheduler.get_black_area_num_pixels_time(time-1)
                    # white_area_num_t            = self.args.data_size * self.args.data_size - black_area_num_t
                    
                    if self.args.sampling_mask_dependency == 'independent':
                        sample_t, degrade_mask, mean_value    = self.Scheduler.degrade_independent_base_sampling(black_area_num_t, sample_0, mean_option=self.args.mean_option)
                    elif self.args.sampling_mask_dependency == 'dependent':
                        sample_t, degrade_mask, mean_value    = self.Scheduler.degrade_dependent_base_sampling(sample_0, mean_option=self.args.mean_option, black_area_num=black_area_num_t[0], index_list=index_list)
                    
                    degrade_mask = degrade_mask.expand_as(sample_0)
                    t_mask_list  = torch.cat((t_mask_list, degrade_mask.unsqueeze(dim=1).cpu()), dim=1)
                    
                    t_list      = torch.cat((t_list, sample_t.unsqueeze(dim=1).cpu()), dim=1)
                    
                    mean_values[:, len(timesteps_used_epoch)-i-1] = mean_value.mean(dim=(1,2,3))
            
                sample_progress_bar.update(1)
        sample_progress_bar.close()
        
        return sample_0, t_list, t_mask_list, sample_list, mask_list, mean_values
    
    
    def _sample_mean_shift(self, model: Module, timesteps_used_epoch):
        latent              = self._get_latent_initial(model)
        sample_t            = latent.to(model.device)
        sample_t_shift_list = sample_t.unsqueeze(dim=1).cpu()
        sample_shift_list   = sample_t.unsqueeze(dim=1).cpu()
        sample_list         = sample_t.unsqueeze(dim=1).cpu()
        t_list              = sample_t.unsqueeze(dim=1).cpu()
        t_mask_list         = sample_t.unsqueeze(dim=1).cpu()
        
        with torch.no_grad():
            sample_progress_bar = tqdm(total=len(timesteps_used_epoch), leave=False)
            sample_progress_bar.set_description(f"Sampling(base sampling)")
            
            if self.args.sampling_mask_dependency == 'independent':
                index_list  = None
            elif self.args.sampling_mask_dependency == 'dependent':
                # index_list  = torch.randperm(len(timesteps_used_epoch), dtype=torch.int64)
                index_list = torch.stack([torch.randperm(self.args.data_size * self.args.data_size) for _ in range(self.args.sample_num)]).to(model.device)
                
            index_start = 0
            mean_values = torch.zeros(self.args.sample_num, len(timesteps_used_epoch))
            for i in range(len(timesteps_used_epoch)-1, -1, -1):
                t       = timesteps_used_epoch[i]
                time    = torch.Tensor([t])
                time    = time.expand(self.args.sample_num).to(model.device)
                
                shift                   = self.Scheduler.get_schedule_shift_time(time)
                sample_t_shift          = self.Scheduler.perturb_shift(sample_t, shift)
                sample_t_shift_list     = torch.cat((sample_t_shift_list, sample_t_shift.unsqueeze(dim=1).cpu()), dim=1)
                
                mask                = model(sample_t_shift, time).sample
                sample_0            = sample_t_shift + mask # x`_0
                sample_shift_list   = torch.cat((sample_shift_list, sample_0.unsqueeze(dim=1).cpu()), dim=1)
                sample_0            = self.Scheduler.perturb_shift_inverse(sample_0, shift)
                sample_list         = torch.cat((sample_list, sample_0.unsqueeze(dim=1).cpu()), dim=1)
                if i > 0:
                    black_area_num_t            = self.Scheduler.get_black_area_num_pixels_time(time-1)
                    # white_area_num_t            = self.args.data_size * self.args.data_size - black_area_num_t
                    
                    if self.args.sampling_mask_dependency == 'independent':
                        sample_t, degrade_mask, mean_value    = self.Scheduler.degrade_independent_base_sampling(black_area_num_t, sample_0, mean_option=self.args.mean_option)
                    elif self.args.sampling_mask_dependency == 'dependent':
                        sample_t, degrade_mask, mean_value    = self.Scheduler.degrade_dependent_base_sampling(sample_0, mean_option=self.args.mean_option, black_area_num=black_area_num_t[0], index_list=index_list)
                    
                    degrade_mask = degrade_mask.expand_as(sample_0)
                    t_mask_list  = torch.cat((t_mask_list, degrade_mask.unsqueeze(dim=1).cpu()), dim=1)
                    
                    t_list      = torch.cat((t_list, sample_t.unsqueeze(dim=1).cpu()), dim=1)
                    
                    mean_values[:, len(timesteps_used_epoch)-i-1] = mean_value.mean(dim=(1,2,3))
                    
                sample_progress_bar.update(1)
        sample_progress_bar.close()
        
        return sample_0, t_list, t_mask_list, sample_list, mean_values, sample_t_shift_list, sample_shift_list
    
    
    def _sample_momentum(self, model: Module, timesteps_used_epoch):
        latent      = self._get_latent_initial(model)
        sample_t    = latent.to(model.device)
        sample_list = sample_t.unsqueeze(dim=1).cpu()
        t_list      = sample_t.unsqueeze(dim=1).cpu()
        t_mask_list = sample_t.unsqueeze(dim=1).cpu()
        mask_list   = sample_t.unsqueeze(dim=1).cpu()
        
        with torch.no_grad():
            sample_progress_bar = tqdm(total=len(timesteps_used_epoch), leave=False)
            sample_progress_bar.set_description(f"Sampling(momentum sampling)")
            
            if self.args.sampling_mask_dependency == 'independent':
                index_list  = None
            elif self.args.sampling_mask_dependency == 'dependent':
                # index_list  = torch.randperm(len(timesteps_used_epoch), dtype=torch.int64)
                index_list = torch.stack([torch.randperm(self.args.data_size * self.args.data_size) for _ in range(self.args.sample_num)]).to(model.device)
                
            index_start = 0
            mean_values = []
            for i in range(len(timesteps_used_epoch)-1, -1, -1):
                t       = timesteps_used_epoch[i]
                time    = torch.Tensor([t])
                time    = time.expand(self.args.sample_num).to(model.device)
                
                mask            = model(sample_t, time).sample
                sample_0        = sample_t + mask # x`_0
                
                sample_list     = torch.cat((sample_list, sample_0.unsqueeze(dim=1).cpu()), dim=1)
                mask_list       = torch.cat((mask_list, mask.unsqueeze(dim=1).cpu()), dim=1)
                if i > 0:
                    next_t                      = time - 1
                    black_area_num_t            = self.Scheduler.get_black_area_num_pixels_time(time)
                    black_area_num_next_t       = self.Scheduler.get_black_area_num_pixels_time(next_t)
                    black_area_num_difference   = black_area_num_t - black_area_num_next_t
                    
                    if self.args.sampling_mask_dependency == 'independent':
                        degraded_difference, difference_mask    = self.Scheduler.degrade_independent_sampling(black_area_num_difference, sample_0, mean_option=self.args.mean_option)
                    elif self.args.sampling_mask_dependency == 'dependent':
                        index_end   = index_start+black_area_num_difference[0]
                        sample_t, difference_mask, mean_value    = self.Scheduler.degrade_dependent_momentum_sampling(sample_t, sample_0, mean_option=self.args.mean_option, index_start=index_start, index_end=index_end, index_list=index_list)
                        index_start = index_end
                    
                    difference_mask = difference_mask.expand_as(sample_0)
                    t_mask_list  = torch.cat((t_mask_list, difference_mask.unsqueeze(dim=1).cpu()), dim=1)
                    
                    # sample_t    = sample_t + degraded_difference
                    t_list      = torch.cat((t_list, sample_t.unsqueeze(dim=1).cpu()), dim=1)
                    
                    mean_values.append(mean_value.mean().cpu())
                    
                sample_progress_bar.update(1)
        sample_progress_bar.close()
        
        return sample_0, t_list, t_mask_list, sample_list, mean_values
    
    
    def _sample_mean_shift_momentum(self, model: Module, timesteps_used_epoch):
        latent      = self._get_latent_initial(model)
        sample_t    = latent.to(model.device)
        sample_list = sample_t.unsqueeze(dim=1).cpu()
        t_list      = sample_t.unsqueeze(dim=1).cpu()
        t_mask_list = sample_t.unsqueeze(dim=1).cpu()
        
        with torch.no_grad():
            sample_progress_bar = tqdm(total=len(timesteps_used_epoch), leave=False)
            sample_progress_bar.set_description(f"Sampling")
            
            if self.args.sampling_mask_dependency == 'independent':
                index_list  = None
            elif self.args.sampling_mask_dependency == 'dependent':
                # index_list  = torch.randperm(len(timesteps_used_epoch), dtype=torch.int64)
                index_list = torch.stack([torch.randperm(self.args.data_size * self.args.data_size) for _ in range(self.args.sample_num)])
                
            index_start = 0
            for i in range(len(timesteps_used_epoch)-1, -1, -1):
                t       = timesteps_used_epoch[i]
                time    = torch.Tensor([t])
                time    = time.expand(self.args.sample_num).to(model.device)
                
                shift           = self.Scheduler.get_schedule_shift_time(time)
                sample_t_shift  = self.Scheduler.perturb_shift(sample_t, shift)
                
                mask            = model(sample_t_shift, time).sample
                sample_0        = sample_t_shift + mask # x`_0
                sample_0        = self.Scheduler.perturb_shift_inverse(sample_0, shift)   # x`_0
                
                sample_list     = torch.cat((sample_list, sample_0.unsqueeze(dim=1).cpu()), dim=1)
                if i > 0:
                    next_t                      = time - 1
                    black_area_num_t            = self.Scheduler.get_black_area_num_pixels_time(time)
                    black_area_num_next_t       = self.Scheduler.get_black_area_num_pixels_time(next_t)
                    black_area_num_difference   = black_area_num_t - black_area_num_next_t
                    
                    if self.args.sampling_mask_dependency == 'independent':
                        raise Exception("not implemented yet")

                        degraded_difference, difference_mask    = self.Scheduler.degrade_independent_sampling(black_area_num_difference, sample_0, mean_option=self.args.mean_option)
                    elif self.args.sampling_mask_dependency == 'dependent':
                        raise Exception("need to revise 'degrade_dependent_momentum_sampling' code ")
                        index_using = index_list[:, index_start:index_start+black_area_num_difference[0]]
                        index_start = index_start + black_area_num_difference[0]
                        degraded_difference, difference_mask    = self.Scheduler.degrade_dependent_momentum_sampling(sample_0, mean_option=self.args.mean_option, index=index_using)
                    
                    raise Exception("not implemented")

                    t_mask_list  = torch.cat((t_mask_list, difference_mask.unsqueeze(dim=1).cpu()), dim=1)
                    
                    # sample_t            = sample_t - degraded_t + degraded_next_t
                    sample_t    = sample_t + degraded_difference
                    t_list      = torch.cat((t_list, sample_t.unsqueeze(dim=1).cpu()), dim=1)
                    
                sample_progress_bar.update(1)
        sample_progress_bar.close()
        
        return sample_0, t_list, t_mask_list, sample_list
        
        
    # def _sample_mean_shift_momentum(self, model: Module, timesteps_used_epoch):
    #     # x_(t-1)   = x_t - D(x`_0, t) + D(x`_0, t-1)
    #     latent      = self._get_latent_initial(model)   # T0 
    #     sample_0    = latent.to(model.device)
    #     sample_list = sample_0.unsqueeze(dim=1).cpu()
    #     t_list      = sample_0.unsqueeze(dim=1).cpu()
    #     t_mask_list = sample_0.unsqueeze(dim=1).cpu()
        
    #     black_idx_t = None
        
    #     with torch.no_grad():
    #         sample_progress_bar = tqdm(total=len(timesteps_used_epoch), leave=False)
    #         sample_progress_bar.set_description(f"Sampling about trained t")
            
    #         for i in range(len(timesteps_used_epoch)-1, -1, -1):
    #             t               = timesteps_used_epoch[i]
    #             time            = torch.Tensor([t])
    #             time            = time.expand(self.args.sample_num).to(model.device)
                
    #             shift           = self.Scheduler.get_schedule_shift_time(time)  
    #             sample          = self.Scheduler.perturb_shift(sample_0, shift)   # x`_t
                
    #             mask            = model(sample, time).sample
    #             reconstruction  = sample + mask
    #             reconstruction  = self.Scheduler.perturb_shift_inverse(reconstruction, shift)   # x`_0
                
    #             sample_list     = torch.cat((sample_list, reconstruction.unsqueeze(dim=1).cpu()), dim=1)
                
    #             if i > 0:
    #                 next_t          = timesteps_used_epoch[i-1]
    #                 next_time       = torch.Tensor([next_t])
    #                 next_time       = next_time.expand(self.args.sample_num).to(model.device)
                        
    #                 black_area_num_t                            = self.Scheduler.get_black_area_num_pixels_time(time)
    #                 degraded_t,t_mask,black_idx_t, black_mean   = self.Scheduler.get_mean_mask(black_area_num_t, reconstruction, index=black_idx_t) # D(x`_0, t)
    #                 black_area_num_next_t                       = self.Scheduler.get_black_area_num_pixels_time(next_time)
                    
    #                 t_mask_list  = torch.cat((t_mask_list, t_mask.unsqueeze(dim=1).cpu()), dim=1)
                    
    #                 if self.args.mean_value_accumulate:
    #                     # degraded_next_t,next_t_mask,_,_               = self.Scheduler.get_mean_mask(black_area_num_next_t, reconstruction, index=black_idx_t, mean_value=black_mean) # D(x`_0, t-1)
    #                     degraded_next_t,next_t_mask,black_idx_t,_               = self.Scheduler.get_mean_mask(black_area_num_next_t, reconstruction, index=black_idx_t, mean_value=black_mean) # D(x`_0, t-1)
    #                 else:
    #                     # degraded_next_t,next_t_mask,_,_               = self.Scheduler.get_mean_mask(black_area_num_next_t, reconstruction, index=black_idx_t) # D(x`_0, t-1)
    #                     degraded_next_t,next_t_mask,black_idx_t,_               = self.Scheduler.get_mean_mask(black_area_num_next_t, reconstruction, index=black_idx_t) # D(x`_0, t-1)
                    
    #                 sample_0            = sample_0 - degraded_t + degraded_next_t
    #                 t_list              = torch.cat((t_list, sample_0.unsqueeze(dim=1).cpu()), dim=1)
                    
    #             else:
    #                 sample  = reconstruction
    #                 # t_list  = torch.cat((t_list, sample.unsqueeze(dim=1)), dim=1)
                    
    #             sample_progress_bar.update(1)
    #         sample_progress_bar.close()
            
    #         return sample, sample_list, t_list, t_mask, next_t_mask, t_mask_list
    
    
    def sample_all_t(self, model: Module):
        '''
        Generate the sampling result using all t include not used for training
        ex) t -> 0 -> t-1 -> 0 -> t-2 -> ... -> 1 -> 0
        '''
        if self.args.method == 'base':
            pass
        elif self.args.method == 'mean_shift':
            sample_list = self._sample_all_mean_shift_t(model)
        
        return sample_list
    
    
    def _sample_all_t(self, img: torch.Tensor, model: Module):
        sample_list = [img]
        time_length = self.args.updated_ddpm_num_steps
        
        with torch.no_grad():
            sample_all_t_progress_bar    = tqdm(total=time_length, leave=False)
            sample_all_t_progress_bar.set_description(f"Sampling random t")
            for sampleTime in range(1, time_length+1):
                
                t_noisy = self._get_noisy(sampleTime, img, model)
                
                for time in range(sampleTime, 0, -1):
                    time                = torch.Tensor([time])
                    
                    mask                = model(t_noisy, time.to(model.device)).sample
                    prediction          = mask + t_noisy
                    
                    if time == 1:
                        sample  = prediction
                        sample_list.append(sample)
                    
                    else:
                        t_noisy = self._get_noisy(sampleTime-1, prediction, model)

                sample_all_t_progress_bar.update(1)
        sample_all_t_progress_bar.close()
            
        return sample_list
    
    
    def _sample_all_mean_shift_t(self, model: Module):
        time_length = self.args.updated_ddpm_num_steps
      
        latent      = self._get_latent_initial(model)
        sample      = latent.to(model.device)
        
        # sample_list = [sample[0].unsqueeze(dim=0)]
        # sample_list = [sample]
        sample_list = sample.unsqueeze(dim=1)
        
        with torch.no_grad():
            sample_all_t_progress_bar   = tqdm(total=time_length, leave=False)
            sample_all_t_progress_bar.set_description(f"Sampling all t")
            for t in range(time_length, 0, -1): # t = time_length, time_length-1, ..., 2, 1
                time                = torch.Tensor([t])
                time                = time.expand(self.args.batch_size).to(model.device)
                
                shift               = self.Scheduler.get_schedule_shift_time(time)
                sample              = self.Scheduler.perturb_shift(sample, shift)
                
                mask                = model(sample, time).sample
                prediction          = sample + mask
                # prediction          = self.Scheduler.perturb_shift_inverse(prediction, shift)
                prediction          = self._shift_mean(prediction)
                # sample_list.append(prediction[0].unsqueeze(dim=0))
                # sample_list.append(prediction)
                sample_list         = torch.cat((sample_list, prediction.unsqueeze(dim=1)), dim=1)
        
                if t == 1:
                    sample  = prediction
                    # sample_list.append(sample[0].unsqueeze(dim=0))
                    # sample_list.append(sample)
                    sample_list         = torch.cat((sample_list, prediction.unsqueeze(dim=1)), dim=1)
                else:
                    black_area_num      = self.Scheduler.get_black_area_num_pixels_time(time-1)
                    noisy_img, noise    = self.Scheduler.get_mean_mask(black_area_num, prediction)
                    sample              = noisy_img
                    
                sample_all_t_progress_bar.update(1)
            sample_all_t_progress_bar.close()

        return sample, sample_list
    
    
    def result_each_t(self, img: torch.Tensor, model: Module):
        '''
        Generate first output from all time t
        ex) T -> 0, T-1 -> 0, T-2 -> 0, ... , 1 -> 0
        '''
        if self.args.method == 'base':
            pass
        elif self.args.method == 'shift':
            noisy_list, mask_list, sample_list = self._each_result_shift_t(img[0], model)
        elif self.args.method == 'mean_shift':
            noisy_list, mask_list, sample_list = self._each_result_mean_shift_t(img[0], model)
        
        return noisy_list, mask_list, sample_list
    
    
    def _each_result_mean_shift_t(self, img: torch.Tensor, model: Module):
        sample_list = []
        noise_list  = []
        noisy_list  = []
        mask_list   = []
        
        time_length = self.args.updated_ddpm_num_steps
        
        img = img.unsqueeze(dim=0)
        with torch.no_grad():
            each_result_t_progress_bar    = tqdm(total=time_length, leave=False)
            each_result_t_progress_bar.set_description(f"each first result of t")
            for sampleTime in range(1, time_length+1):
                
                shift   = self.Scheduler.get_schedule_shift_time(torch.tensor([sampleTime]))
                t_noisy = self._get_noisy_shift(sampleTime, img, shift, model)
                               
                time                = torch.Tensor([sampleTime])
                shift_time          = self.Scheduler.get_schedule_shift_time(time)
                
                mask                = model(t_noisy, time.to(model.device)).sample
                prediction          = mask + t_noisy
                prediction          = self.Scheduler.perturb_shift_inverse(prediction, shift_time)
                
                noisy_list.append(t_noisy)
                mask_list.append(mask)
                sample_list.append(prediction)

                each_result_t_progress_bar.update(1)
        each_result_t_progress_bar.close()
            
        return noisy_list, mask_list, sample_list
    
    
    def _get_noisy(self, time: int, img: torch.Tensor, model: Module):
        time                = torch.tensor([time])
        # time                = torch.Tensor(time, device=model.device)
                
        black_area_num      = self.Scheduler.get_black_area_num_pixels_time(time)
        noise               = self.Scheduler.get_mask(black_area_num)
        noise               = noise.to(model.device)
        sample              = img * noise
        
        return sample


    def _get_noisy_shift(self, time: int, img: torch.Tensor, shift: torch.Tensor, model: Module):
        time                = torch.tensor([time])
        # time                = torch.Tensor(time, device=model.device)
                
        black_area_num      = self.Scheduler.get_black_area_num_pixels_time(time)
        noisy_img, noise    = self.Scheduler.get_mean_mask(black_area_num, img)
        sample_shift        = self.Scheduler.perturb_shift(noisy_img, shift)
        
        return sample_shift


    def _save_image_grid(self, sample: torch.Tensor, dir_save=None, file_sample=None):
        batch_size  = sample.shape[0]
        nrow        = int(np.ceil(np.sqrt(batch_size)))
        sample      = normalize01(sample)
        grid_sample = make_grid(sample, nrow=nrow, normalize=True)
        
        if dir_save is not None and file_sample is not None:
            file_sample = os.path.join(dir_save, file_sample)
            save_image(grid_sample, file_sample)
        
        return grid_sample
    
    
    def _save_multi_index_image_grid(self, sample: torch.Tensor, nrow=None, option=None):
        # sample.shape = batch_size, timesteps, channle, height, width
        num_timesteps   = sample.shape[1]
        if nrow == None:
            nrow            = int(np.ceil(np.sqrt(num_timesteps))) 
        grid            = []
        for i in range(sample.shape[0]):
            if option == 'skip_first':
                sample_i   = normalize01(sample[i][1:])
            else:
                sample_i   = normalize01(sample[i])
            grid.append(make_grid(sample_i, nrow=nrow, normalize=True))
            
        return grid
        
    
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


    def _shift_mean(self, img: torch.Tensor):
        mean    = img.mean(dim=(1,2,3), keepdim=True)
        img     = img - mean
        
        return img