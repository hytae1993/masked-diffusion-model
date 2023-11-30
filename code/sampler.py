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

from utils.datautils import normalize01

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
        
        latent      = torch.zeros(self.args.batch_size, self.args.out_channel, self.args.data_size, self.args.data_size)
        
        return latent
    
    
    def sample(self, model: Module):
        if self.args.method == 'base':
            sample, sample_list, sample_t = self._sample(model) 
        elif self.args.method == 'shift':
            sample, sample_list, sample_t = self._sample_shift(model)
        elif self.args.method == 'mean_shift':
            sample, sample_list, sample_t = self._sample_mean_shift(model) 
        return sample, sample_list, sample_t


    def _sample(self, model: Module):
        time_length = self.args.updated_ddpm_num_steps
      
        latent      = self._get_latent_initial(model)
        sample      = latent.to(model.device)
        
        sample_per  = int(time_length / 5)
        sample_list = [sample]
        sample_t    = [time_length - sample_per, time_length - sample_per*2, time_length - sample_per*3, time_length - sample_per*4, time_length - sample_per*5]
        
        with torch.no_grad():
            sample_progress_bar = tqdm(total=time_length, leave=False)
            sample_progress_bar.set_description(f"Sampling")
            for t in range(time_length, 0, -1): # t = time_length, time_length-1, ..., 2, 1
                time                = torch.Tensor([t])
                time                = time.expand(self.args.batch_size).to(model.device)
                
                mask                = model(sample, time).sample
                prediction          = sample + mask
                
                if t == 1:
                    sample  = prediction
                    sample_list.append(sample)
                else:
                    black_area_num      = self.Scheduler.get_black_area_num_pixels_time(time-1)
                    noise               = self.Scheduler.get_mask(black_area_num)
                    noise               = noise.to(model.device)
                    sample              = prediction * noise
                    
                if t in sample_t:
                    sample_list.append(prediction)
                    
                sample_progress_bar.update(1)
            # sample_progress_bar.clear()
            sample_progress_bar.close()

        return sample, sample_list, sample_t
    
    
    def _sample_shift(self, model: Module):
        time_length = self.args.updated_ddpm_num_steps
      
        latent      = self._get_latent_initial(model)
        sample      = latent.to(model.device)
        
        sample_per  = int(time_length / 5)
        sample_list = [sample]
        sample_t    = [time_length - sample_per, time_length - sample_per*2, time_length - sample_per*3, time_length - sample_per*4, time_length - sample_per*5]
        
        with torch.no_grad():
            sample_progress_bar = tqdm(total=time_length, leave=False)
            sample_progress_bar.set_description(f"Sampling")
            for t in range(time_length, 0, -1): # t = time_length, time_length-1, ..., 2, 1
                time                = torch.Tensor([t])
                time                = time.expand(self.args.batch_size).to(model.device)
                
                shift               = self.Scheduler.get_schedule_shift_time(time)
                sample              = self.Scheduler.perturb_shift(sample, shift)
                
                mask                = model(sample, time).sample
                prediction          = sample + mask
                prediction          = self.Scheduler.perturb_shift_inverse(prediction, shift)
                
                if t == 1:
                    sample  = prediction
                    sample_list.append(sample)
                else:
                    black_area_num      = self.Scheduler.get_black_area_num_pixels_time(time-1)
                    noise               = self.Scheduler.get_mask(black_area_num)
                    noise               = noise.to(model.device)
                    sample              = prediction * noise
                    
                if t in sample_t:
                    sample_list.append(prediction)
                    
                sample_progress_bar.update(1)
            # sample_progress_bar.clear()
            sample_progress_bar.close()

        return sample, sample_list, sample_t
    
    
    def _sample_mean_shift(self, model: Module):
        time_length = self.args.updated_ddpm_num_steps
      
        latent      = self._get_latent_initial(model)
        sample      = latent.to(model.device)
        
        sample_per  = int(time_length / 5)
        sample_list = [sample]
        sample_t    = [time_length - sample_per, time_length - sample_per*2, time_length - sample_per*3, time_length - sample_per*4, time_length - sample_per*5]
        
        with torch.no_grad():
            sample_progress_bar = tqdm(total=time_length, leave=False)
            sample_progress_bar.set_description(f"Sampling")
            for t in range(time_length, 0, -1): # t = time_length, time_length-1, ..., 2, 1
                time                = torch.Tensor([t])
                time                = time.expand(self.args.batch_size).to(model.device)
                
                shift               = self.Scheduler.get_schedule_shift_time(time)
                sample              = self.Scheduler.perturb_shift(sample, shift)
                
                mask                = model(sample, time).sample
                prediction          = sample + mask
                prediction          = self.Scheduler.perturb_shift_inverse(prediction, shift)
                
                if t == 1:
                    sample  = prediction
                    sample_list.append(sample)
                else:
                    black_area_num      = self.Scheduler.get_black_area_num_pixels_time(time-1)
                    noisy_img, noise    = self.Scheduler.get_mean_mask(black_area_num, prediction)
                    sample              = noisy_img
                    
                if t in sample_t:
                    sample_list.append(prediction)
                    
                sample_progress_bar.update(1)
            # sample_progress_bar.clear()
            sample_progress_bar.close()

        return sample, sample_list, sample_t
    
    
    def sample_random_t(self, img: torch.Tensor, model: Module):
        '''
        Generate final output from all time t
        ex) t -> 0 -> t-1 -> 0 -> t-2 -> ... -> 1 -> 0
        '''
        if self.args.method == 'base':
            sample_list = self._sample_random_t(img[0], model)
        elif self.args.method == 'shift':
            sample_list = self._sample_random_shift_t(img[0], model)
        
        return sample_list
    
    
    def _sample_random_t(self, img: torch.Tensor, model: Module):
        sample_list = [img.unsqueeze(dim=0)]
        time_length = self.args.updated_ddpm_num_steps
        
        with torch.no_grad():
            sample_random_t_progress_bar    = tqdm(total=time_length, leave=False)
            sample_random_t_progress_bar.set_description(f"Sampling random t")
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

                sample_random_t_progress_bar.update(1)
        sample_random_t_progress_bar.close()
            
        return sample_list
    
    
    def _sample_random_shift_t(self, img: torch.Tensor, model: Module):
        sample_list = [img.unsqueeze(dim=0)]
        time_length = self.args.updated_ddpm_num_steps
        
        with torch.no_grad():
            sample_random_t_progress_bar    = tqdm(total=time_length, leave=False)
            sample_random_t_progress_bar.set_description(f"Sampling random t")
            for sampleTime in range(1, time_length+1):
                
                shift   = self.Scheduler.get_schedule_shift_time(torch.tensor([sampleTime]))
                t_noisy = self._get_noisy_shift(sampleTime, img, shift, model)
                               
                for time in range(sampleTime, 0, -1):
                    time                = torch.Tensor([time])
                    shift_time          = self.Scheduler.get_schedule_shift_time(time)
                    
                    mask                = model(t_noisy, time.to(model.device)).sample
                    prediction          = mask + t_noisy
                    prediction          = self.Scheduler.perturb_shift_inverse(prediction, shift_time)
                    
                    if time == 1:
                        sample  = prediction
                        sample_list.append(sample)
                    
                    else:
                        shift_time  = self.Scheduler.get_schedule_shift_time(time-1)
                        t_noisy     = self._get_noisy_shift(sampleTime-1, prediction, shift_time, model)

                sample_random_t_progress_bar.update(1)
        sample_random_t_progress_bar.close()
            
        return sample_list
    
    
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
    
    
    def _each_result_shift_t(self, img: torch.Tensor, model: Module):
        sample_list = [img.unsqueeze(dim=0)]
        noise_list  = []
        noisy_list  = []
        mask_list   = []
        
        time_length = self.args.updated_ddpm_num_steps
        
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
    
    
    def _each_result_mean_shift_t(self, img: torch.Tensor, model: Module):
        sample_list = [img]
        noise_list  = []
        noisy_list  = []
        mask_list   = []
        
        time_length = self.args.updated_ddpm_num_steps
        
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


    def _save_image_grid(self, sample: torch.Tensor, dir_save: str, file_sample: str):
        batch_size  = sample.shape[0]
        nrow        = int(np.ceil(np.sqrt(batch_size)))
        sample      = normalize01(sample)
        grid_sample = make_grid(sample, nrow=nrow, normalize=True)
        file_sample = os.path.join(dir_save, file_sample)
        save_image(grid_sample, file_sample)
        
    
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
