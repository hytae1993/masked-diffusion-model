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


class Sampler:
    def __init__(self, 
                dataset: Dataset, 
                args,
                Mask,
                ):

        self.dataset        = dataset
        self.args           = args 
        self.Mask           = Mask
      
        '''
        if evaluate: 
            self.metric_fid = FrechetInceptionDistance(feature=64).to(device)
        ''' 


    def compute_metric(self, sample: torch.Tensor, target: torch.Tensor):
        transform   = Resize(size=(299, 299), antialias=True)
        sample      = transform(sample)
        target      = transform(target)
        sample      = (sample * 255.0).to(torch.uint8)
        target      = (target * 255.0).to(torch.uint8)
        fid         = self.compute_fid(sample, target)
        fid         = fid.cpu().detach().numpy()
        metric      = {'fid': fid}
        return metric
   
    
    def compute_fid(self, sample: torch.Tensor, target: torch.Tensor):
        self.metric_fid.update(sample, real=False)
        self.metric_fid.update(target, real=True)
        fid = self.metric_fid.compute()
        return fid 


    def compute_fid_update(self, sample: torch.Tensor, target: torch.Tensor, metric):
        transform   = Resize(size=(299, 299), antialias=True)
        sample      = transform(sample)
        target      = transform(target)
        sample      = (sample * 255.0).to(torch.uint8)
        target      = (target * 255.0).to(torch.uint8)
        metric.update(sample, real=False)
        metric.update(target, real=True)
 
    '''
    def compute_fid_update(self, sample: torch.Tensor, target: torch.Tensor):
        transform   = Resize(size=(299, 299), antialias=True)
        sample      = transform(sample)
        target      = transform(target)
        sample      = (sample * 255.0).to(torch.uint8)
        target      = (target * 255.0).to(torch.uint8)
        self.metric_fid.update(sample, real=False)
        self.metric_fid.update(target, real=True)
    '''
  

    '''
    def compute_is(self, metric_is, sample: torch.Tensor, target: torch.Tensor):
        print('compute IS')
        inception_score = 0
        return inception_score
    '''
    
    def _get_latent_initial(self, model: Module):
        time_length = self.args.ddpm_num_steps
        time_length = torch.Tensor([time_length])
        time_length = time_length.expand(self.args.batch_size).to(model.device)
        
        # make all black tensor
        # latent      = self._generate_random_mask(self.args.batch_size, self.args.data_size, self.args.data_size, time_length)
        # latent      = model(latent, time_length).sample
        
        latent      = torch.zeros(self.args.batch_size, 3, self.args.data_size, self.args.data_size)
        
        return latent
    
    
    def sample(self, model: Module):
        sample, sample_list = self._sample(model) 
        return sample, sample_list


    def _sample(self, model: Module):
        time_length = self.args.updated_ddpm_num_steps
      
        latent      = self._get_latent_initial(model)
        sample      = latent.to(model.device)
        
        sample_per  = int(time_length / 5)
        sample_list = [sample]
        sample_t    = [time_length - sample_per, time_length - sample_per*2, time_length - sample_per*3, time_length - sample_per*4, time_length - sample_per*5]
        
        # time_length_batch = torch.Tensor([time_length])
        # time_length_batch = time_length_batch.expand(self.args.batch_size)

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
                    # black_area_ratio    = self._mask_schedule(time-1)
                    # noise               = self._generate_random_mask(self.args.batch_size, self.args.data_size, self.args.data_size, black_area_ratio).to(model.device)
                    black_area_ratio    = self.Mask.get_list_black_area_ratios(time-1)
                    noise               = self.Mask.get_mask(black_area_ratio)
                    noise               = noise.to(model.device)
                    sample              = prediction * noise
                    
                if t in sample_t:
                    sample_list.append(prediction)
                    
                sample_progress_bar.update(1)
            # sample_progress_bar.clear()
            sample_progress_bar.close()

        return sample, sample_list
    
    
    def sample_random_t(self, img: torch.Tensor, model: Module):
        sample_list = self._sample_random_t(img[0], model)
        return sample_list
    
    
    def _sample_random_t(self, img: torch.Tensor, model: Module):
        # img         = img.unsqueeze(dim=0)
        sample_list = [img.unsqueeze(dim=0)]
        time_length = self.args.updated_ddpm_num_steps
        # time_length = 5
        
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
                
    
    def _get_noisy(self, time: int, img: torch.Tensor, model: Module):
        time                = torch.tensor([time])
        # time                = torch.Tensor(time, device=model.device)
                
        black_area_ratio    = self.Mask.get_list_black_area_ratios(time)
        noise               = self.Mask.get_mask(black_area_ratio)
        noise               = noise.to(model.device)
        sample              = img * noise
        
        return sample

    def eval(self, model_info: dict, model: Module, num_batch: int):
        dim_channel = model_info['dim_channel']
        batch_size  = model_info['batch_size']
        dataloader  = DataLoader(self.dataset, batch_size=batch_size, drop_last=True, shuffle=True)

        # self.metric_fid.reset()
        list_fid = []
        list_metric_fid = []
        list_sample = []
         
        for n in range(num_batch):
            metric_fid = FrechetInceptionDistance(feature=64).to(self.device)
            metric_fid.reset()
            list_metric_fid.append(metric_fid)
        
        for i, (data, label) in enumerate(dataloader):
            if i == num_batch:
                break
            print('[evaluate] index of mini-batch:', i+1, '/', num_batch)
            data    = data.to(self.device)
            data    = normalize01(data)
            sample  = self._sample(model_info, model, batch_size) 
            sample  = sample.to(self.device)
            sample  = normalize01(sample)
 
            if dim_channel == 1: 
                data    = data.repeat(1,3,1,1)
                sample  = sample.repeat(1,3,1,1)
            
            for j in range(i, num_batch):
                transform   = Resize(size=(299, 299), antialias=True)
                sample_     = transform(sample.detach())
                data_       = transform(data.detach())
                sample_     = (sample_ * 255.0).to(torch.uint8)
                data_       = (data_ * 255.0).to(torch.uint8)
                list_metric_fid[j].update(sample_, real=False)
                list_metric_fid[j].update(data_, real=True)
                # self.compute_fid_update(sample.detach(), data.detach())
            list_sample.append(sample.detach().cpu())
             
        for n in range(num_batch):
            fid = list_metric_fid[n].compute()
            list_fid.append(fid.detach().cpu().numpy())
        # fid = self.metric_fid.compute()
        return list_fid, list_sample

        
    def get_nearest_neighbor(self, source: torch.Tensor, augment: bool=True, metric: str='cosine'):
        batch_size  = source.shape[0]
        score       = torch.Tensor()
        score       = score.to(self.device)
        dataloader  = DataLoader(self.dataset, batch_size=batch_size, drop_last=False, shuffle=False)
        transform   = transforms.Compose([transforms.RandomHorizontalFlip()]) 
        transform_resize    = transforms.Compose([transforms.Resize([32, 32])]) 
        source_small        = transform_resize(source)
        
        for i, (data, label) in enumerate(dataloader):
            data = normalize01(data)
            data = data.to(self.device)
            data_small = transform_resize(data)
            sim = self._compute_similarity(source_small, data_small, metric)
            sim = sim.to(self.device)

            if augment:
                data_aug = transform(data)
                data_aug_small = transform_resize(data_aug)
                sim_aug = self._compute_similarity(source_small, data_aug_small, metric)
                sim_aug = sim_aug.to(self.device)
                sim = torch.max(sim, sim_aug) 
            
            score = torch.cat((score, sim), dim=0)

        max_val, max_idx = score.max(dim=0)
        nearest_neighbor = torch.zeros_like(source) 

        for i in range(len(max_idx)):
            nearest = self.dataset[max_idx[i]][0]
            nearest_neighbor[i] = nearest
        return nearest_neighbor
        
       
    def save_real(self, dir_save: str, num_data: int=100):
        dataloader  = DataLoader(self.dataset, batch_size=num_data, drop_last=True, shuffle=True)
        data, label = next(iter(dataloader))
        real = normalize01(data)
        for i in range(num_data):
            file_save = 'real_{:03d}.png'.format(i+1)   # real_001, real_002, ..., real_100
            file_save = os.path.join(dir_save, file_save)
            save_image(real[i], file_save)
            
  
    def _compute_similarity(self, source: torch.Tensor, target: torch.Tensor, metric: str='cosine'):
        vec_source = nn.Flatten()(source)
        vec_target = nn.Flatten()(target)
        if metric.lower() == 'cosine':
            score = nn.functional.cosine_similarity(vec_source[None,:,:], vec_target[:,None,:], dim=2) 
        return score
       
        
    def _save_image_grid(self, sample: torch.Tensor, dir_save: str, file_sample: str):
        batch_size  = sample.shape[0]
        nrow        = int(np.ceil(np.sqrt(batch_size)))
        grid_sample = make_grid(sample, nrow=nrow, normalize=True)
        file_sample = os.path.join(dir_save, file_sample)
        save_image(grid_sample, file_sample)
        
    
    def _save_image_multi_grid(self, sample: list, dir_save: str, file_sample: str):
        batch_size  = sample[0].shape[0]
        nrow        = int(np.ceil(np.sqrt(batch_size)))
        grid_name   = os.path.join(dir_save, file_sample)
        
        grid1       = make_grid(sample[0], nrow=nrow)
        grid2       = make_grid(sample[1], nrow=nrow, normalize=True)
        grid3       = make_grid(sample[2], nrow=nrow, normalize=True)
        grid4       = make_grid(sample[3], nrow=nrow, normalize=True)
        grid5       = make_grid(sample[4], nrow=nrow, normalize=True)
        grid6       = make_grid(sample[5], nrow=nrow, normalize=True)
        
        grid1       = (grid1.cpu().numpy() * 255).round().astype("uint8")
        grid2       = (grid2.cpu().numpy() * 255).round().astype("uint8")
        grid3       = (grid3.cpu().numpy() * 255).round().astype("uint8")
        grid4       = (grid4.cpu().numpy() * 255).round().astype("uint8")
        grid5       = (grid5.cpu().numpy() * 255).round().astype("uint8")
        grid6       = (grid6.cpu().numpy() * 255).round().astype("uint8")
        
        fig, axarr = plt.subplots(2,3) 
        axarr[0][0].imshow(grid1.transpose((1,2,0)))
        axarr[0][1].imshow(grid2.transpose((1,2,0)))
        axarr[0][2].imshow(grid3.transpose((1,2,0)))
        axarr[1][0].imshow(grid4.transpose((1,2,0)))
        axarr[1][1].imshow(grid5.transpose((1,2,0)))
        axarr[1][2].imshow(grid6.transpose((1,2,0)))
        
        axarr[0][0].axis("off")
        axarr[0][1].axis("off")
        axarr[0][2].axis("off")
        axarr[1][0].axis("off")
        axarr[1][1].axis("off")
        axarr[1][2].axis("off")
        
        plt.tight_layout()
        fig.suptitle('T ------> 0',fontweight ="bold") 
        fig.savefig(grid_name)
        plt.close(fig)
       
        
    def _save_image_pair_grid(self, data1: torch.Tensor, data2: torch.Tensor, dir_save: str, file_save: str):
        batch_size  = data1.shape[0]
        data = torch.zeros(batch_size*2, data1.shape[1], data1.shape[2], data1.shape[3])
        for i in range(batch_size):
            data[2*i]   = data1[i]
            data[2*i+1] = data2[i]
        nrow_batch  = int(np.ceil(np.sqrt(batch_size))) * 2
        grid_data   = make_grid(data, nrow=nrow_batch, normalize=True)
        file_save   = os.path.join(dir_save, file_save)
        save_image(grid_data, file_save)


    def _save_plot(self, curve_mean, curve_std, title: str, dir_save: str, file_save: str):
        file_save = os.path.join(dir_save, file_save)
        fig = plt.figure()
        plt.title(title)
        plt.plot(curve_mean, color='red')
        plt.fill_between(list(range(curve_mean.size)), curve_mean-curve_std, curve_mean+curve_std, color='blue', alpha=0.2)
        plt.tight_layout()
        plt.savefig(file_save, bbox_inches='tight', dpi=100)
        plt.close(fig)


    def _save_image_list(self, image_list: torch.Tensor, dir_save: str, file_save_prefix: str, file_save_suffix: str=''):
        num_image = image_list.shape[0]
        for i in range(num_image):
            file_save = '{:s}_{:03d}{:s}.png'.format(file_save_prefix, i+1, file_save_suffix)
            file_save = os.path.join(dir_save, file_save)
            save_image(image_list[i], file_save)