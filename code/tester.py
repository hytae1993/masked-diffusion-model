import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from matplotlib.ticker import MaxNLocator
import numpy as np
import csv
import statistics
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os
import cv2
import random
from collections import OrderedDict
import time
import math

from tqdm.auto import tqdm

from sampler import Sampler
from scheduler import Scheduler
from utils.datautils import normalize01
from utils import util

# ===============================================================================================
# Generete image with masked diffusion model - input: image & time - Base code of masked DDPM
# ===============================================================================================

class Tester:
    def __init__(self,
        args,
        dataloader,
        dataset,
        model,
        ema_model,
        optimizer,
        lr_scheduler, 
        accelerator, 
        ):
        self.args               = args
        self.dataloader         = dataloader
        self.dataset            = dataset
        self.model              = model
        self.ema_model          = ema_model
        self.accelerator        = accelerator
        
        self.Scheduler          = Scheduler(args)
        self.Sampler            = Sampler(self.dataset, self.args, self.Scheduler)
        
        self.cosine_similarity_th  = 0.9
        
        self.timesteps_used_epoch   = None
         
    def train(self, epoch_start: int, epoch_length: int, resume_step: int, global_step: int, dirs: dict, visualizer):
        updated_ddpm_num_steps              = self.Scheduler.update_ddpm_num_steps(self.args.ddpm_num_steps)
        self.args.updated_ddpm_num_steps    = updated_ddpm_num_steps
        self.time_steps                     = self.Scheduler.get_black_area_num_pixels_all()
        
        self.timesteps_used_epoch   = self.Scheduler.get_timesteps_epoch(1, 10)
        
        epoch_length    = epoch_length
        epoch_start     = epoch_start
        self.global_step = global_step
        
        if self.accelerator.is_main_process: 
            
            dir_sample_img_save         = dirs.list_dir['test_sample_img']
            dir_sample_num_save         = dirs.list_dir['test_sample_num']
            dir_sample_neighbor_save    = dirs.list_dir['test_sample_neighbor']
            
            total_unique_images     = torch.empty(0,3,self.args.data_size,self.args.data_size)
            num_total_unique_images = []
            
            dataloader  = DataLoader(self.dataset, batch_size=1, drop_last=False, shuffle=False)
            train_set   = torch.zeros(self.args.data_subset_num, 3, self.args.data_size, self.args.data_size)
            for i, (data, label) in enumerate(dataloader):
                data            = normalize01(data)
                train_set[i]    = data
            
            img_set = [torch.empty(0,3,self.args.data_size,self.args.data_size) for _ in range(self.args.data_subset_num)]
            
            idx = 0
            while(len(total_unique_images) < self.args.data_subset_num):
                self.ema_model.store(self.model.parameters())
                # model_ema.parameters => model.parameters
                self.ema_model.copy_to(self.model.parameters())
                generated_image = self.Sampler.sample(self.model.eval(), self.timesteps_used_epoch)
                # sample_0, _, _ = self.Sampler.sample(self.model.eval(), self.timesteps_used_epoch)
                # model_ema.temp => model.parameters
                self.ema_model.restore(self.model.parameters())
            
                unique_in_batch     = self.remove_duplicates_in_batches(generated_image)
                unique_across_batch = self.remove_duplicates_across_batches(unique_in_batch, total_unique_images)
                total_unique_images = torch.cat((total_unique_images, unique_across_batch.cpu()),dim=0)
                num_total_unique_images.append(total_unique_images.shape[0])
                
                
                
                for i in range(int(total_unique_images.shape[0] / 100)+1):
                    try:
                        save_part       = total_unique_images[i*100:(i+1)*100]
                    except IndexError:
                        save_part       = total_unique_images[i*100:]
                    sample          = self.Sampler._save_image_grid(save_part.squeeze(dim=1), normalization='image')
                    img_name        = os.path.join(dir_sample_img_save, 'sample_{}_{}.png'.format(idx, i))
                    if sample != None:
                        save_image(sample, img_name)
                
                plot_name   = os.path.join(dir_sample_num_save, 'number_of_sample.png')
                plt.plot(num_total_unique_images)
                plt.savefig(plot_name)
                
                # sampling 이미지 하나와 dataset 전체와 비교하여 가장 가까운 이미지 찾기
                # 이때, 가장 가까운 이미지는 cosine similarity가 가장 큰 값 기준
                # 가장 큰 이미지와 매칭할 때, 이미 매칭된 이미지가 있다면 매칭되어 있는 이미지와의 cs 계산, cs가 일정 값 이하면 추가
                max_idx = self.get_nearest_neighbor_idx(unique_across_batch)
                img_set = self.get_similar_neighbor(unique_across_batch.cpu(), img_set, max_idx)
                self.save_neighbor(img_set, train_set, dir_sample_neighbor_save)
                
                idx += 1
                
            unique_images = torch.stack(total_unique_images)
            
            sample = self.Sampler._save_image_grid(unique_images.squeeze(dim=1), normalization='image')
            img_name    = os.path.join(dir_sample_img_save, 'final_sample.png')
            save_image(sample, img_name)
            
            plot_name   = os.path.join(dir_sample_num_save, 'number_of_sample.png')
            plt.plot(num_total_unique_images)
            plt.savefig(plot_name)
            
            
    def cosine_similarity(self, image1, image2):
        image1_tensor = image1.flatten()
        image2_tensor = image2.flatten()
        return F.cosine_similarity(image1_tensor, image2_tensor, dim=0)
    
    
    def _compute_similarity(self, source: torch.Tensor, target: torch.Tensor, metric: str='cosine'):
            vec_source = nn.Flatten()(source)
            vec_target = nn.Flatten()(target)
            if metric.lower() == 'cosine':
                score = nn.functional.cosine_similarity(vec_source[None,:,:], vec_target[:,None,:], dim=2) 
            return score
    
    
    def remove_duplicates_in_batches(self, current_batch):
        
        unique_in_batch = [current_batch[0]]
        for image in current_batch[1:]:
            is_similar = False
            for existing_image in unique_in_batch:
                similarity_score = self.cosine_similarity(image, existing_image)
                if similarity_score >= self.cosine_similarity_th:
                    is_similar = True
                    break
            if not is_similar:
                unique_in_batch.append(image)
        return torch.stack(unique_in_batch)
        

    def remove_duplicates_across_batches(self, unique_in_batch, previous_images):
        
        unique_images = []
        for image in unique_in_batch:
            is_similar = False
            for prev_image in previous_images:
                try:
                    similarity = self.cosine_similarity(image, prev_image)
                except RuntimeError:
                    similarity = self.cosine_similarity(image.cpu(), prev_image)
                    
                if similarity > self.cosine_similarity_th:  
                    is_similar = True
                    break
            if not is_similar:
                unique_images.append(image)
        try:
            unique  = torch.stack(unique_images)
        except RuntimeError:
            unique  = torch.empty(0,3,self.args.data_size,self.args.data_size)
            
        return unique
    
    
    def get_nearest_neighbor_idx(self, source: torch.Tensor):
        batch_size  = source.shape[0]
        score       = torch.Tensor()
        score       = score.to(source.device)
        dataloader  = DataLoader(self.dataset, batch_size=self.args.sample_num, drop_last=False, shuffle=False)
        
        for i, (data, label) in enumerate(dataloader):
            data = normalize01(data)
            data = data.to(source.device)
            
            sim = self._compute_similarity(source, data, 'cosine')
            sim = sim.to(source.device)
            
            score = torch.cat((score, sim), dim=0)

        max_val, max_idx = score.max(dim=0)

        return max_idx
    
    
    def get_similar_neighbor(self, generated_image, img_set, idx):
        for i in range(len(generated_image)):
            is_similar = False
            if img_set[idx[i]].shape[0] > 0:
                for j in range(img_set[idx[i]].shape[0]):
                    score   = self.cosine_similarity(generated_image[i], img_set[idx[i]][j])
                    if score > self.cosine_similarity_th:
                        is_similar = True
                        break
                if not is_similar:
                    img_set[idx[i]] = torch.cat((img_set[idx[i]], generated_image[i].unsqueeze(dim=0)), dim=0)
            else:
                img_set[idx[i]] = torch.cat((img_set[idx[i]], generated_image[i].unsqueeze(dim=0)), dim=0)
                
        return img_set
    
    
    def save_neighbor(self, img_set, train_set, dir):
        chunk_length    = 10
        chunk           = math.ceil(self.args.data_subset_num / chunk_length)
        
        for idx in range(chunk):
            try:
                train_list  = train_set[idx*chunk_length:(idx+1)*chunk_length, :, :, :]
                sample_list = img_set[idx*chunk_length:(idx+1)*chunk_length]
            except IndexError:
                train_list  = train_set[idx*chunk_length:, :, :, :]
                sample_list = img_set[idx*chunk_length:]
                
            image_set = []
            for i in range(train_list.shape[0]):
                image_set.append(torch.cat((train_list[i].unsqueeze(dim=0), sample_list[i]), dim=0))
            
            num_rows = len(image_set)
            num_cols = max(len(images) for images in image_set)
            # if num_cols > 10:
            #     num_cols    = 10

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3*num_rows))
            
            
            for i, images in enumerate(image_set):
                if i > 10:
                    break
                row_idx = i
                for j, image in enumerate(images):
                    col_idx = j

                    image   = normalize01(image)
                    # image   = image.float().mul(255).add_(0.5).clamp_(0, 255)
                    
                    if len(image.shape) == 4: 
                        image = image[0].permute(1, 2, 0).cpu().numpy()  
                    else:
                        image = image.permute(1, 2, 0).cpu().numpy() 

                    if num_cols > 1:
                        ax = axes[row_idx, col_idx]
                    else:
                        ax = axes[row_idx]
                    
                    ax.imshow(image)
                    ax.axis('off')

                if len(images) == 1:
                    for k in range(1, num_cols):
                        fig.delaxes(axes[row_idx, k])

            plt.tight_layout()
            
            img_name    = os.path.join(dir, 'neighbor_{}.png'.format(idx))
            fig.savefig(img_name)