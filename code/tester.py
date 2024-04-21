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
        
        self.cosine_similarity_th  = 0.80
        
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
            
            total_unique_images     = []
            num_total_unique_images = []
            
            dataloader  = DataLoader(self.dataset, batch_size=1, drop_last=False, shuffle=False)
            train_set   = torch.zeros(self.args.data_subset_num, 3, self.args.data_size, self.args.data_size)
            for i, (data, label) in enumerate(dataloader):
                data            = normalize01(data)
                train_set[i]    = data
            
            img_set = [[] for _ in range(self.args.data_subset_num)]
            
            idx = 0
            while(len(total_unique_images) < self.args.data_subset_num):
                self.ema_model.store(self.model.parameters())
                # model_ema.parameters => model.parameters
                self.ema_model.copy_to(self.model.parameters())
                sample_0    = self.Sampler.sample(self.model.eval(), self.timesteps_used_epoch)
                # sample_0, _, _ = self.Sampler.sample(self.model.eval(), self.timesteps_used_epoch)
                # model_ema.temp => model.parameters
                self.ema_model.restore(self.model.parameters())
            
                generated_image = [sample_0[i].unsqueeze(dim=0) for i in range(sample_0.shape[0])]

                unique_in_batch     = self.remove_duplicates_in_batches(generated_image)
                unique_across_batch = self.remove_duplicates_across_batches(unique_in_batch, total_unique_images)
                total_unique_images.extend(unique_across_batch)
                num_total_unique_images.append(len(total_unique_images))
                
                unique_images   = torch.stack(total_unique_images)
                
                for i in range(int(unique_images.shape[0] / 100)+1):
                    try:
                        save_part       = unique_images[i*100:(i+1)*100]
                    except IndexError:
                        save_part       = unique_images[i*100:]
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
                max_idx = self.get_nearest_neighbor_idx(torch.stack(generated_image))
                img_set = self.get_similar_neighbor(generated_image, img_set, max_idx)
                # self.save_neighbor(img_set, train_set)
                
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
        unique_in_batch = []
        
        for idx, image in enumerate(current_batch):
            is_duplicate = False
            for i in range(len(current_batch)):
                if i == idx:
                    continue
                else:
                    similarity = self.cosine_similarity(image, current_batch[i])
                    if similarity > self.cosine_similarity_th:  
                        is_duplicate = True
                        break
            if not is_duplicate:
                unique_in_batch.append(image)
                
        return unique_in_batch
        

    def remove_duplicates_across_batches(self, unique_in_batch, previous_images):
        
        unique_images = []
        for idx, image in enumerate(unique_in_batch):
            is_duplicate = False
            for prev_image in previous_images:
                similarity = self.cosine_similarity(image, prev_image)
                if similarity > self.cosine_similarity_th:  
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_images.append(image)

        return unique_images
    
    
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
            similar = False
            if len(img_set[idx[i]]) > 0:
                for j in range(len(img_set[idx[i]])):
                    score   = self.cosine_similarity(generated_image[i], img_set[idx[i]][j])
                    if score > 0.95:
                        similar = True
                        break
                if not similar:
                    img_set[idx[i]].append(generated_image[i])
            else:
                img_set[idx[i]].append(generated_image[i])
                
        return img_set
    
    
    def save_neighbor(self, img_set, train_set):
        chunk   = math.ceil(self.args.data_subset_num / 10)
        
        for idx in range(chunk):
            print(idx)
            print("=================")
            try:
                train_list  = train_set[idx*10:(idx+1)*10]
                sample_list = img_set[idx*10:(idx+1)*10]
            except IndexError:
                train_list  = train_set[idx*10:]
                sample_list = img_set[idx*10:]
                
            image_set = []
            for i in range(len(train_list)):
                try:
                    image_set.append([train_list[i]] + sample_list[i])
                except IndexError:
                    image_set.append([train_list[i]])
            
            num_rows = len(image_set)
            num_cols = max(len(images) for images in image_set)

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

            for i, images in enumerate(image_set):
                for j, image in enumerate(images):
                    ax = axes[i, j] if num_rows > 1 else axes[j]
                    image   = normalize01(image)
                    image   = image.float().mul(255).add_(0.5).clamp_(0, 255)
                    try:
                        ax.imshow(image.squeeze().permute(1,2,0))
                    except TypeError:
                        image   = image.cpu()
                        ax.imshow(image.squeeze().permute(1,2,0))
                        
                    ax.axis('off')

            for i in range(num_rows):
                for j in range(len(image_set[i]), num_cols):
                    axes[i, j].axis('off')

            fig.tight_layout()
            fig.savefig('test_{}.png'.format(idx))