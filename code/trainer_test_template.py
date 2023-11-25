import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.utils import make_grid

from matplotlib.ticker import MaxNLocator
import numpy as np
import csv
import statistics
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os
import cv2
import random

from tqdm.auto import tqdm

from sampler import Sampler
from scheduler import Scheduler
from utils.datautils import normalize01

# from accelerate import data_loader

# ===============================================================================================
# Generete image with masked diffusion model - input: image & time - Base code of masked DDPM
# ===============================================================================================

class Trainer:
    def __init__(self,
        args,
        dataloader,
        model,
        ema_model,
        optimizer,
        lr_scheduler, 
        accelerator, 
        ):
        self.args               = args
        self.dataloader         = dataloader
        self.model              = model
        self.ema_model          = ema_model
        self.optimizer          = optimizer
        self.lr_scheduler       = lr_scheduler
        self.lr_list            = []
        self.accelerator        = accelerator
        
        self.criterion          = nn.MSELoss()
        
        self.Scheduler          = Scheduler(args)
        self.Sampler            = Sampler(self.dataloader, self.args, self.Scheduler)
        
        self.global_step        = 0
        
        # data_loader._PYTORCH_DATALOADER_KWARGS["shuffle"] = True
        
    def _compute_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        loss    = self.criterion(prediction, target)
        
        return loss
         
    def _run_batch(self, batch: int, input, epoch: int, epoch_length: int, resume_step: int, dirs: dict):
        # print(input)
        if 'huggingface' in self.args.dir_dataset:
            img     = input["image"]
            label   = input["label"]
        else:
            img     = input
            
        img         = img.to(self.args.weight_dtype)
        
        
        with self.accelerator.accumulate(self.model):
            
            # if self.accelerator.sync_gradients:
            #     self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
            self.lr_scheduler.step()
        
        if self.accelerator.sync_gradients:
            if self.args.use_ema:
                self.ema_model.step(self.model.parameters())
            self.global_step += 1
            
            if self.accelerator.is_main_process:
                if self.global_step % self.args.checkpointing_steps == 0:
                    save_path   = os.path.join(dirs.list_dir['checkpoint'], f"checkpoint-{self.global_step}")
                    self.accelerator.save_state(save_path)
        
        lr  = self.lr_scheduler.get_last_lr()[0]
        self.lr_list.append(lr)
        self.accelerator.wait_for_everyone()
        
        return img


    def _run_epoch(self, epoch: int, epoch_length: int, resume_step: int, dirs: dict):
        
        batch_progress_bar   = tqdm(total=len(self.dataloader), disable=not self.accelerator.is_local_main_process, leave=False)
        batch_progress_bar.set_description(f"Batch ")
        
        
        for i, input in enumerate(self.dataloader, 0):
            
            img = self._run_batch(i, input, epoch, epoch_length, resume_step, dirs)
            batch_progress_bar.update(1)
            
        batch_progress_bar.close()
        return img
    
    
    def train(self, epoch_start: int, epoch_length: int, resume_step: int, global_step: int, dirs: dict):
        
        epoch_length    = epoch_length
        epoch_start     = epoch_start
       
        self.model.train()
        
        self.global_step = global_step
       
        epoch_progress_bar   = tqdm(total=epoch_length, disable=not self.accelerator.is_local_main_process)
        epoch_progress_bar.set_description(f"Epoch ")
        for epoch in range(epoch_start,epoch_start+epoch_length):
            start = timer()
            img = self._run_epoch(epoch, epoch_length, resume_step, dirs)
            
            end = timer()
            elapsed_time = end - start
            
            if self.accelerator.is_main_process:

                #     self._save_model(dirs, epoch)
                self._save_result_image(dirs, img, epoch)
                #     self._save_inference_image(dirs, inference_image_set, epoch)
                #     self._save_black_image(dirs, black_image_set, epoch)
                #     self._save_sample(dirs, epoch)
                #     self._save_sample_random_t(dirs, img_set[0], epoch)
                #     self._save_learning_curve(dirs, loss_mean_epoch, loss_std_epoch)
                #     self._save_time_step(dirs, timesteps_count, epoch)
            
            epoch_progress_bar.update(1)
        
        epoch_progress_bar.close()
        
    
    def _save_result_image(self, dirs, img, epoch):
        input       = img    # input image
        batch_size  = input.shape[0]
        nrow        = int(np.ceil(np.sqrt(batch_size)))
        
        input_dir_save      = dirs.list_dir['train_img'] 
        file_input          = 'input_epoch_{:05d}.png'.format(epoch)
        file_input          = os.path.join(input_dir_save, file_input)
        # input               = normalize01(input)
        grid_input          = make_grid(input, nrow=nrow, normalize=True)
        save_image(grid_input, file_input)
            