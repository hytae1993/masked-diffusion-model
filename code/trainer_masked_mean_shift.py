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
from collections import OrderedDict

from tqdm.auto import tqdm

from sampler import Sampler
from scheduler import Scheduler
from utils.datautils import normalize01
from utils import util

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
        
        self.visual_names       = ['input','degraded_img', 'degradation_mask', 'shifted_degrade_img', 'shifted_input', 'mask', 'reconstructed_img', 'inverse_shifted_reconstrucion', \
                                    'sample_result', 'sample_trained_x_0_list', 'sample_trained_t_list', 'sample_trained_mask_list', \
                                    'ema_sample_result', 'ema_sample_trained_x_0_list', 'ema_sample_trained_t_list', 'ema_sample_trained_mask_list']
        
        self.loss_names         = ['reconstruct_loss', 'learning_rate', 'mean']
        
        self.timesteps_used_epoch   = None
        
    def _compute_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        loss    = self.criterion(prediction, target)
        
        return loss
    
    def _shift_mean(self, img: torch.Tensor):
        mean    = img.mean(dim=(1,2,3), keepdim=True)
        img     = img - mean
        
        return img
         
    def _run_batch(self, batch: int, input, epoch: int, epoch_length: int, resume_step: int, dirs: dict, visualizer):
        # print(input)
        # time
        if 'huggingface' in self.args.dir_dataset:
            try:
                input     = input["image"]
                self.label   = input["label"]
            except KeyError:
                input     = input["image"]
        else:
            self.input     = input[0]

        self.input      = self.input.to(self.args.weight_dtype)
        self.input      = self._shift_mean(self.input)         # make each mean of image to zero
        
        # ===================================================================================
        # Create masks with random area black and obtation degraded image
        # ===================================================================================
        # timesteps           = torch.randint(low=1, high=self.args.updated_ddpm_num_steps+1, size=(input.shape[0],), device=input.device)
        timeindex           = torch.randint(low=0, high=len(self.timesteps_used_epoch), size=(self.input.shape[0],), device=self.input.device)
        timesteps           = torch.index_select(torch.tensor(self.timesteps_used_epoch, device=timeindex.device), 0, timeindex)
        
        black_area_num      = self.Scheduler.get_black_area_num_pixels_time(timesteps)      # get number of removed pixels at each timestep 
        
        self.degraded_img, self.degradation_mask  = self.Scheduler.degrade_training(black_area_num, self.input, mean_option=self.args.mean_option)
        
        # ===================================================================================
        # shift 
        # ===================================================================================
        shift                       = self.Scheduler.get_schedule_shift_time(timesteps) 
        self.shifted_degrade_img    = self.Scheduler.perturb_shift(self.degraded_img, shift)
        self.shifted_input          = self.Scheduler.perturb_shift(self.input, shift)
        
        # ===================================================================================
        # reconstruct and train 
        # ===================================================================================
        with self.accelerator.accumulate(self.model):
            self.mask               = self.model(self.shifted_degrade_img, timesteps).sample
            self.reconstructed_img  = self.shifted_degrade_img + self.mask
            
            self.reconstruct_loss   = self._compute_loss(self.reconstructed_img, self.shifted_input)
            
            self.accelerator.backward(self.reconstruct_loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        if self.accelerator.sync_gradients:
            if self.args.use_ema:
                self.ema_model.step(self.model.parameters())
            self.global_step += 1
            
            if self.accelerator.is_main_process:
                if self.global_step % self.args.checkpointing_steps == 0:
                    save_path   = os.path.join(dirs.list_dir['checkpoint'], f"checkpoint-{self.global_step}")
                    self.accelerator.save_state(save_path)
        
        # ===================================================================================
        # inverse shift 
        # ===================================================================================
        self.inverse_shifted_reconstrucion  = self.Scheduler.perturb_shift_inverse(self.reconstructed_img, shift)
        self.mean   = self.inverse_shifted_reconstrucion.mean()
        
        self.learning_rate  = self.lr_scheduler.get_last_lr()[0]
        self.lr_list.append(self.learning_rate)
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process and visualizer is not None:
            losses = self.get_current_losses()
            visualizer.plot_current_losses(epoch, losses)
        
        return self.reconstruct_loss.item(), self.mean.item()


    def _run_epoch(self, epoch: int, epoch_length: int, resume_step: int, dirs: dict, visualizer):
        loss_batch              = []
        mean_batch              = []
        
        batch_progress_bar   = tqdm(total=len(self.dataloader), disable=not self.accelerator.is_local_main_process, leave=False)
        batch_progress_bar.set_description(f"Batch ")
        
        self.timesteps_used_epoch     = self.Scheduler.get_timesteps_epoch(epoch, epoch_length)
        
        for i, input in enumerate(self.dataloader, 0):
            # time
            loss, mean = self._run_batch(i, input, epoch, epoch_length, resume_step, dirs, visualizer)
            batch_progress_bar.update(1)
            
            if self.accelerator.is_main_process: 
                
                loss_batch.append(loss)
                mean_batch.append(mean)
                
        batch_progress_bar.close()
        return loss_batch, mean_batch
    
    
    def train(self, epoch_start: int, epoch_length: int, resume_step: int, global_step: int, dirs: dict, visualizer):
        
        updated_ddpm_num_steps              = self.Scheduler.update_ddpm_num_steps(self.args.ddpm_num_steps)
        self.args.updated_ddpm_num_steps    = updated_ddpm_num_steps
        
        epoch_length    = epoch_length
        epoch_start     = epoch_start
        self.global_step = global_step
        
        loss_mean_epoch = []
        loss_std_epoch  = []
       
        epoch_progress_bar   = tqdm(total=epoch_length, disable=not self.accelerator.is_local_main_process)
        epoch_progress_bar.set_description(f"Epoch ")
        self.model.train()
        for epoch in range(epoch_start,epoch_start+epoch_length):
            start = timer()
            if self.accelerator.is_main_process and visualizer is not None:
                visualizer.reset()
            # self.dataloader.batch_sampler.batch_sampler.sampler.set_epoch(epoch)
            loss, mean = self._run_epoch(epoch, epoch_length, resume_step, dirs, visualizer)
            
            end = timer()
            elapsed_time = end - start
            
            if self.accelerator.is_main_process:
                
                loss_mean       = statistics.mean(loss)
                loss_std        = statistics.stdev(loss, loss_mean)
                self.reconstruct_loss   = loss_mean
                
                mean_mean       = statistics.mean(mean)
                self.mean       = mean_mean

                loss_mean_epoch.append(loss_mean)
                loss_std_epoch.append(loss_std)
                
                if epoch > 0 and epoch % self.args.save_images_epochs == 0 or epoch == (epoch_start+epoch_length-1) or (epoch+1) % (epoch_length / self.args.scheduler_num_scale_timesteps) == 0:
                # if epoch == epoch_start or epoch % self.args.save_images_epochs == 0 or epoch == (epoch_start+epoch_length-1) or (epoch+1) % (epoch_length / self.args.scheduler_num_scale_timesteps) == 0:
    
                    self._save_model(dirs, epoch)
                    self._save_sample(dirs, epoch)
                    if self.args.use_ema:
                        self._save_ema_sample(dirs, epoch)
                    # save to wandb
                    if visualizer is not None:
                        visualizer.display_current_results(self.get_current_visuals(), epoch)
            
            epoch_progress_bar.update(1)
        
        epoch_progress_bar.close()
        
    
    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
    
    
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
 
    def _save_sample(self, dirs, epoch):
        dir_save            = dirs.list_dir['sample_img'] 

        # sample, sample_list, t_list, t_mask, next_t_mask, t_mask_list = self.Sampler.sample(self.model.eval(), self.timesteps_used_epoch)
        sample, t_list, t_mask_list, sample_list  = self.Sampler.sample(self.model.eval(), self.timesteps_used_epoch)
        file_save                       = 'sample_{:05d}.png'.format(epoch)
        self.sample_result              = self.Sampler._save_image_grid(sample, dir_save, file_save)
        
        self.sample_trained_x_0_list    = self.Sampler._save_multi_index_image_grid(sample_list, option='skip_first')    # result of x_0 for each t
        self.sample_trained_t_list      = self.Sampler._save_multi_index_image_grid(t_list)         # result of each t
        self.sample_trained_mask_list   = self.Sampler._save_multi_index_image_grid(t_mask_list)
        
        
    def _save_ema_sample(self, dirs, epoch):
        dir_sample_save            = dirs.list_dir['ema_sample_img']
        dir_sample_all_t_save      = dirs.list_dir['ema_sample_all_t_img']
        
        self.ema_model.store(self.model.parameters())
        # model_ema.parameters => model.parameters
        self.ema_model.copy_to(self.model.parameters())
        
        # ema_sample, ema_sample_list, ema_t_list, ema_t_mask, ema_next_t_mask, ema_mask_list = self.Sampler.sample(self.model.eval(), self.timesteps_used_epoch)
        ema_sample, ema_t_list, ema_mask_list, ema_sample_list  = self.Sampler.sample(self.model.eval(), self.timesteps_used_epoch)
        # model_ema.temp => model.parameters
        self.ema_model.restore(self.model.parameters())
        
        file_ema_save                       = 'ema_sample_{:05d}.png'.format(epoch)
        self.ema_sample_result              = self.Sampler._save_image_grid(ema_sample, dir_sample_save, file_ema_save)
        
        self.ema_sample_trained_x_0_list    = self.Sampler._save_multi_index_image_grid(ema_sample_list, option='skip_first')
        self.ema_sample_trained_t_list      = self.Sampler._save_multi_index_image_grid(ema_t_list)
        self.ema_sample_trained_mask_list   = self.Sampler._save_multi_index_image_grid(ema_mask_list)


    def _save_model(self, dirs: dict, epoch: int):
        '''
        https://huggingface.co/docs/accelerate/usage_guides/checkpoint
        '''
        filename    = 'model_epoch_{:05d}'.format(epoch)
        dir_save    = dirs.list_dir['model'] 
        filename    = os.path.join(dir_save, filename)
        
        # self.accelerator.save_state(filename)
        
        # unet        = self.accelerator.unwrap_model(self.model)
        self.accelerator.save_model(self.model, filename)

        
    def _save_loss(self, dirs: dict, loss_generator_mean, loss_generator_std, loss_discriminator_mean, loss_discriminator_std):
        filename    = 'loss.csv'
        dir_save    = dirs.list_dir['loss'] 
        filename    = os.path.join(dir_save, filename)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(loss_generator_mean)
            writer.writerow(loss_generator_std)
            writer.writerow(loss_discriminator_mean)
            writer.writerow(loss_discriminator_std)
        f.close()
        
        
    def _save_log(self, dirs: dict):
        filename = 'progress.log'
        dir_save    = dirs.list_dir['log'] 
        filename    = os.path.join(dir_save, filename)
        with open(filename, 'a', newline='') as f:
            f.write(self.log)
        f.close()
    