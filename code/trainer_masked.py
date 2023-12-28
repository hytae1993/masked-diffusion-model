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
        
        if self.args.sampling == 'momentum':
            if self.args.loss_space == 'x_0':
                self.visual_names       = ['input','degraded_img', 'degradation', 'mask', 'reconstructed_img', \
                                        'sample_result', 'sample_trained_x_0_list', 'sample_trained_t_list', 'sample_trained_mask_list', 't_mask', 'next_t_mask', \
                                        'ema_sample_result', 'ema_sample_trained_x_0_list', 'ema_sample_trained_t_list', 'ema_sample_trained_mask_list', 'ema_t_mask', 'ema_next_t_mask']
            elif self.args.loss_space == 'time':
                self.visual_names       = ['input','degraded_img', 'degradation', 'mask', 'reconstructed_img', 're_degraded_img', \
                                        'sample_result', 'sample_trained_x_0_list', 'sample_trained_t_list', 'sample_trained_mask_list', 't_mask', 'next_t_mask', \
                                        'ema_sample_result', 'ema_sample_trained_x_0_list', 'ema_sample_trained_t_list', 'ema_sample_trained_mask_list', 'ema_t_mask', 'ema_next_t_mask']
        elif self.args.sampling == 'base':
            if self.args.loss_space == 'x_0':
                self.visual_names       = ['input','degraded_img', 'degradation', 'mask', 'reconstructed_img', \
                                        'sample_trained_t', 'sample_trained_t_list', 'ema_sample_trained_t', 'ema_sample_trained_t_list']
            elif self.args.loss_space == 'time':
                self.visual_names       = ['input','degraded_img', 'degradation', 'mask', 'reconstructed_img', 're_degraded_img', \
                                        'sample_trained_t', 'sample_trained_t_list', 'ema_sample_trained_t', 'ema_sample_trained_t_list']
                
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
        if 'huggingface' in self.args.dir_dataset:
            try:
                self.input     = input["image"]
                self.label   = input["label"]
            except KeyError:
                self.input     = input["image"]
        else:
            self.input     = input[0]

        self.input                 = self.input.to(self.args.weight_dtype)
        self.input                 = self._shift_mean(self.input)         # make each mean of image to zero
        
        # ===================================================================================
        # Create masks with random area black and obtation degraded image
        # ===================================================================================
        # timesteps           = torch.randint(low=1, high=self.args.updated_ddpm_num_steps+1, size=(self.input.shape[0],), device=self.input.device)
        timeindex           = torch.randint(low=0, high=len(self.timesteps_used_epoch), size=(self.input.shape[0],), device=self.input.device)
        timesteps           = torch.index_select(torch.tensor(self.timesteps_used_epoch, device=timeindex.device), 0, timeindex)
        timesteps_count     = torch.bincount(timesteps, minlength=self.args.updated_ddpm_num_steps+1)[1:]
        T_steps             = torch.where(timesteps == self.args.updated_ddpm_num_steps)
        inference_t_steps   = torch.where(timesteps > int(self.args.updated_ddpm_num_steps/2))
        
        black_area_num      = self.Scheduler.get_black_area_num_pixels_time(timesteps)      # get number of removed pixels at each timestep 
        
        self.degraded_img, self.degradation, black_idx, _   = self.Scheduler.get_mean_mask(black_area_num, self.input)
        
        with self.accelerator.accumulate(self.model):
            self.mask               = self.model(self.degraded_img, timesteps).sample
            self.reconstructed_img  = self.degraded_img + self.mask
            
            self.reconstruct_loss   = self._compute_loss(self.reconstructed_img, self.input)
            
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
        
        self.mean   = self.reconstructed_img.mean()
        
        self.learning_rate  = self.lr_scheduler.get_last_lr()[0]
        self.lr_list.append(self.learning_rate)
        self.accelerator.wait_for_everyone()
        
        black_image_index       = None
        if len(T_steps[0]) > 0:
            black_image_index   = T_steps[0]
            
        inference_image_index, inference_check_set  = None, None
        if len(inference_t_steps[0]) > 0:
            inference_image_index   = inference_t_steps[0][0]
            inference_check_set = [inference_image_index, timesteps[inference_image_index]]
                                
        img_set             = [self.input, self.degradation, self.degraded_img, self.mask, self.reconstructed_img]
        
        if self.accelerator.is_main_process and visualizer is not None:
            losses = self.get_current_losses()
            visualizer.plot_current_losses(epoch, losses)
        
        return img_set, self.reconstruct_loss.item(), self.mean.item(), timesteps_count, black_image_index, inference_check_set


    def _run_epoch(self, epoch: int, epoch_length: int, resume_step: int, dirs: dict, visualizer):
        loss_batch              = []
        mean_batch              = []
        epoch_timesteps_count   = torch.zeros(self.args.updated_ddpm_num_steps, dtype=torch.int)
        
        batch_progress_bar   = tqdm(total=len(self.dataloader), disable=not self.accelerator.is_local_main_process, leave=False)
        batch_progress_bar.set_description(f"Batch ")
        
        black_image_set     = [[],[],[],[],[]]
        inference_image_set = [[],[],[]]
        
        self.timesteps_used_epoch     = self.Scheduler.get_timesteps_epoch(epoch, epoch_length)
        
        for i, input in enumerate(self.dataloader, 0):
            img_set, loss, mean, batch_timesteps_count, black_index, inference_set = self._run_batch(i, input, epoch, epoch_length, resume_step, dirs, visualizer)
            batch_progress_bar.update(1)
            
            if self.accelerator.is_main_process: 
                
                loss_batch.append(loss)
                mean_batch.append(mean)
                epoch_timesteps_count += batch_timesteps_count.cpu()
                
                if len(black_image_set[0]) < self.args.batch_size and black_index is not None:
                    black_image_set[0].append(img_set[0][black_index])
                    black_image_set[1].append(img_set[2][black_index])
                    black_image_set[2].append(img_set[3][black_index])
                    black_image_set[3].append(img_set[4][black_index])
                    black_image_set[4].append(img_set[1][black_index])
                    
                if len(inference_image_set[0]) < self.args.batch_size and inference_set is not None:
                    inference_image_set[0].append(img_set[0][inference_set[0]:inference_set[0]+1]) # input image of time T
                    inference_image_set[1].append(img_set[4][inference_set[0]:inference_set[0]+1]) # prediction image of time T
                    inference_image_set[2].append(inference_set[1].item())             # time T

        batch_progress_bar.close()
        return img_set, loss_batch, mean_batch, epoch_timesteps_count, black_image_set, inference_image_set
    
    
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
            img_set, loss, mean, timesteps_count, black_image_set, inference_image_set = self._run_epoch(epoch, epoch_length, resume_step, dirs, visualizer)
            
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
                    self._save_learning_curve(dirs, loss_mean_epoch, loss_std_epoch)
                    self._save_time_step(dirs, timesteps_count, epoch)
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
    
        
    def _save_learning_curve(self, dirs, loss_mean, loss_std):
        dir_save    = dirs.list_dir['train_loss'] 
        # file_loss = 'loss_epoch_{:05d}.png'.format(epoch)
        file_loss = 'loss.png'
        file_loss = os.path.join(dir_save, file_loss)
        fig = plt.figure(figsize=(16, 8))
        
        plt.subplot(1,2,1)
        plt.plot(np.array(loss_mean), color='red')
        plt.fill_between(list(range(len(loss_mean))), np.array(loss_mean)-np.array(loss_std), np.array(loss_mean)+np.array(loss_std), color='blue', alpha=0.2)
        plt.title('loss')
        
        plt.subplot(1,2,2)
        plt.plot(np.array(self.lr_list), color='red')
        plt.title('learning rate')
        
        plt.tight_layout()
        plt.savefig(file_loss, bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        
    def _save_time_step(self, dirs, time_step, epoch: int):
        black_area_pixels   = self.Scheduler.get_black_area_num_pixels_all()
        dir_save            = dirs.list_dir['time_step'] 
        # file_loss = 'loss_epoch_{:05d}.png'.format(epoch)
        file_loss           = 'time_step_{}.png'.format(epoch)
        file_loss           = os.path.join(dir_save, file_loss)
        
        self.time_step      = np.array(time_step)
        time                = range(1, len(time_step) + 1)
        
        fig = plt.figure(figsize=(16,8))
        
        plt.subplot(2,1,1)
        plt.plot(time, self.time_step, color='red')
        plt.title('number of time step')
        
        plt.subplot(2,1,2)
        plt.plot(time, black_area_pixels, color='red')
        plt.title('number of pixels in each time step')
        
        plt.tight_layout()
        plt.savefig(file_loss, bbox_inches='tight', dpi=100)
        plt.close(fig)
 
 
    def _save_sample(self, dirs, epoch):
        dir_save            = dirs.list_dir['sample_img'] 

        sample, sample_list, t_list, t_mask, next_t_mask, t_mask_list = self.Sampler.sample(self.model.eval(), self.timesteps_used_epoch)
        # sample      = normalize01(sample)
        file_save                   = 'sample_{:05d}.png'.format(epoch)
        self.sample_result          = self.Sampler._save_image_grid(sample, dir_save, file_save)
        if self.args.sampling == 'momentum':
            self.t_mask                 = self.Sampler._save_image_grid(t_mask)
            self.next_t_mask            = self.Sampler._save_image_grid(next_t_mask)
        
        # sample_list                 = torch.cat(sample_list, dim=0)
        self.sample_trained_x_0_list    = self.Sampler._save_multi_index_image_grid(sample_list, option='skip_first')    # result of x_0 for each t
        # self.sample_trained_t_list  = self.Sampler._save_image_grid(sample_list, None, None)
        # self.sample_trained_t_list  = util.make_multi_grid(sample_list, nrow=3, ncol=3)
        self.sample_trained_t_list      = self.Sampler._save_multi_index_image_grid(t_list)         # result of each t
        self.sample_trained_mask_list      = self.Sampler._save_multi_index_image_grid(t_mask_list)
        
        
    def _save_ema_sample(self, dirs, epoch):
        dir_sample_save            = dirs.list_dir['ema_sample_img']
        dir_sample_all_t_save      = dirs.list_dir['ema_sample_all_t_img']
        
        self.ema_model.store(self.model.parameters())
        # model_ema.parameters => model.parameters
        self.ema_model.copy_to(self.model.parameters())
        
        ema_sample, ema_sample_list, ema_t_list, ema_t_mask, ema_next_t_mask, ema_mask_list = self.Sampler.sample(self.model.eval(), self.timesteps_used_epoch)
        # model_ema.temp => model.parameters
        self.ema_model.restore(self.model.parameters())
        
        file_ema_save                   = 'ema_sample_{:05d}.png'.format(epoch)
        self.ema_sample_result          = self.Sampler._save_image_grid(ema_sample, dir_sample_save, file_ema_save)
        if self.args.sampling == 'momentum':
            self.ema_t_mask                 = self.Sampler._save_image_grid(ema_t_mask)
            self.ema_next_t_mask            = self.Sampler._save_image_grid(ema_next_t_mask)
        
        # file_ema_all_t_save             = 'ema_sample_all_t_{:05d}.png'.format(epoch)
        # ema_sample_all_t                = torch.cat(ema_sample_all_t, dim=0)
        # self.ema_sample_trained_t_list  = self.Sampler._save_image_grid(ema_sample_all_t, dir_sample_all_t_save, file_ema_all_t_save)
        # self.ema_sample_trained_t_list  = util.make_multi_grid(ema_sample_all_t, nrow=3, ncol=3)    
        self.ema_sample_trained_x_0_list    = self.Sampler._save_multi_index_image_grid(ema_sample_list, option='skip_first')
        self.ema_sample_trained_t_list      = self.Sampler._save_multi_index_image_grid(ema_t_list)
        self.ema_sample_trained_mask_list   = self.Sampler._save_multi_index_image_grid(ema_mask_list)


    def _save_train_result_each_t(self, dirs, img, epoch):
        #=====================================================
        # degrade the original train image with each t's noise
        # get the result about each degraded image in t
        # it's not sampling, it just use one t to get result
        #=====================================================
        dir_save    = dirs.list_dir['each_time_result'] 
        
        noisy_list, mask_list, sample_list = self.Sampler.result_each_t(img, self.model.eval())
        
        noisy_list  = torch.cat(noisy_list, dim=0)
        mask_list   = torch.cat(mask_list, dim=0)
        sample_list = torch.cat(sample_list, dim=0)
        
        # sample      = normalize01(sample)
        file_save   = 'each_result_t_{:05d}.png'.format(epoch)
        file_save   = os.path.join(dir_save, file_save)
        
        nrow        = int(np.ceil(np.sqrt(len(sample_list))))
        noisy_list  = normalize01(noisy_list)
        mask_list   = normalize01(mask_list)
        sample_list = normalize01(sample_list)
        
        noisy_grid  = make_grid(noisy_list, nrow=nrow, normalize=True)
        mask_grid   = make_grid(mask_list, nrow=nrow, normalize=True)
        sample_grid = make_grid(sample_list, nrow=nrow, normalize=True)
        
        noisy_grid  = noisy_grid.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        mask_grid   = mask_grid.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        sample_grid = sample_grid.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    
    
        fig, axarr = plt.subplots(1,3,figsize=(15, 10)) 
        axarr[0].imshow(X=noisy_grid)
        axarr[1].imshow(X=mask_grid)
        axarr[2].imshow(X=sample_grid)
        
        axarr[0].set_title("noisy")
        axarr[1].set_title("output")
        axarr[2].set_title("prediction")
        
        axarr[0].axis("off")
        axarr[1].axis("off")
        axarr[2].axis("off")
        
        plt.tight_layout()
        fig.savefig(file_save)
        plt.close(fig)
        
        self.each_train_first_result_t   = util.make_multi_grid([noisy_list, mask_list, sample_list], nrow=1, ncol=3)
        
    def _save_sample_all_t(self, dirs, epoch):
        dir_save    = dirs.list_dir['sample_all_t'] 

        self.sample_all_t, sample_all_t_list = self.Sampler.sample_all_t(self.model.eval())
        
        file_save               = 'sample_all_t{:05d}.png'.format(epoch)
        self.sample_all_t       = self.Sampler._save_image_grid(self.sample_all_t, dir_save, file_save)
        
        # sample_all_t_list       = torch.cat(sample_all_t_list, dim=0)
        # self.sample_all_t_list  = self.Sampler._save_image_grid(sample_all_t_list, None, None)
        # self.sample_all_t_list  = util.make_multi_grid(sample_all_t_list, nrow=3, ncol=3)
        self.sample_all_t_list  = self.Sampler._save_multi_index_image_grid(sample_all_t_list)
        
        
    def _save_ema_sample_all_t(self, dirs, epoch):
        
        self.ema_model.store(self.model.parameters())
        # model_ema.parameters => model.parameters
        self.ema_model.copy_to(self.model.parameters())
        
        self.ema_sample_all_t, ema_sample_all_t_list = self.Sampler.sample_all_t(self.model.eval())
        
        self.ema_sample_all_t       = self.Sampler._save_image_grid(self.ema_sample_all_t, None, None)
        
        # ema_sample_all_t_list       = torch.cat(ema_sample_all_t_list, dim=0)
        # self.ema_sample_all_t_list  = self.Sampler._save_image_grid(ema_sample_all_t_list, None, None)
        # self.ema_sample_all_t_list  = util.make_multi_grid(ema_sample_all_t_list, nrow=3, ncol=3)
        self.ema_sample_all_t_list  = self.Sampler._save_multi_index_image_grid(ema_sample_all_t_list)
        
        
        self.ema_model.restore(self.model.parameters())
        
        
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
    