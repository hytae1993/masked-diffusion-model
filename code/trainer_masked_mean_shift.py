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
        
        self.visual_names       = ['input','degraded_img', 'degradation', 'shifted_degrade_img', 'shifted_input', 'mask', 'reconstructed_img', 'each_train_first_result_t', 'sample', 'sample_t']
        self.loss_names         = ['reconstruct_loss', 'learning_rate']
        
        
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
        # Create a mask with a random area black and obtation degraded image
        # ===================================================================================
        timesteps           = torch.randint(low=1, high=self.args.updated_ddpm_num_steps+1, size=(self.input.shape[0],), device=self.input.device)
        timesteps_count     = torch.bincount(timesteps, minlength=self.args.updated_ddpm_num_steps+1)[1:]
        T_steps             = torch.where(timesteps == self.args.updated_ddpm_num_steps)
        inference_t_steps   = torch.where(timesteps > int(self.args.updated_ddpm_num_steps/2))
        
        black_area_num      = self.Scheduler.get_black_area_num_pixels_time(timesteps)      # get number of removed pixels at each timestep 
        
        self.degraded_img, self.degradation    = self.Scheduler.get_mean_mask(black_area_num, self.input)
        
        # ===================================================================================
        # shift 
        # ===================================================================================
        shift                       = self.Scheduler.get_schedule_shift_time(timesteps) 
        self.shifted_degrade_img    = self.Scheduler.perturb_shift(self.degraded_img, shift)
        self.shifted_input          = self.Scheduler.perturb_shift(self.input, shift)
        
        with self.accelerator.accumulate(self.model):
            self.mask               = self.model(self.shifted_degrade_img, timesteps).sample
            self.reconstructed_img      = self.shifted_degrade_img + self.mask
            
            # loss            = self._compute_loss(prediction.to(torch.float32), target.to(torch.float32))
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
                                
        img_set             = [self.input, self.degradation, self.degraded_img, self.shifted_degrade_img, self.shifted_input, self.mask, self.reconstructed_img]
        
        if self.accelerator.is_main_process:
            losses = self.get_current_losses()
            visualizer.plot_current_losses(epoch, losses)
        
        return img_set, self.reconstruct_loss.item(), timesteps_count, black_image_index, inference_check_set


    def _run_epoch(self, epoch: int, epoch_length: int, resume_step: int, dirs: dict, visualizer):
        loss_batch              = []
        epoch_timesteps_count   = torch.zeros(self.args.updated_ddpm_num_steps, dtype=torch.int)
        
        batch_progress_bar   = tqdm(total=len(self.dataloader), disable=not self.accelerator.is_local_main_process, leave=False)
        batch_progress_bar.set_description(f"Batch ")
        
        black_image_set     = [[],[],[],[],[]]
        inference_image_set = [[],[],[]]
        
        # label   = [1 for _ in range(10)]
        
        for i, input in enumerate(self.dataloader, 0):
            # print(input['label'])
            # for j in range(len(input['label'])):
            #     label[input['label'][j]] += 1
                
            img_set, loss, batch_timesteps_count, black_index, inference_set = self._run_batch(i, input, epoch, epoch_length, resume_step, dirs, visualizer)
            batch_progress_bar.update(1)
            
            if self.accelerator.is_main_process: 
                
                loss_batch.append(loss)
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

        # print(label)
        # exit(1)
        batch_progress_bar.close()
        return img_set, loss_batch, epoch_timesteps_count, black_image_set, inference_image_set
    
    
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
            
            if self.accelerator.is_main_process:
                visualizer.reset()
            self.dataloader.batch_sampler.batch_sampler.sampler.set_epoch(epoch)
            img_set, loss, timesteps_count, black_image_set, inference_image_set = self._run_epoch(epoch, epoch_length, resume_step, dirs, visualizer)
            
            end = timer()
            elapsed_time = end - start
            
            if self.accelerator.is_main_process:
                loss_mean     = statistics.mean(loss)
                loss_std      = statistics.stdev(loss, loss_mean)
                self.reconstruct_loss   = loss_mean

                loss_mean_epoch.append(loss_mean)
                loss_std_epoch.append(loss_std)
                
                if epoch == epoch_start or epoch % self.args.save_images_epochs == 0 or epoch == (epoch_start+epoch_length-1):
    
                    self._save_model(dirs, epoch)
                    self._save_result_image(dirs, img_set, epoch)
                    self._save_inference_image(dirs, inference_image_set, epoch)
                    self._save_black_image(dirs, black_image_set, epoch)
                    self._save_train_result_each_t(dirs, img_set[0], epoch)
                    self._save_sample(dirs, epoch)
                    # self._save_sample_random_t(dirs, img_set[0], epoch)
                    self._save_learning_curve(dirs, loss_mean_epoch, loss_std_epoch)
                    self._save_time_step(dirs, timesteps_count, epoch)
                    # save to wandb
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
    
    def _save_inference_image(self, dirs, inference_set, epoch):
        input               = torch.cat(inference_set[0], dim=0)
        prediction          = torch.cat(inference_set[1], dim=0)
        timesteps           = torch.tensor(inference_set[2], device=input.device) - 1
        
        black_area_pixels   = self.Scheduler.get_black_area_num_pixels_time(timesteps)
        noisy_img, noise    = self.Scheduler.get_mean_mask(black_area_pixels, input)
        
        shift               = self.Scheduler.get_schedule_shift_time(timesteps) 
        source              = self.Scheduler.perturb_shift(noisy_img, shift).to(self.args.weight_dtype)
        
        mask                = self.model(source, timesteps).sample
        new_prediction      = source + mask
        
        # input predict new_predict
        # noise   noisy     output   
        inf_dir_save        = dirs.list_dir['inference_grid']
        batch_size          = input.shape[0]
        nrow                = int(np.ceil(np.sqrt(batch_size)))
        
        input               = normalize01(input)
        prediction          = normalize01(prediction)
        new_prediction      = normalize01(new_prediction)
        noise               = normalize01(noise)
        noisy_img           = normalize01(noisy_img)
        mask                = normalize01(mask)
        
        grid_input          = make_grid(input, nrow=nrow, normalize=True)
        grid_predict        = make_grid(prediction, nrow=nrow, normalize=True)
        grid_new_predict    = make_grid(new_prediction, nrow=nrow, normalize=True)
        grid_noise          = make_grid(noise, nrow=nrow, normalize=True)
        grid_noisy          = make_grid(noisy_img, nrow=nrow, normalize=True)
        grid_mask           = make_grid(mask, nrow=nrow, normalize=True)
        
        inf_final           = 'inference_epoch_{:05d}.png'.format(epoch)
        inf_final           = os.path.join(inf_dir_save, inf_final)
        
        grid_input          = grid_input.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_predict        = grid_predict.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_new_predict    = grid_new_predict.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_noise          = grid_noise.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_noisy          = grid_noisy.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_mask           = grid_mask.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        
        fig, axarr = plt.subplots(2,3,figsize=(15, 10)) 
        axarr[0][0].imshow(X=grid_input)
        axarr[0][1].imshow(X=grid_predict)
        axarr[0][2].imshow(X=grid_new_predict)
        axarr[1][0].imshow(X=grid_noise)
        axarr[1][1].imshow(X=grid_noisy)
        axarr[1][2].imshow(X=grid_mask)
        
        axarr[0][0].set_title("input")
        axarr[0][1].set_title("predict")
        axarr[0][2].set_title("new predict")
        axarr[1][0].set_title("noise")
        axarr[1][1].set_title("noisy")
        axarr[1][2].set_title("output")
        
        axarr[0][0].axis("off")
        axarr[0][1].axis("off")
        axarr[0][2].axis("off")
        axarr[1][0].axis("off")
        axarr[1][1].axis("off")
        axarr[1][2].axis("off")
        plt.tight_layout()
        fig.savefig(inf_final)
        plt.close(fig)
        
        
    def _save_result_image(self, dirs, img, epoch):
        input       = img[0]    # input image
        noise       = img[1]    # randomly generated mask
        noisy       = img[2]    # noise * input
        source      = img[3]    # shift of noisy
        target      = img[4]    # shift of input
        mask        = img[5]    # output of model
        predict     = img[6]    # output of model + source image
        batch_size  = input.shape[0]
        nrow        = int(np.ceil(np.sqrt(batch_size)))
        
        input_dir_save      = dirs.list_dir['train_img'] 
        file_input          = 'input_epoch_{:05d}.png'.format(epoch)
        file_input          = os.path.join(input_dir_save, file_input)
        input               = normalize01(input)
        grid_input          = make_grid(input, nrow=nrow, normalize=True)
        save_image(grid_input, file_input)
        
        noise_dir_save      = dirs.list_dir['noise_img']
        file_noise          = 'noise_epoch_{:05d}.png'.format(epoch)
        file_noise          = os.path.join(noise_dir_save, file_noise)
        noise               = normalize01(noise)
        grid_noise          = make_grid(noise, nrow=nrow, normalize=True)
        save_image(grid_noise, file_noise)
        
        noisy_dir_save      = dirs.list_dir['noisy_img']
        file_noisy          = 'noisy_epoch_{:05d}.png'.format(epoch)
        file_noisy          = os.path.join(noisy_dir_save, file_noisy)
        noisy               = normalize01(noisy)
        grid_noisy          = make_grid(noisy, nrow=nrow, normalize=True)
        save_image(grid_noisy, file_noisy)
        
        mask_dir_save       = dirs.list_dir['mask_img']
        file_mask           = 'mask_epoch_{:05d}.png'.format(epoch)
        file_mask           = os.path.join(mask_dir_save, file_mask)
        mask                = normalize01(mask)
        grid_mask           = make_grid(mask, nrow=nrow, normalize=True)
        save_image(grid_mask, file_mask)
        
        predict_dir_save    = dirs.list_dir['predict_img']
        file_final          = 'predict_epoch_{:05d}.png'.format(epoch)
        file_final          = os.path.join(predict_dir_save, file_final)
        predict             = normalize01(predict)
        grid_final          = make_grid(predict, nrow=nrow, normalize=True)
        save_image(grid_final, file_final)
        
        shift_img_dir_save  = dirs.list_dir['shift_img']
        file_shift_img      = 'shifted_img_epoch_{:05d}.png'.format(epoch)
        file_shift_img      = os.path.join(shift_img_dir_save, file_shift_img)
        target              = normalize01(target)
        grid_shift_input    = make_grid(target, nrow=nrow, normalize=True)
        save_image(grid_shift_input, file_shift_img)
        
        shift_noisy_dir_save    = dirs.list_dir['shift_noisy']
        file_shift_noisy        = 'shifted_noisy_epoch_{:05d}.png'.format(epoch)
        file_shift_noisy        = os.path.join(shift_noisy_dir_save, file_shift_noisy)
        source                  = normalize01(source)
        grid_shift_noisy        = make_grid(source, nrow=nrow, normalize=True)
        save_image(grid_shift_noisy, file_shift_noisy)
        
        img_dir_save        = dirs.list_dir['img']
        img_final           = 'img_epoch_{:05d}.png'.format(epoch)
        img_final           = os.path.join(img_dir_save, img_final)
        grid_input          = grid_input.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_noisy          = grid_noisy.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_noise          = grid_noise.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_mask           = grid_mask.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_final          = grid_final.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_shift_input    = grid_shift_input.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_shift_noisy    = grid_shift_noisy.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        
        grid                = [grid_input, grid_noise, grid_noisy, grid_shift_input, grid_shift_noisy, grid_mask, grid_final]
        grid_name           = ['input', 'noise', 'noisy', 'input shift', 'noisy shift', 'mask', 'final']
        fig = plt.figure(figsize=(15, 10))
        # positions = [(0, 1, 2), (0, 3, 2), (1, 0, 2), (1, 2, 2), (1, 4, 2)]
        positions = [(0, 1, 2), (0, 3, 2), (0, 5, 2), (1, 0, 2), (1, 2, 2), (1, 4, 2), (1, 6, 2)]
        for i, (row, col, colspan) in enumerate(positions):
            ax = plt.subplot2grid((2, 8), (row, col), colspan=colspan)
            ax.imshow(X=grid[i])
            ax.set_title(grid_name[i])
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(img_final)
        plt.close(fig)
        
    def _save_black_image(self, dirs, img, epoch):
        
        if len(img[0]):
        
            input       = torch.cat(img[0], dim=0)    # input image
            noisy       = torch.cat(img[1], dim=0)    # noise * input
            mask        = torch.cat(img[2], dim=0)    # output of model
            predict     = torch.cat(img[3], dim=0)    # output of model + noisy image
            noise       = torch.cat(img[4], dim=0)    # noise
            batch_size  = input.shape[0]
            nrow        = int(np.ceil(np.sqrt(batch_size)))
            
            input       = normalize01(input)
            noisy       = normalize01(noisy)
            mask        = normalize01(mask)
            predict     = normalize01(predict)
            noise       = normalize01(noise)
            
            grid_input          = make_grid(input, nrow=nrow, normalize=True)
            grid_noisy          = make_grid(noisy, nrow=nrow, normalize=True)
            grid_mask           = make_grid(mask, nrow=nrow, normalize=True)
            grid_final          = make_grid(predict, nrow=nrow, normalize=True)
            grid_noise          = make_grid(noise, nrow=nrow, normalize=True)
            
            img_dir_save        = dirs.list_dir['black_res_img']
            img_final           = 'black_epoch_{:05d}.png'.format(epoch)
            img_final           = os.path.join(img_dir_save, img_final)
            grid_input          = grid_input.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            grid_noisy          = grid_noisy.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            grid_mask           = grid_mask.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            grid_final          = grid_final.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            grid_noise          = grid_noise.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            
            grid                = [grid_input, grid_noise, grid_noisy, grid_mask, grid_final]
            grid_name           = ['input', 'noise', 'noisy', 'mask', 'final']
            fig = plt.figure(figsize=(15, 10))
            positions = [(0, 0, 2), (0, 2, 2), (0, 4, 2), (1, 1, 2), (1, 3, 2)]
            for i, (row, col, colspan) in enumerate(positions):
                ax = plt.subplot2grid((2, 6), (row, col), colspan=colspan)
                ax.imshow(X=grid[i])
                ax.set_title(grid_name[i])
                ax.axis("off")
                
            plt.tight_layout()
            fig.savefig(img_final)
            plt.close(fig)
                    

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
        
        fig = plt.figure()
        
        plt.subplot(1,2,1)
        plt.plot(time, self.time_step, color='red')
        plt.title('number of time step')
        
        plt.subplot(1,2,2)
        plt.plot(time, black_area_pixels, color='red')
        plt.title('number of pixels in each time step')
        
        plt.tight_layout()
        plt.savefig(file_loss, bbox_inches='tight', dpi=100)
        plt.close(fig)
 
 
    def _save_sample(self, dirs, epoch):
        dir_save            = dirs.list_dir['sample_img'] 
        dir_grid_save       = dirs.list_dir['sample_grid']

        sample, sample_list, sample_t = self.Sampler.sample(self.model.eval())
        # sample      = normalize01(sample)
        file_save       = 'sample_{:05d}.png'.format(epoch)
        self.sample     = self.Sampler._save_image_grid(sample, dir_save, file_save)
        
        self.Sampler._save_image_multi_grid(sample_list, sample_t, dir_grid_save, file_save)
        self.sample_t   = util.make_multi_grid(sample_list, nrow=2, ncol=3) # result of t during sampling
 

    def _save_train_result_each_t(self, dirs, img, epoch):
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
        
    def _save_sample_random_t(self, dirs, img, epoch):
        dir_save    = dirs.list_dir['sample_random'] 

        sample_list = self.Sampler.sample_random_t(img, self.model.eval())
        sample_list = torch.cat(sample_list, dim=0)
        # sample      = normalize01(sample)
        file_save   = 'sample_random_t_{:05d}.png'.format(epoch)
        file_save   = os.path.join(dir_save, file_save)
        
        nrow        = int(np.ceil(np.sqrt(len(sample_list))))
        sample_list = normalize01(sample_list)
        grid        = make_grid(sample_list, nrow=nrow, normalize=True)
        save_image(grid, file_save)
        
        
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
    