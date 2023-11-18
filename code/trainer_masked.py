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
from utils.mask import Mask

# ===============================================================================
# Generete image with diffusion model - input: image & time - Base code of DDPM
# ===============================================================================

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
        self.accelerator        = accelerator
        
        self.criterion          = nn.MSELoss()
        
        self.Mask               = Mask(args)
        
        
    def _compute_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        loss    = self.criterion(prediction, target)
        
        return loss
         
    def _run_batch(self, batch: int, img: torch.Tensor, epoch: int, epoch_length: int, resume_step: int, global_step: int, dirs: dict):
        
        img                                 = img.to(self.args.weight_dtype)
        
        # ===================================================================================
        # Create a mask with a random area black and obtation generator prediction
        timesteps           = torch.randint(low=1, high=self.args.updated_ddpm_num_steps+1, size=(img.shape[0],), device=img.device)
        timesteps_count     = torch.bincount(timesteps, minlength=self.args.updated_ddpm_num_steps+1)[1:]
        T_steps             = torch.where(timesteps == self.args.updated_ddpm_num_steps)
        inference_t_steps   = torch.where(timesteps > int(self.args.updated_ddpm_num_steps/2))
        
        black_area_ratio    = self.Mask.get_list_black_area_ratios(timesteps)
        noise               = self.Mask.get_mask(black_area_ratio)
        noise               = noise.to(img.device)
        
        noisy_img           = img * noise
        
        with self.accelerator.accumulate(self.model):
            # mask            = self.model(noisy_img, black_area_ratio).sample
            mask            = self.model(noisy_img, timesteps).sample
            prediction      = noisy_img + mask
            
            loss            = self._compute_loss(prediction.to(torch.float32), img.to(torch.float32))
            
            self.accelerator.backward(loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        if self.accelerator.sync_gradients:
            if self.args.use_ema:
                self.ema_model.step(self.model.parameters())
            global_step += 1
            
            if self.accelerator.is_main_process:
                if global_step % self.args.checkpointing_steps == 0:
                    
                    save_path   = os.path.join(dirs.list_dir['checkpoint'], f"checkpoint-{global_step}")
                    self.accelerator.save_state(save_path)
        
        
        lr  = self.lr_scheduler.get_last_lr()[0]
        self.accelerator.wait_for_everyone()
        
        black_image_index       = None
        if len(T_steps[0]) > 0:
            black_image_index   = T_steps[0]
            
        inference_image_index, inference_check_set  = None, None
        if len(inference_t_steps[0]) > 0:
            inference_image_index   = inference_t_steps[0][0]
            inference_check_set = [inference_image_index, timesteps[inference_image_index]]
                                
        img_set             = [img, noise, noisy_img, mask, prediction]
        
        return img_set, loss.item(), timesteps_count, black_image_index, inference_check_set


    def _run_epoch(self, epoch: int, epoch_length: int, resume_step: int, global_step: int, dirs: dict):
        loss_batch              = []
        epoch_timesteps_count   = torch.zeros(self.args.updated_ddpm_num_steps, dtype=torch.int)
        
        # for i, (input, saliency) in enumerate(tqdm(self.dataloader, desc='batch', leave=False, position=1, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}')):
        batch_progress_bar   = tqdm(total=len(self.dataloader), disable=not self.accelerator.is_local_main_process, leave=False)
        batch_progress_bar.set_description(f"Batch ")
        
        black_image_set     = [[],[],[],[],[]]
        inference_image_set = [[],[],[]]
        for i, input in enumerate(self.dataloader, 0):
            
            img_set, loss, batch_timesteps_count, black_index, inference_set = self._run_batch(i, input, epoch, epoch_length, resume_step, global_step, dirs)
            batch_progress_bar.update(1)
            
            if self.accelerator.is_main_process:
            #     self._save_result_image(dirs, img_set, epoch)
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


        batch_progress_bar.close()
        return img_set, loss_batch, epoch_timesteps_count, black_image_set, inference_image_set
    
    
    def train(self, epoch_start: int, epoch_length: int, resume_step: int, global_step: int, dirs: dict):
        
        updated_ddpm_num_steps, rate        = self.Mask.update_ddpm_num_steps(self.args.ddpm_num_steps)
        self.args.updated_ddpm_num_steps    = updated_ddpm_num_steps
        
        epoch_length    = epoch_length
        epoch_start     = epoch_start
        loss_mean_epoch = []
        loss_std_epoch  = []
        lr_list_all     = [] 
       
        self.model.train()
       
        # for epoch in tqdm(range(epoch_start,epoch_length), desc='epoch', position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}'):
        epoch_progress_bar   = tqdm(total=epoch_length, disable=not self.accelerator.is_local_main_process)
        epoch_progress_bar.set_description(f"Epoch ")
        for epoch in range(epoch_start,epoch_length):
            start = timer()
            img_set, loss, timesteps_count, black_image_set, inference_image_set = self._run_epoch(epoch, epoch_length, resume_step, global_step, dirs)
            
            end = timer()
            elapsed_time = end - start
            epoch_progress_bar.update(1)
 
            loss_mean     = statistics.mean(loss)
            loss_std      = statistics.stdev(loss, loss_mean)

            if epoch > 0:
                loss_mean_epoch.append(loss_mean)
                loss_std_epoch.append(loss_std)
            
            
            if self.accelerator.is_main_process:
    
                # self._save_model(dirs, epoch+1)
                self._save_result_image(dirs, img_set, epoch)
                self._save_inference_image(dirs, inference_image_set, epoch)
                self._save_black_image(dirs, black_image_set, epoch)
                self._save_sample(dirs, epoch)
                self._save_learning_curve(dirs, loss_mean_epoch, loss_std_epoch, epoch)
                self._save_time_step(dirs, timesteps_count, rate, epoch)
                # self._save_log(dirs)
                # self.log = ''
        # self._save_loss(dirs, loss_generator_mean_epoch, loss_generator_std_epoch, loss_discriminator_mean_epoch, loss_discriminator_std_epoch)
        epoch_progress_bar.close()
        
    
    def _save_inference_image(self, dirs, inference_set, epoch):
        input               = torch.cat(inference_set[0], dim=0)
        prediction          = torch.cat(inference_set[1], dim=0)
        timesteps           = torch.tensor(inference_set[2], device=input.device) - 1
        
        black_area_ratio    = self.Mask.get_list_black_area_ratios(timesteps)
        noise               = self.Mask.get_mask(black_area_ratio)
        noise               = noise.to(input.device)
        
        noisy_img           = prediction * noise
        mask                = self.model(noisy_img, timesteps).sample
        new_prediction      = noisy_img + mask
        
        # input predict new_predict
        # noise   noisy     output   
        inf_dir_save        = dirs.list_dir['inference_grid']
        batch_size          = input.shape[0]
        nrow                = int(np.ceil(np.sqrt(batch_size)))
        
        grid_input          = make_grid(input, nrow=nrow, normalize=True)
        grid_predict        = make_grid(prediction, nrow=nrow, normalize=True)
        grid_new_predict    = make_grid(new_prediction, nrow=nrow, normalize=True)
        grid_noise          = make_grid(noise, nrow=nrow, normalize=True)
        grid_noisy          = make_grid(noisy_img, nrow=nrow, normalize=True)
        grid_mask           = make_grid(mask, nrow=nrow, normalize=True)
        
        inf_final           = 'inference_epoch_{:05d}.png'.format(epoch)
        inf_final           = os.path.join(inf_dir_save, inf_final)
        
        grid_input          = (grid_input.cpu().numpy() * 255).round().astype("uint8")
        grid_predict        = (grid_predict.cpu().numpy() * 255).round().astype("uint8")
        grid_new_predict    = (grid_new_predict.cpu().numpy() * 255).round().astype("uint8")
        grid_noise          = (grid_noise.cpu().numpy() * 255).round().astype("uint8")
        grid_noisy          = (grid_noisy.cpu().numpy() * 255).round().astype("uint8")
        grid_mask           = (grid_mask.cpu().numpy() * 255).round().astype("uint8")
        
        fig, axarr = plt.subplots(2,3) 
        axarr[0][0].imshow(grid_input.transpose((1,2,0)))
        axarr[0][1].imshow(grid_predict.transpose((1,2,0)))
        axarr[0][2].imshow(grid_new_predict.transpose((1,2,0)))
        axarr[1][0].imshow(grid_noise.transpose((1,2,0)))
        axarr[1][1].imshow(grid_noisy.transpose((1,2,0)))
        axarr[1][2].imshow(grid_mask.transpose((1,2,0)))
        
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
        mask        = img[3]    # output of model
        predict     = img[4]    # output of model + noisy image
        batch_size  = input.shape[0]
        nrow        = int(np.ceil(np.sqrt(batch_size)))
        
        input_dir_save      = dirs.list_dir['train_img'] 
        file_input          = 'input_epoch_{:05d}.png'.format(epoch)
        file_input          = os.path.join(input_dir_save, file_input)
        grid_input          = make_grid(input, nrow=nrow, normalize=True)
        save_image(grid_input, file_input)
        
        noise_dir_save      = dirs.list_dir['noise_img']
        file_noise          = 'noise_epoch_{:05d}.png'.format(epoch)
        file_noise          = os.path.join(noise_dir_save, file_noise)
        grid_noise          = make_grid(noise, nrow=nrow, normalize=True)
        save_image(grid_noise, file_noise)
        
        noisy_dir_save      = dirs.list_dir['noisy_img']
        file_noisy          = 'noise_epoch_{:05d}.png'.format(epoch)
        file_noisy          = os.path.join(noisy_dir_save, file_noisy)
        grid_noisy          = make_grid(noisy, nrow=nrow, normalize=True)
        save_image(grid_noisy, file_noisy)
        
        mask_dir_save       = dirs.list_dir['mask_img']
        file_mask           = 'mask_epoch_{:05d}.png'.format(epoch)
        file_mask           = os.path.join(mask_dir_save, file_mask)
        grid_mask           = make_grid(mask, nrow=nrow, normalize=True)
        save_image(grid_mask, file_mask)
        
        predict_dir_save    = dirs.list_dir['predict_img']
        file_final          = 'predict_epoch_{:05d}.png'.format(epoch)
        file_final          = os.path.join(predict_dir_save, file_final)
        grid_final          = make_grid(predict, nrow=nrow, normalize=True)
        save_image(grid_final, file_final)
        
        img_dir_save        = dirs.list_dir['img']
        img_final           = 'img_epoch_{:05d}.png'.format(epoch)
        img_final           = os.path.join(img_dir_save, img_final)
        grid_input          = (grid_input.cpu().numpy() * 255).round().astype("uint8")
        grid_noisy          = (grid_noisy.cpu().numpy() * 255).round().astype("uint8")
        grid_noise          = (grid_noise.cpu().numpy() * 255).round().astype("uint8")
        grid_mask           = (grid_mask.cpu().numpy() * 255).round().astype("uint8")
        grid_final          = (grid_final.cpu().numpy() * 255).round().astype("uint8")
        # grid_sample         = (grid_sample.cpu().numpy() * 255).round().astype("uint8")
        
        fig, axarr = plt.subplots(1,5) 
        axarr[0].imshow(grid_input.transpose((1,2,0)))
        axarr[1].imshow(grid_noise.transpose((1,2,0)))
        axarr[2].imshow(grid_noisy.transpose((1,2,0)))
        axarr[3].imshow(grid_mask.transpose((1,2,0)))
        axarr[4].imshow(grid_final.transpose((1,2,0)))
        
        axarr[0].set_title("input")
        axarr[1].set_title("noise")
        axarr[2].set_title("noisy")
        axarr[3].set_title("output")
        axarr[4].set_title("output + noisy")
        
        axarr[0].axis("off")
        axarr[1].axis("off")
        axarr[2].axis("off")
        axarr[3].axis("off")
        axarr[4].axis("off")
        plt.tight_layout()
        fig.savefig(img_final)
        plt.close(fig)
        
    def _save_black_image(self, dirs, img, epoch):
        input       = torch.cat(img[0], dim=0)    # input image
        noisy       = torch.cat(img[1], dim=0)    # noise * input
        mask        = torch.cat(img[2], dim=0)    # output of model
        predict     = torch.cat(img[3], dim=0)    # output of model + noisy image
        noise       = torch.cat(img[4], dim=0)    # noise
        batch_size  = input.shape[0]
        nrow        = int(np.ceil(np.sqrt(batch_size)))
        
        grid_input          = make_grid(input, nrow=nrow, normalize=True)
        grid_noisy          = make_grid(noisy, nrow=nrow, normalize=True)
        grid_mask           = make_grid(mask, nrow=nrow, normalize=True)
        grid_final          = make_grid(predict, nrow=nrow, normalize=True)
        grid_noise          = make_grid(noise, nrow=nrow, normalize=True)
        
        img_dir_save        = dirs.list_dir['black_res_img']
        img_final           = 'black_epoch_{:05d}.png'.format(epoch)
        img_final           = os.path.join(img_dir_save, img_final)
        grid_input          = (grid_input.cpu().numpy() * 255).round().astype("uint8")
        grid_noisy          = (grid_noisy.cpu().numpy() * 255).round().astype("uint8")
        grid_mask           = (grid_mask.cpu().numpy() * 255).round().astype("uint8")
        grid_final          = (grid_final.cpu().numpy() * 255).round().astype("uint8")
        grid_noise          = (grid_noise.cpu().numpy() * 255).round().astype("uint8")
        # grid_sample         = (grid_sample.cpu().numpy() * 255).round().astype("uint8")
        
        fig, axarr = plt.subplots(1,5) 
        axarr[0].imshow(grid_input.transpose((1,2,0)))
        axarr[1].imshow(grid_noise.transpose((1,2,0)))
        axarr[2].imshow(grid_noisy.transpose((1,2,0)))
        axarr[3].imshow(grid_mask.transpose((1,2,0)))
        axarr[4].imshow(grid_final.transpose((1,2,0)))
        
        axarr[0].set_title("input")
        axarr[1].set_title("noise")
        axarr[2].set_title("noisy")
        axarr[3].set_title("output")
        axarr[4].set_title("output + noisy")
        
        axarr[0].axis("off")
        axarr[1].axis("off")
        axarr[2].axis("off")
        axarr[3].axis("off")
        axarr[4].axis("off")
        plt.tight_layout()
        fig.savefig(img_final)
        plt.close(fig)
                    

    def _save_learning_curve(self, dirs, loss_mean, loss_std, epoch: int):
        dir_save    = dirs.list_dir['train_loss'] 
        # file_loss = 'loss_epoch_{:05d}.png'.format(epoch)
        file_loss = 'loss.png'.format(epoch)
        file_loss = os.path.join(dir_save, file_loss)
        fig = plt.figure()
        
        plt.subplot(1,1,1)
        plt.plot(np.array(loss_mean), color='red')
        plt.fill_between(list(range(len(loss_mean))), np.array(loss_mean)-np.array(loss_std), np.array(loss_mean)+np.array(loss_std), color='blue', alpha=0.2)
        plt.title('loss')
        
        plt.tight_layout()
        plt.savefig(file_loss, bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        
    def _save_time_step(self, dirs, time_step, rate, epoch: int):
        dir_save    = dirs.list_dir['time_step'] 
        # file_loss = 'loss_epoch_{:05d}.png'.format(epoch)
        file_loss = 'time_step_{}.png'.format(epoch)
        file_loss = os.path.join(dir_save, file_loss)
        
        time_step   = np.array(time_step)
        time        = range(1, len(time_step) + 1)
        
        fig = plt.figure()
        
        plt.subplot(1,2,1)
        plt.plot(time, time_step, color='red')
        plt.title('number of time step')
        
        plt.subplot(1,2,2)
        plt.plot(time, rate, color='red')
        plt.title('rate of each time step')
        
        plt.tight_layout()
        plt.savefig(file_loss, bbox_inches='tight', dpi=100)
        plt.close(fig)
 
 
    def _save_sample(self, dirs, epoch):
        dir_save            = dirs.list_dir['sample_img'] 
        dir_grid_save       = dirs.list_dir['sample_grid']
        sampler             = Sampler(self.dataloader, self.args, self.Mask)

        sample, sample_list = sampler.sample(self.model.eval())
        # sample      = normalize01(sample)
        file_save       = 'sample_{:05d}.png'.format(epoch)
        sampler._save_image_grid(sample, dir_save, file_save)
        sampler._save_image_multi_grid(sample_list, dir_grid_save, file_save)

        '''
        (sample, sample_time) = sampler.sample(model_info, self.model.eval())
        file_save       = 'sample_{:05d}.png'.format(epoch)
        file_save_time  = 'sample_{:05d}_time.png'.format(epoch)
        sample          = normalize01(sample)
        sampler._save_image_grid(sample, dir_save, file_save)
        # sampler._save_image_grid(sample_time, dir_save, file_save_time)
        ''' 
        
        '''
        num_trial = 10 
        sample_interp = sampler.sample_interpolate(model_info, self.model.eval(), num_trial)
        for trial in range(num_trial):
            sample_interp_trial = sample_interp[trial]
            file_save_interp    = 'sample_epoch_{:05d}_{:03d}_interp_{:02d}.png'.format(args.epoch, i+1, trial+1) 
            sample_interp_trial = normalize01(sample_interp_trial)
            sampler._save_image_grid(sample_interp_trial, dir_save, file_save_interp)
        ''' 
    
        # if (self.ema_model is not None) and (self.ema_model.ema_start <= epoch):
        #     '''
        #     (sample, sample_time) = sampler.sample(model_info, self.ema.eval())
        #     file_save       = 'sample_{:05d}_ema.png'.format(epoch)
        #     file_save_time  = 'sample_{:05d}_ema_time.png'.format(epoch)
        #     sample = normalize01(sample)
        #     sampler._save_image_grid(sample, dir_save, file_save)
        #     sampler._save_image_grid(sample_time, dir_save, file_save_time)
        #     '''
        #     sample = sampler.sample(self.ema_model.eval())
        #     # sample = normalize01(sample)
        #     file_save = 'sample_{:05d}_ema.png'.format(epoch)
        #     sampler._save_image_grid(sample, dir_save, file_save)
            
        #     '''
        #     sample_interp = sampler.sample_interpolate(model_info, self.ema.eval(), num_trial)
        #     for trial in range(num_trial):
        #         sample_interp_trial = sample_interp[trial]
        #         file_save_interp    = 'sample_epoch_{:05d}_{:03d}_interp_{:02d}_ema.png'.format(args.epoch, i+1, trial+1) 
        #         sample_interp_trial = normalize01(sample_interp_trial)
        #         sampler._save_image_grid(sample_interp_trial, dir_save, file_save_interp)
        #    ''' 
 
    def _save_model(self, dirs: dict, epoch: int):
        filename    = 'model_epoch_{:05d}.pth'.format(epoch)
        dir_save    = dirs.list_dir['model'] 
        filename    = os.path.join(dir_save, filename)
        
        for param_group in self.optim_G.param_groups:
            lr_generator = param_group['lr']
        
        for param_group in self.optim_D.param_groups:
            lr_discriminator = param_group['lr']

        torch.save({
            'generator'         : self.G.state_dict(),
            'discriminator'     : self.D.state_dict(),
            'lr_generator'      : lr_generator,
            'lr_discriminator'  : lr_discriminator,
            'epoch'             : epoch,
        }, filename)

        
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
    