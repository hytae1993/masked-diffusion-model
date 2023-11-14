import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim.optimizer as Optimizer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms

from diffusers import DDPMPipeline

from matplotlib.ticker import MaxNLocator
import numpy as np
import csv
import statistics
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os
import sys
import math
from PIL import Image
import cv2

# from tqdm.notebook import trange
# from tqdm import tqdm
from tqdm.auto import tqdm

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
        noise_scheduler,
        accelerator, 
        ):
        self.args               = args
        self.dataloader         = dataloader
        self.model              = model
        self.ema_model          = ema_model
        self.optimizer          = optimizer
        self.lr_scheduler       = lr_scheduler
        self.noise_scheduler    = noise_scheduler
        self.accelerator        = accelerator


    def _tempt(self):
        pass
         
    def _run_batch(self, batch: int, img: torch.Tensor, epoch: int, epoch_length: int, resume_step: int, global_step: int, dirs: dict):
        
        img             = img.to(self.args.weight_dtype)
        noise           = torch.randn(img.shape, dtype=self.args.weight_dtype, device=img.device)
        bsz             = img.shape[0]
        
        timesteps       = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(bsz,), device=img.device
        ).long()
        
        # ================================================================
        # obtaion generator prediction
        # ================================================================
        noisy_img       = self.noise_scheduler.add_noise(img, noise, timesteps)
                
        with self.accelerator.accumulate(self.model):
            model_output    = self.model(noisy_img, timesteps).sample
            
            loss            = F.mse_loss(model_output.float(), noise.float())
                
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
        
        # sampling
        sample = None
        if self.accelerator.is_main_process and (batch+1) % self.args.save_images_batch == 0:
            if epoch % self.args.save_images_epochs == 0 or epoch == self.args.epoch_length - 1:
                unet    = self.accelerator.unwrap_model(self.model)
                
                if self.args.use_ema:
                    self.ema_model.store(unet.parameters())
                    self.ema_model.copy_to(unet.parameters())
                    
                pipeline    = DDPMPipeline(
                    unet=unet,
                    scheduler=self.noise_scheduler
                )
                
                generator   = torch.Generator(device=pipeline.device).manual_seed(0)
                
                sample      = pipeline(
                    generator=generator,
                    batch_size=self.args.batch_size,
                    num_inference_steps=self.args.ddpm_num_inference_steps,
                    output_type='numpy'
                ).images
                
                sample  = torch.tensor(sample.transpose(0,3,1,2))
                sample  = make_grid(sample, nrow=4, normalize=True)
                
                if self.args.use_ema:
                    self.ema_model.restore(unet.parameters())
                    
        img_set = [img, noisy_img, noisy_img-img, model_output, sample]
        
        return img_set, loss.item()


    def _run_epoch(self, epoch: int, epoch_length: int, resume_step: int, global_step: int, dirs: dict):
        loss_batch        = []
        
        # for i, (input, saliency) in enumerate(tqdm(self.dataloader, desc='batch', leave=False, position=1, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}')):
        batch_progress_bar   = tqdm(total=len(self.dataloader), disable=not self.accelerator.is_local_main_process, leave=False)
        batch_progress_bar.set_description(f"Batch ")
        for i, input in enumerate(self.dataloader, 0):
            
            img_set, loss = self._run_batch(i, input, epoch, epoch_length, resume_step, global_step, dirs)
            batch_progress_bar.update(1)
            
            if self.accelerator.is_main_process and (i+1) % self.args.save_images_batch == 0:
                self._save_result_image(dirs, img_set, epoch)
            loss_batch.append(loss)

        batch_progress_bar.close()
        return img_set, loss_batch
    
    
    def train(self, epoch_start: int, epoch_length: int, resume_step: int, global_step: int, dirs: dict):
        
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
            img_set, loss = self._run_epoch(epoch, epoch_length, resume_step, global_step, dirs)
            
            end = timer()
            elapsed_time = end - start
            epoch_progress_bar.update(1)
 
            loss_mean     = statistics.mean(loss)
            loss_std      = statistics.stdev(loss, loss_mean)

            if epoch > 0:
                loss_mean_epoch.append(loss_mean)
                loss_std_epoch.append(loss_std)
            
            if (epoch+1) % self.args.save_loss == 0:
 
        #         self._save_model(dirs, epoch+1)
        #         # self._save_fake_image(dirs, img_set[1], epoch+1)
        #         self._save_result_image(dirs, img_set, epoch+1)
                self._save_learning_curve(dirs, loss_mean_epoch, loss_std_epoch, epoch+1)
        #         self._save_prediction_curve(dirs, pred_real_mean_epoch, pred_real_std_epoch, pred_fake_mean_epoch, pred_fake_std_epoch, epoch+1)
        #         self._save_log(dirs)
        #         self.log = ''
        # self._save_loss(dirs, loss_generator_mean_epoch, loss_generator_std_epoch, loss_discriminator_mean_epoch, loss_discriminator_std_epoch)
        epoch_progress_bar.close()
        
    def _save_result_image(self, dirs, img, epoch):
        input       = img[0]
        noisy       = img[1]
        noise       = img[2]
        predict     = img[3]
        sample      = img[4]
        batch_size  = input.shape[0]
        nrow        = int(np.ceil(np.sqrt(batch_size)))
        
        input_dir_save      = dirs.list_dir['train_img'] 
        file_input          = 'input_epoch_{:05d}.png'.format(epoch)
        file_input          = os.path.join(input_dir_save, file_input)
        grid_input          = make_grid(input, nrow=nrow, normalize=True)
        save_image(grid_input, file_input)
        
        masked_dir_save     = dirs.list_dir['masked_img']
        file_noisy          = 'noisy_epoch_{:05d}.png'.format(epoch)
        file_noisy          = os.path.join(masked_dir_save, file_noisy)
        grid_noisy          = make_grid(noisy, nrow=nrow, normalize=True)
        save_image(grid_noisy, file_noisy)
        
        noise_dir_save      = dirs.list_dir['noise_img']
        file_noise          = 'noise_epoch_{:05d}.png'.format(epoch)
        file_noise          = os.path.join(noise_dir_save, file_noise)
        grid_noise          = make_grid(noise, nrow=nrow, normalize=True)
        save_image(grid_noise, file_noise)
        
        predict_dir_save    = dirs.list_dir['predict_img']
        file_final          = 'predict_epoch_{:05d}.png'.format(epoch)
        file_final          = os.path.join(predict_dir_save, file_final)
        grid_final          = make_grid(predict, nrow=nrow, normalize=True)
        save_image(grid_final, file_final)
        
        # grid_sample         = (sample * 255).round().astype("uint8")
        # grid_sample         = grid_sample.transpose((0,3,1,2))
        sample_dir_save     = dirs.list_dir['sample_img']
        file_sample         = 'sample_epoch_{:05d}.png'.format(epoch)
        file_sample         = os.path.join(sample_dir_save, file_sample)
        save_image(sample, file_sample)
        # self._save_images_to_grid(grid_sample, (4, 4), file_sample)
        
        # grid_sample         = Image.fromarray(grid_sample) # NumPy array to PIL image
        # grid_sample         = make_grid(sample, nrow=nrow, normalize=True)
        # save_image(grid_sample, file_sample)
        
        img_dir_save        = dirs.list_dir['img']
        img_final           = 'img_epoch_{:05d}.png'.format(epoch)
        img_final           = os.path.join(img_dir_save, img_final)
        grid_input          = (grid_input.cpu().numpy() * 255).round().astype("uint8")
        grid_noisy          = (grid_noisy.cpu().numpy() * 255).round().astype("uint8")
        grid_noise          = (grid_noise.cpu().numpy() * 255).round().astype("uint8")
        grid_final          = (grid_final.cpu().numpy() * 255).round().astype("uint8")
        # grid_sample         = (grid_sample.cpu().numpy() * 255).round().astype("uint8")
        
        fig, axarr = plt.subplots(1,4) 
        axarr[0].imshow(grid_input.transpose((1,2,0)))
        axarr[1].imshow(grid_noisy.transpose((1,2,0)))
        axarr[2].imshow(grid_noise.transpose((1,2,0)))
        axarr[3].imshow(grid_final.transpose((1,2,0)))
        axarr[0].axis("off")
        axarr[1].axis("off")
        axarr[2].axis("off")
        axarr[3].axis("off")
        
        fig.savefig(img_final)
        plt.close()
                    


    def _save_learning_curve(self, dirs, loss_mean, loss_std, epoch: int):
        dir_save    = dirs.list_dir['train_loss'] 
        # file_loss = 'loss_epoch_{:05d}.png'.format(epoch)
        file_loss = 'loss.png'.format(epoch)
        file_loss = os.path.join(dir_save, file_loss)
        fig = plt.figure()
        
        plt.subplot(1,2,1)
        plt.plot(np.array(loss_mean), color='red')
        plt.fill_between(list(range(len(loss_mean))), np.array(loss_mean)-np.array(loss_std), np.array(loss_mean)+np.array(loss_std), color='blue', alpha=0.2)
        plt.title('loss')
        
        plt.tight_layout()
        plt.savefig(file_loss, bbox_inches='tight', dpi=100)
        plt.close(fig)
 
 
    def _save_prediction_curve(self, dirs, prediction_real_mean, prediction_real_std, prediction_fake_mean, prediction_fake_std, epoch: int):
        dir_save    = dirs.list_dir['train_pred'] 
        # file_pred = 'pred_epoch_{:05d}.png'.format(epoch)
        file_pred = 'pred.png'.format(epoch)
        file_pred = os.path.join(dir_save, file_pred)
        fig = plt.figure()
        
        plt.subplot(1,2,1)
        plt.plot(np.array(prediction_real_mean), color='red')
        plt.fill_between(list(range(len(prediction_real_mean))), np.array(prediction_real_mean)-np.array(prediction_real_std), np.array(prediction_real_mean)+np.array(prediction_real_std), color='blue', alpha=0.2)
        plt.title('prediction (real)')
        
        plt.subplot(1,2,2)
        plt.plot(np.array(prediction_fake_mean), color='red')
        plt.fill_between(list(range(len(prediction_fake_mean))), np.array(prediction_fake_mean)-np.array(prediction_fake_std), np.array(prediction_fake_mean)+np.array(prediction_fake_std), color='blue', alpha=0.2)
        plt.title('prediction (fake)')
        
        plt.tight_layout()
        plt.savefig(file_pred, bbox_inches='tight', dpi=100)
        plt.close(fig)
 
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
        
        
    def _save_images_to_grid(self, images, grid_size, save_path):
        """
        Save a batch of images to a single grid image.

        Parameters:
        - images: NumPy array with shape (batch_size, height, width, channels)
        - grid_size: Tuple (rows, cols) specifying the grid layout for the images
        - save_path: Path to save the resulting grid image
        """
        batch_size, height, width, channels = images.shape
        rows, cols = grid_size

        # Calculate total rows and cols required for the grid
        total_rows = math.ceil(batch_size / cols)
        total_cols = cols

        # Create a blank canvas for the grid
        grid_image = np.zeros((total_rows * height, total_cols * width, channels), dtype=images.dtype)

        # Fill the grid with images
        for i in range(batch_size):
            row = i // total_cols
            col = i % total_cols
            grid_image[row * height:(row + 1) * height, col * width:(col + 1) * width, :] = images[i]

        # Plot the grid image using matplotlib
        fig, ax = plt.subplots()
        ax.imshow(grid_image)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.axis('off')

        # Save the resulting grid image
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()