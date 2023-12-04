
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim

import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path
import numpy as np
import json
import random

import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate import data_loader
from accelerate.logging import get_logger
from packaging import version

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from diffusers.training_utils import EMAModel

from utils.visualizer import Visualizer

from trainer_masked import Trainer as BaseTrainer
from trainer_masked_shift import Trainer as ShiftTrainer
from trainer_masked_mean_shift import Trainer as MeanShiftTrainer
from trainer_test_template import Trainer as TestTrainer

import utils.datasetutils as datasetutils
import utils.datasetutilsHugging as datasetHugging
import utils.dirutils as dirutils

# data_loader._PYTORCH_DATALOADER_KWARGS["shuffle"] = True

def get_dataset(data_path: str, data_name: str, data_set: str,  data_height: int, data_width: int, data_subset: bool, data_subset_num: int, method: str):
    
    if 'hugging' in data_path:
        # datset using huggingface
        dataset = datasetHugging.DatasetUtils(data_path, data_name, data_set, data_height, data_width, data_subset, data_subset_num, method)
        # dataset = datasetutils.DatasetUtils(data_path, data_name, data_set, data_height, data_width, data_subset, data_subset_num)
    else:
        # dataset using torch
        dataset = datasetutils.DatasetUtils(data_path, data_name, data_set, data_height, data_width, data_subset, data_subset_num)
    
    return dataset
 

def get_dataloader(dataset: Dataset, batch_size: int, num_workers: int):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True, 
        num_workers=num_workers,  # not working for CPU
        )
    return dataloader
    
        
def get_model(args: dict):
    if args.model == 'default':
        model   = UNet2DModel(
            sample_size=args.data_size,
            in_channels=args.in_channel,
            out_channels=args.out_channel,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    else:
        config  = UNet2DModel.load_config(args.model)
        model   = UNet2DModel.from_config(config)
    
    return model

def get_ema(args:dict, model):
    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )
        return ema_model
    
    else:
        return None
   

def get_optimizer(model, optim_name: str, lr: float):
    if optim_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optim_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    return optimizer


def get_lr_scheduler(scheduler_name: str, optimizer: optim.Optimizer, dataloader: DataLoader, lr_warmup_steps, gradient_accumulation_steps, num_epochs, num_cycles):
    '''
    Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
    '''
    if scheduler_name == "cosine":
        scheduler   = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps, num_training_steps=len(dataloader)*num_epochs, num_cycles=num_cycles)

    elif scheduler_name == "constant":
        scheduler   = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps)
        
    elif scheduler_name == "linear":
        scheduler   = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps, num_training_steps=len(dataloader)*num_epochs)
        
    
    # scheduler    = get_scheduler(
    #     scheduler_name, optimizer, num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps, num_training_steps=(len(dataloader)*num_epochs)
    # )
    
    return scheduler 


def get_noise_scheduler():
    
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_schedule,
            # prediction_type=args.prediction_type,
            prediction_type="epsilon",
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_schedule)
        
    return noise_scheduler


def get_accelerator(args, ema_model):
    kwargs      = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[kwargs],
    )  
    
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model   = models.pop()

                # load diffusers style into model
                load_model  = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        
    return accelerator
        
def get_weight_type(args, accelerator):
    weight_dtype    = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype            = torch.float16
        args.mixed_precision    = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype            = torch.bfloat16
        args.mixed_precision    = accelerator.mixed_precision
        
    args.weight_dtype           = weight_dtype
    
def get_log(args, logger, dataset, total_batch_size, max_train_setps):
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_setps}")
    

def resume_train(args, accelerator, num_update_steps_per_epoch):
    global_step, first_epoch    = 0, 0
    
    if args.resume_from_checkpoint != "latest":
            path    = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs    = os.listdir(args.output_dir)
        dirs    = [d for d in dirs if d.startswith("checkpoint")]
        dirs    = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path    = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step  = global_step * args.gradient_accumulation_steps
        first_epoch         = global_step // num_update_steps_per_epoch
        resume_step         = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    
    return global_step, first_epoch, resume_step

    
def main(dirs: dict, args: dict):
    
    dataset     = get_dataset(args.dir_dataset, args.data_name, args.data_set, args.data_size, args.data_size, args.data_subset, args.data_subset_num, args.method)
    dataloader  = get_dataloader(dataset, args.batch_size, args.num_workers)
    
    model       = get_model(args)
    ema_model   = get_ema(args, model)
    accelerator = get_accelerator(args, ema_model)
    
    get_weight_type(args, accelerator)
    
    optimizer       = get_optimizer(model, args.optim, args.lr)
    lr_scheduler    = get_lr_scheduler(args.lr_scheduler, optimizer, dataloader, args.lr_warmup_steps, args.gradient_accumulation_steps, args.num_epochs, args.lr_cycle)
    
    model, optimizer, dataloader, lr_scheduler  = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    
    if args.use_ema:
        ema_model.to(accelerator.device)
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)
        
    total_batch_size            = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch  = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps             = args.num_epochs * num_update_steps_per_epoch
    logger  = get_logger(__name__, log_level="INFO")
    get_log(args, logger, dataset, total_batch_size, max_train_steps)
    
    if accelerator.is_main_process:
        visualizer  = Visualizer(args)
    else:
        visualizer  = None
    
    if args.resume_from_checkpoint:
        global_step, first_epoch, resume_step   = resume_train(args, accelerator, num_update_steps_per_epoch)
    else:
        global_step, first_epoch, resume_step  = 0, 0, 0
        
    if args.method.lower()  == 'base':
        trainer = BaseTrainer(args, dataloader, model, ema_model, optimizer, lr_scheduler, accelerator)
    elif args.method.lower() == 'shift':
        trainer = ShiftTrainer(args, dataloader, model, ema_model, optimizer, lr_scheduler, accelerator)
    elif args.method.lower() == 'mean_shift':
        trainer = MeanShiftTrainer(args, dataloader, model, ema_model, optimizer, lr_scheduler, accelerator)
    elif args.method.lower() == 'test':
        trainer = TestTrainer(args, dataloader, model, ema_model, optimizer, lr_scheduler, accelerator)
        
    trainer.train(first_epoch, args.num_epochs, resume_step, global_step, dirs, visualizer)
 

def save_option(args, dir_save: str):
    filename = 'option.ini'
    filename = os.path.join(dir_save, filename)
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        f.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ======================================================================
    # input to the [dirutilis]
    # ======================================================================
    parser.add_argument('--use_wandb', type=eval, default=True, choices=[True, False])
    parser.add_argument('--task', help='name of the task', type=str, choices=['train', 'sample', 'dataset'], default='train')
    parser.add_argument('--content', help='name of the content', type=str, default='test_code')
    parser.add_argument('--dir_work', help='path to the working directory', type=str, default='./')
    parser.add_argument('--dir_dataset', help='path to the original dataset', type=str, default='/nas2/dataset')
    parser.add_argument('--data_name', help='name of the dataset', type=str, default='mnist')
    parser.add_argument('--data_set', help='name of the subset of the dataset', type=str, default='train')
    parser.add_argument('--data_size', help='size of the data', type=int, default=64)
    parser.add_argument('--data_subset', help='using data subset', type=eval, default=False)
    parser.add_argument('--data_subset_num', help='number of data subset', type=int, default=1000)
    parser.add_argument('--date', help='date of the program execution', type=str, default='')
    parser.add_argument('--time', help='time of the program execution', type=str, default='')
    parser.add_argument('--method', help='algorithm of code', type=str, default='base')
    parser.add_argument('--title', help='title of experiment', type=str, default='')
    # ======================================================================
    parser.add_argument('--model', help='name of the neural network', type=str, default='default')
    parser.add_argument('--batch_size', help='mini-batch size', type=int, default=128)
    parser.add_argument('--in_channel', help='number of channel of input', type=int, default=3)
    parser.add_argument('--out_channel', help='number of channel of output', type=int, default=3)
    parser.add_argument('--num_epochs', help='number of epochs', type=int, default=1000)
    parser.add_argument('--optim', help='name of the optimizer', type=str, choices=(['adam', 'adamw', 'sgd']), default='adamw')
    parser.add_argument('--lr', help='learning rate (maximum)', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', help='learning rate scheduler', type=str, default='linear')
    parser.add_argument('--lr_warmup_steps', help='number of steps for the warmup in the lr scheduler', type=int, default=500)
    parser.add_argument('--lr_cycle', help='number of cycle of the lr scheduler', type=float, default=0.5)
    parser.add_argument('--gradient_accumulation_steps', help='number of updates steps to accumulate before performing a backward/update pass', type=int, default=1)
    parser.add_argument('--mixed_precision', help='use mixed precision', type=str, default="no", choices=["no", "fp16", "bf16"],)
    # ======================================================================
    parser.add_argument('--use_ema', help='use of the exponential moving average', type=eval, default=True, choices=[True, False])
    parser.add_argument('--ema_inv_gamma', help='the inverse gamma value for the EMA decay', type=float, default=1.0)
    parser.add_argument('--ema_power', help='the power value for the EMA decay', type=float, default=3/4)
    parser.add_argument('--ema_max_decay', help='the maximum decay magnitude for EMA', type=float, default=0.9999)
    parser.add_argument('--loss_weight_use', help='use weighted loss per timesteps', type=eval, default=False)
    parser.add_argument('--loss_weight_power_base', type=float, default=10.0)
    parser.add_argument('--ddpm_num_steps', type=int, default=1000)
    parser.add_argument('--updated_ddpm_num_steps', help='removed duplicated time step after time scheduling', type=int, default=1000)
    parser.add_argument("--ddpm_schedule", type=str, default="linear")
    # ======================================================================
    parser.add_argument('--sample_num', help='number of samples during the training', type=int, default=100)
    parser.add_argument('--sample_epoch_ratio', help='ratio of the epoch length for the training', type=float, default=0.2)
    parser.add_argument('--resume_from_checkpoint', help='resume training', type=eval, default=False)
    parser.add_argument('--num_workers', help='number of workers', type=int, default=32)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    
    args = parser.parse_args()
   
    # ======================================================================
    # directories to save results 
    # ======================================================================
    dirs = dirutils.Dir(
        task=args.task,
        content=args.content,
        dir_work=args.dir_work, 
        dir_dataset=args.dir_dataset, 
        data_name=args.data_name, 
        data_set=args.data_set, 
        data_size=args.data_size, 
        date=args.date, 
        time=args.time,
        method=args.method,
        title=args.title,
        )

    # ======================================================================
    # random seed
    # ======================================================================
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    
    save_option(args, dirs.list_dir['option'])
    main(dirs, args)
