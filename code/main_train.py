
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import os
import time as ti
from os.path import join
import argparse
import configparser
import sys
import datetime 
import random
import re
import numpy as np
import json
import socket
import matplotlib.pyplot as plt

from trainer import Trainer
import utils.datasetutils as datasetutils
import utils.dirutils as dirutils
import models.models_Mnist as models_Mnist


def get_dataset(data_path: str, data_name: str, data_set: str, data_subset_use: bool, data_subset_label: int, data_height: int, data_width: int, num_data: int=0):
    dataset = datasetutils.DatasetUtils(data_path, data_name, data_set, data_subset_use, data_subset_label, data_height, data_width, num_data)
    return dataset
 

def get_dataloader(dataset: Dataset, batch_size: int, num_workers: int):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True, 
        # num_workers=num_workers,  # not working for CPU
        )
    return dataloader
    
        
def get_model(in_channels: int, out_channels: int, dim_latent: int, dim_features: int):
    G = models_Mnist.Generator(dim_latent, dim_features, out_channels) 
    D = models_Mnist.Discriminator(in_channels, dim_features) 
    return G, D
   

def get_optimizer(model: Module, optim_name: str, lr: float):
    if optim_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optim_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    return optimizer


def get_lr_scheduler(optimizer: optim.Optimizer, dataloader: DataLoader, scheduler_name: str, epoch_length: int, lr_min: float, lr_max: float):
    if scheduler_name.lower() == 'cosineannealinglr':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=len(dataloader) * epoch_length,
            last_epoch=-1,
            eta_min=lr_min,
            # verbose=True
            )
    elif scheduler_name.lower() == 'onecyclelr':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer, 
            max_lr=lr_max,
            epochs=epoch_length, 
            steps_per_epoch=len(dataloader),
            last_epoch=-1,
            anneal_strategy='cos',
            pct_start=0.3,  # The percentage of the cycle (in number of steps) spent increasing the learning rate (default: 0.3)        
            div_factor=10,  # initial_lr = max_lr/div_factor (default: 25)
            final_div_factor=100,  # min_lr = initial_lr/final_div_factor (default: 10,000)
            cycle_momentum=False,
            # verbose=True
            )
    elif scheduler_name.lower() == 'cycliclr':
        scheduler = lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=lr_min,
            max_lr=lr_max,
            step_size_up=len(dataloader),
            mode="exp_range",
            cycle_momentum=False,
            # verbose=True,
            )
    elif scheduler_name.lower() == 'triangular':
        scheduler = lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=lr_min,
            max_lr=lr_max,
            step_size_up=len(dataloader),
            mode="triangular",
            cycle_momentum=False,
            # verbose=True,
            )
    return scheduler 

    
def main(device, dirs: dict, args: dict):
    dataset     = get_dataset(args.dir_dataset, args.data_name, args.data_set, args.data_subset_use, args.data_subset_label, args.data_size, args.data_size)
    dataloader  = get_dataloader(dataset, args.batch_size, args.num_workers)
    data_shape  = dataset[0][0].shape
    dim_channel = data_shape[0]

    G, D    = get_model(dim_channel, dim_channel, args.dim_latent, args.dim_feature)
    optim_G = get_optimizer(G, args.optim, args.lr_generator_max)
    optim_D = get_optimizer(D, args.optim, args.lr_discriminator_max)
    lr_scheduler_G = get_lr_scheduler(optim_G, dataloader, args.lr_scheduler, args.epoch_length, args.lr_generator_min, args.lr_generator_max)
    lr_scheduler_D = get_lr_scheduler(optim_D, dataloader, args.lr_scheduler, args.epoch_length, args.lr_discriminator_min, args.lr_discriminator_max)
    trainer = Trainer(device, args.dim_latent, args.batch_size, dataloader, G, D, optim_G, optim_D, lr_scheduler_G, lr_scheduler_D, args.weight_reg, args.langevin_length, args.langevin_lr, args.langevin_noise_lr, args.save_every)
    trainer.train(args.epoch_resume, args.epoch_length, dirs)
 

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
    parser.add_argument('--task', help='name of the task', type=str, choices=['train', 'sample', 'dataset'], default='train')
    parser.add_argument('--dir_work', help='path to the working directory', type=str, default='/home/hong/work/generative_wong')
    parser.add_argument('--dir_dataset', help='path to the dataset', type=str, default='/home/hong/dataset')
    parser.add_argument('--data_name', help='name of the dataset', type=str, default='mnist')
    parser.add_argument('--data_set', help='name of the subset of the dataset', type=str, choices=['train', 'test', 'eval', 'val', 'all'], default='train')
    parser.add_argument('--data_subset_use', help='use of the subset in the dataset', type=eval, default=True, choices=[True, False])
    parser.add_argument('--data_subset_label', help='label of the subset in the dataset', nargs='+', type=int, default=[5])
    parser.add_argument('--data_size', help='size of the data', type=int, default=32)
    parser.add_argument('--date', help='date of the program execution', type=str, default='')
    parser.add_argument('--time', help='time of the program execution', type=str, default='')
    # ======================================================================
    parser.add_argument('--dim_latent', help='dimension of the latent vector', type=int, default=128)
    parser.add_argument('--dim_feature', help='dimension of the features in the model', type=int, default=8)
    parser.add_argument('--batch_size', help='mini-batch size', type=int, default=625)
    parser.add_argument('--epoch_length', help='number of epochs', type=int, default=300)
    parser.add_argument('--optim', help='name of the optimizer', type=str, choices=(['adam', 'adamw', 'sgd']), default='adamw')
    parser.add_argument('--lr_scheduler', help='learning rate scheduler', type=str, default='CyclicLR')
    parser.add_argument('--lr_generator_min', help='learning rate for the generator', type=float, default=0.001)
    parser.add_argument('--lr_generator_max', help='learning rate for the generator', type=float, default=0.001)
    parser.add_argument('--lr_discriminator_min', help='learning rate for the discriminator', type=float, default=0.001)
    parser.add_argument('--lr_discriminator_max', help='learning rate for the discriminator', type=float, default=0.001)
    parser.add_argument('--weight_reg', help='weight for the regularization', type=float, default=0.001)
    parser.add_argument('--langevin_length', help='number of the langevin steps', type=int, default=10)
    parser.add_argument('--langevin_lr', help='lr for the langevin steps', type=float, default=1)
    parser.add_argument('--langevin_noise_lr', help='lr for noise of the langevin steps', type=float, default=0)
    parser.add_argument('--save_every', help='step of epoch for saving results', type=int, default=10)
    parser.add_argument('--epoch_resume', help='epoch of the model checkpoint', type=int, default=0)
    parser.add_argument('--num_workers', help='number of workers', type=int, default=4)
    parser.add_argument('--cuda_device', help='cuda device number', type=int, default=0)
    args = parser.parse_args()
   
    # ======================================================================
    # directories to save results 
    # ======================================================================
    dirs = dirutils.Dir(
        task=args.task,
        dir_work=args.dir_work, 
        dir_dataset=args.dir_dataset, 
        data_name=args.data_name, 
        data_set=args.data_set, 
        data_subset_use=args.data_subset_use, 
        data_subset_label=args.data_subset_label, 
        data_size=args.data_size, 
        date=args.date, 
        time=args.time,
        )

    # ======================================================================
    # random seed
    # ======================================================================
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
  
    # ======================================================================
    # setting device
    # ======================================================================
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    
    save_option(args, dirs.list_dir['option'])
    main(device, dirs, args)
