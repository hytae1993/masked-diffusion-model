import os
from os.path import join
import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
# from datasets import load_dataset, Image
# import wandb
import argparse
import configparser
import torch
from torchvision import datasets
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import csv

# from diffusers.utils import make_image_grid
# import struct

# 1e3b63afb62b52e0cfe8bbae214347743d793c28

def normalize01(data: torch.Tensor):
    batch_size  = data.shape[0]
    data_max    = torch.amax(data, dim=(1,2,3)).reshape(batch_size,1,1,1)
    data_min    = torch.amin(data, dim=(1,2,3)).reshape(batch_size,1,1,1)
    data_norm   = (data - data_min) / (data_max - data_min)
    return data_norm


def normalize01_global(data: torch.Tensor):
    data_max    = torch.max(data)
    data_min    = torch.min(data)
    data_norm   = (data - data_min) / (data_max - data_min)
    return data_norm


def whiten(data: torch.Tensor):
    batch_size  = data.shape[0]
    data_mean   = torch.mean(data, dim=(1,2,3)).reshape(batch_size,1,1,1)
    data_std    = torch.std(data, dim=(1,2,3)).reshape(batch_size,1,1,1)
    data_whiten = (data - data_mean) / data_std
    return data_whiten


def demean(data: torch.Tensor):
    batch_size  = data.shape[0]
    data_mean   = torch.mean(data, dim=(1,2,3)).reshape(batch_size,1,1,1)
    data_demean = data - data_mean
    return data_demean


def match_mean(source: torch.Tensor, target: torch.Tensor):
    batch_size  = source.shape[0]
    source_mean = torch.mean(source, dim=(1,2,3)).reshape(batch_size,1,1,1)
    target_mean = torch.mean(target, dim=(1,2,3)).reshape(batch_size,1,1,1)
    source_norm = source - source_mean + target_mean
    return source_norm

   
def get_dataset(path: str, name: str, size: int, split: str, data_subset: bool=False, num_data: int=0, use_augment: bool=False):
    if use_augment:
        transform = transforms.Compose([ 
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t - torch.mean(t)) / torch.std(t)), # mean 0, std 1
            # transforms.Normalize([0.5], [0.5]), # [0, 1] => [-1, 1] globally
            # torchvision.transforms.Lambda(lambda t: 2 * t - 1), 
            # transforms.Lambda(lambda t: (t - torch.mean(t))), # mean 0
            # transforms.Lambda(lambda t: (t - torch.mean(t)) / torch.std(t)), # mean 0, std 1
        ])
    else:
        transform = transforms.Compose([ 
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # [0, 1] => [-1, 1] globally
            # transforms.Lambda(lambda t: (t - torch.mean(t)) / torch.std(t)), # mean 0, std 1
        ])
    
    # ======================================================================
    # mnist 
    # ======================================================================
    if name.lower() == 'mnist':
        data_path = path
        
        if split == 'train':
            dataset = datasets.MNIST(data_path, transform=transform, train=True, download=True)
        elif split == 'test':
            dataset = datasets.MNIST(data_path, transform=transform, train=False, download=True)
        elif split == 'all':
            dataset_train   = datasets.MNIST(data_path, transform=transform, train=True, download=True)
            dataset_test    = datasets.MNIST(data_path, transform=transform, train=False, download=True)
            dataset         = torch.utils.data.ConcatDataset([dataset_train, dataset_test]) 

    # ======================================================================
    # cifar10 
    # ======================================================================
    elif name.lower() == 'cifar10':
        data_path = os.path.join(path, 'CIFAR')
        
        if split == 'train':
            dataset = datasets.CIFAR10(data_path, transform=transform, train=True, download=True)
        elif split == 'test':
            dataset = datasets.CIFAR10(data_path, transform=transform, train=False, download=True)
        elif split == 'all':
            dataset_train   = datasets.CIFAR10(data_path, transform=transform, train=True, download=True)
            dataset_test    = datasets.CIFAR10(data_path, transform=transform, train=False, download=True)
            dataset         = torch.utils.data.ConcatDataset([dataset_train, dataset_test]) 
        
    # ======================================================================
    # flowers102 
    # ======================================================================
    elif name.lower() == 'flowers102':
        data_path = os.path.join(path, name.lower())

        if split == 'train' or split == 'val' or split == 'test':
            dataset = datasets.Flowers102(data_path, transform=transform, split=split, download=True)
        elif split == 'all':
            dataset_train   = datasets.Flowers102(data_path, transform=transform, split="train", download=True)
            dataset_val     = datasets.Flowers102(data_path, transform=transform, split="val", download=True)
            dataset_test    = datasets.Flowers102(data_path, transform=transform, split="test", download=True)
            dataset         = torch.utils.data.ConcatDataset([dataset_train, dataset_val, dataset_test]) 
        
    # ======================================================================
    # LSUN 
    # ======================================================================
    elif name.lower() == 'lsun':
        data_path = os.path.join(path, name.lower())

        if split == 'church':
            dataset = datasets.LSUN(data_path, classes=['church_outdoor_train'], transform=transform)
        elif split == 'bedroom':
            dataset = datasets.LSUN(data_path, classes=['bedroom_train'], transform=transform)
        elif split == 'tower':
            dataset = datasets.LSUN(data_path, classes=['tower_train'], transform=transform)

    # ======================================================================
    # celeba_hq 
    # source: https://github.com/clovaai/stargan-v2/blob/master/README.md
    # ======================================================================
    elif name.lower() == 'celeba_hq':
        # female: 0, male: 1
        if split == 'train' or split == 'val':
            data_path   = os.path.join(path, name.lower(), split)
            dataset     = datasets.ImageFolder(root=data_path, transform=transform)
        elif split == 'all':
            data_path_train = os.path.join(path, name.lower(), 'train')
            dataset_train   = datasets.ImageFolder(root=data_path_train, transform=transform)
            data_path_val   = os.path.join(path, name.lower(), 'val')
            dataset_val     = datasets.ImageFolder(root=data_path_val, transform=transform)
            dataset         = torch.utils.data.ConcatDataset([dataset_train, dataset_val]) 

    # ======================================================================
    # afhqv2
    # source: https://github.com/clovaai/stargan-v2/blob/master/README.md
    # ======================================================================
    elif name.lower() == 'afhqv2':
        # cat: 0, dog: 1, wild: 2
        if split == 'train' or split == 'test':
            data_path   = os.path.join(path, name.lower(), split)
            dataset     = datasets.ImageFolder(root=data_path, transform=transform)
            '''
            if data_subset_use == True:
                dataset = create_subset(dataset, data_subset_label)
            ''' 
        elif split == 'all':
            data_path_train = os.path.join(path, name.lower(), 'train')
            dataset_train   = datasets.ImageFolder(root=data_path_train, transform=transform)
            data_path_test  = os.path.join(path, name.lower(), 'test')
            dataset_test    = datasets.ImageFolder(root=data_path_test, transform=transform)
            dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test]) 

    # ======================================================================
    # metfaces
    # source: https://github.com/NVlabs/metfaces-dataset
    # ======================================================================
    elif name.lower() == 'metfaces':
        # all: 0
        if (split == 'all') or (split == 'train'):
            data_path   = os.path.join(path, name.lower())    # "images" and "unprocessed"
            dataset     = datasets.ImageFolder(root=data_path, transform=transform)
            # dataset_all = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
            # dataset     = create_subset(dataset_all, [0])
       
    # ======================================================================
    # stanford cars
    # ======================================================================
    elif name.lower() == 'stanfordcars':
        # all: 0
        if (split == 'all') or (split == 'train'):
            data_path   = os.path.join(path, name.lower())    # "images" and "unprocessed"
            dataset     = datasets.ImageFolder(root=data_path, transform=transform)
            # dataset_all = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
            # dataset     = create_subset(dataset_all, [0])
     
    # ======================================================================
    # number of data in the dataset
    # ======================================================================
    if data_subset:
        dataset = torch.utils.data.Subset(dataset, range(0, num_data))
            
    # ======================================================================
    # return dataset 
    # ======================================================================
    return dataset


def save_dataset(dataset, path_save):
    dataloader      = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    file_save_count = os.path.join(path_save, 'count.csv')
    file_save_label = os.path.join(path_save, 'label.csv')
    number_data     = len(dataloader)
    label_list      = []
   
    for i, (image, label) in enumerate(dataloader):
        print(i+1, '/', len(dataloader))
        label_list.append(label.item())
        file_save_image = os.path.join(path_save, '{:06d}.pt'.format(i))  # start from 0 to N_1 (N: number of image)
        torch.save(image.clone().cpu(), file_save_image)
    
    with open(file_save_label, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(label_list)
   
    f = open(file_save_count, 'a')
    f.write(str(number_data))
    f.close()


class MyDataset(Dataset):
    def __init__(self, path: str, name: str, size: int, split: str, data_subset: bool=False, num_data: int=0, use_augment: bool=False, use_label: bool=False, label: int=0, num_timesteps: int=1):
        self.data           = None 
        self.lable          = None 
        self.random         = None 
        self.transform      = None 

        if use_augment: 
            self.transform = transforms.RandomHorizontalFlip() 

        dataset = get_dataset(path, name, size, split, data_subset, num_data) 
        self.number_data = len(dataset)
        
        data        = dataset[0][0] # data of the first element
        dim_channel = data.shape[0]
        dim_height  = data.shape[1]
        dim_width   = data.shape[2]
        self.data   = torch.zeros(self.number_data, dim_channel, dim_height, dim_width)
        self.label  = torch.zeros(self.number_data)
        self.random = torch.zeros(self.number_data)
       
        print('number of data =', self.number_data, ', channel =', dim_channel, ', height =', dim_height, ', width =', dim_width, flush=True) 

        # associate a fixed random shift value with each data 
        min_value   = -1.0
        max_value   = +1.0
        self.random = torch.FloatTensor(self.number_data, num_timesteps).uniform_(min_value, max_value)
        
        for i in range(self.number_data):
            self.data[i]    = dataset[i][0]
            self.label[i]   = dataset[i][1]

    def __len__(self):
        return self.number_data

    def __getitem__(self, idx):
        data    = self.data[idx]
        label   = self.label[idx]
        random  = self.random[idx]  # vector: random[t] : for the time step t
       
        if self.transform:
            sample = self.transform(sample)
             
        return data, label, random


'''
class MyDataset_from_file(Dataset):
    def __init__(self, path: str, name: str, size: int, split: str, use_label: bool=False, label: int=0, num_data: int=0, use_augment: bool=False):
        self.path           = os.path.join(path, name, '{:04d}'.format(size), split)
        self.data           = None 
        self.number_data    = None 
        self.random         = None 
        self.lable          = None 
        self.transform      = None 
     
        file_count  = os.path.join(self.path, 'count.csv')
        file_label  = os.path.join(self.path, 'label.csv')
        file_data   = os.path.join(self.path, '{:06d}.pt'.format(0))  # start from 0 to N_1 (N: number of image)

        if num_data > 0:
            self.number_data = num_data
        else:
            f = open(file_count, 'r')
            number_data = f.read()
            f.close()
            self.number_data = int(number_data)

        # with open(file_label, 'r', newline='') as f:
        #     reader = csv.reader(f, delimiter=',')
        #     self.label = reader.readrow()
  
        self.label = torch.zeros((self.number_data))

        
        
        data        = torch.load(file_data)
        dim_channel = data.shape[1]
        dim_height  = data.shape[2]
        dim_width   = data.shape[3]

        self.data = torch.zeros((self.number_data, dim_channel, dim_height, dim_width))
        print('number of data=', self.number_data, 'channel=', dim_channel, 'height=', dim_height, 'width=', dim_width) 

        # associate a fixed random shift value with each data 
        min_value   = -1.0
        max_value   = +1.0
        self.random = torch.FloatTensor(self.number_data).uniform_(min_value, max_value)
        
        for i in range(self.number_data):
            file_data       = os.path.join(self.path, '{:06d}.pt'.format(i))  # start from 0 to N_1 (N: number of image)
            data            = torch.load(file_data)
            self.data[i]    = data
       
        if use_augment:
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip()])
    

    def __len__(self):
        return self.number_data


    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #    idx = idx.tolist()

        # file_data   = os.path.join(self.path, '{:06d}.pt'.format(idx))  # start from 0 to N_1 (N: number of image) 
        # data        = torch.load(file_data)
        data        = self.data[idx]
        random      = self.random[idx] 
        label       = self.label[idx]
        
        if self.transform is not None:
            data = self.transform(data)

        return data, random, label
'''
    


 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to the dataset to load', type=str, default='/home/hong/dataset_image')
    parser.add_argument('--name', help='name of the dataset', type=str, default='metfaces')
    parser.add_argument('--size', help='size of the data', type=int, default=32)
    parser.add_argument('--split', help='split of the dataset', type=str, choices=['train', 'test', 'eval', 'val', 'all'], default='train')
    parser.add_argument('--use_label', help='use the subset', type=eval, default=False, choices=[True, False])
    parser.add_argument('--label', help='label of the subset', type=int, default=0)
    parser.add_argument('--num_data', help='number of data', type=int, default=0)
    parser.add_argument('--save', help='(True/False) save to files (.pt)',  type=eval, default=False, choices=[True, False])
    args = parser.parse_args()

    if args.save:
        path_data   = '/home/hong/dataset'
        path_save   = os.path.join(args.path, args.name.lower(), '{:04d}'.format(args.size), args.split)
        os.makedirs(os.path.join(args.path, args.name.lower()), exist_ok=True)
        os.makedirs(os.path.join(args.path, args.name.lower(), '{:04d}'.format(args.size)), exist_ok=True)
        os.makedirs(os.path.join(args.path, args.name.lower(), '{:04d}'.format(args.size), args.split), exist_ok=True)
        dataset     = get_dataset(path_data, args.name, args.size, args.split)
        save_dataset(dataset, path_save)
    else:
        dataset     = MyDataset(args.path, args.name, args.size, args.split, args.use_label, args.label, args.num_data) 
        dataloader  = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
        
        iter_data   = iter(dataloader) 
        data, random, label = next(iter_data)
       
        print('=====') 
        print(data.shape, random, label) 
        print('min =', data.min().item(), 'max =', data.max().item(), 'mean =', data.mean().item(), 'std =', data.std().item())
        print(label)

        iter_data   = iter(dataloader) 
        data, random, label = next(iter_data)
        
        print('=====') 
        print(data.shape, random, label) 
        print('min =', data.min().item(), 'max =', data.max().item(), 'mean =', data.mean().item(), 'std =', data.std().item())
        print(label)
        
        batch = normalize01(data)
        image_grid = make_grid(batch, nrow=10, normalize=True)
        save_image(image_grid, 'test_dataloader.png')
        