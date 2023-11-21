import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from os import listdir
from os.path import join
from random import *
import csv
from torchvision.utils import save_image
from torchvision.utils import make_grid
from timeit import default_timer as timer
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, path_data: str, file_list_time_blur: str, file_number_data: str, file_label: str=None, use_data_augmentation: bool=True):
        self.path_data              = path_data
        self.use_data_augmentation  = use_data_augmentation
        
        with open(file_number_data, 'r', newline='') as f:
            self.number_data = int(f.read())
        f.close()
      
        with open(file_list_time_blur, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            self.list_time = next(reader)
        f.close()
      
        with open(file_label, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            self.label = next(reader)
        f.close()
       
        # self.list_time: 1, 2, 3, 4, ..., T (excluding the original t=0)
        self.length_time = len(self.list_time) + 1
        # print('# of data:', self.number_data, '# of label:', len(self.label), 'number of time step including the averaging:', self.length_time)
        print('# of data:', self.number_data, 'number of time step including the averaging:', self.length_time)

        self.transform = transforms.Compose([ 
            transforms.RandomHorizontalFlip(),
        ])


    def _set_number_data(self, number_data: int):
        self.number_data = number_data


    def _get_number_data(self):
        return self.number_data

 
    def _set_length_time(self,length_time: int):
        self.length_time = length_time
        self.list_time[length_time:] = []


    def _get_length_time(self):
        return self.length_time
    

    def _get_list_time(self):
        return self.list_time
    
    
    def __len__(self):
        return self.number_data


    def __getitem__(self, idx):
        # time        = randint(1, self.length_time + 1)  # random integer from 1, 2, 3, ..., self.length_time (T) including the averaging t=T
        time = randint(1, self.length_time)  # random integer from 1, 2, 3, ..., self.length_time (T) including the averaging t=T

        # original data
        filename    = '{:06d}.pt'.format(idx)
        path_data_0 = os.path.join(self.path_data, 'time_000') 
        file_data_0 = os.path.join(path_data_0, filename)
        data_0      = torch.load(file_data_0) 
        time_index  = time - 1
        
        # last time step
        if time == self.length_time:
            # scale_noise = 0.1
            # noise       = torch.rand_like(data_0)
            # noise       = noise - noise.mean()
            data_t      = torch.zeros_like(data_0)
            # data_t      = data_t + scale_noise * noise
            time_value  = '100000000000'
        else: 
            path_data_t = os.path.join(self.path_data, 'time_{:03d}'.format(time))
            file_data_t = os.path.join(path_data_t, filename)
            data_t      = torch.load(file_data_t) 
            time_value  = self.list_time[time - 1]

        '''
        # the last time step (latent)
        if time == (self.length_time + 1):
            filename    = '{:06d}.pt'.format(idx)
            path_data_0 = os.path.join(self.path_data, 'time_{:03d}'.format(self.length_time))
            file_data_0 = os.path.join(path_data_0, filename)
            data_0      = torch.load(file_data_0) 
            data_t      = torch.randn_like(data_0)
            time_value  = '1000000000000'   # time_index == self.length_time : latent
            
            filename    = '{:06d}.pt'.format(idx)
            path_data_0 = os.path.join(self.path_data, 'time_000') 
            file_data_0 = os.path.join(path_data_0, filename)
            data_0      = torch.load(file_data_0) 
            data_t      = torch.randn_like(data_0)
            time_value  = '1000000000000'   # time_index == self.length_time : latent
        else:
            filename    = '{:06d}.pt'.format(idx)
            path_data_0 = os.path.join(self.path_data, 'time_000') 
            path_data_t = os.path.join(self.path_data, 'time_{:03d}'.format(time))
            file_data_0 = os.path.join(path_data_0, filename)
            file_data_t = os.path.join(path_data_t, filename)
            data_0      = torch.load(file_data_0) 
            data_t      = torch.load(file_data_t) 
            time_value  = self.list_time[time_index]
        ''' 
        
        if self.use_data_augmentation:
            # apply data augmentation (left-right flip) 
            dim_channel     = data_0.shape[0]
            data_cat        = torch.cat((data_0, data_t), dim=0)
            data_transform  = self.transform(data_cat)
            data_0          = data_transform[0:dim_channel]
            data_t          = data_transform[dim_channel:2*dim_channel]

        # label = int(self.label[idx])
        label = torch.Tensor([0])
        # label = torch.Tensor([label], dtype=torch.uint8)
        # label = torch.Tensor([label])
        
        return data_0, data_t, time_index, time_value, label


    # index_time: 0 (original), 1, 2, ..., self.length_time (averaging) 
    def get_data(self, index_data, index_time, use_data_augmentation: bool=False):
        data_batch = torch.Tensor() 
        for i in range(len(index_data)):
            index       = index_data[i]
            path_data   = os.path.join(self.path_data, 'time_{:03d}'.format(index_time))
            file_data   = '{:06d}.pt'.format(index)
            file_data   = os.path.join(path_data, file_data)
            data        = torch.load(file_data)
            data        = data.unsqueeze(0)
            data_batch  = torch.cat((data_batch, data), dim=0)

        if use_data_augmentation:
            data_batch = self.transform(data_batch)
        return data_batch
   

    # index_time: 0 (original), 1, 2, ..., self.length_time (averaging) 
    def get_data_all(self, index_time):
        data_batch = torch.Tensor() 
        for i in range(self.number_data):
            index       = i
            path_data   = os.path.join(self.path_data, 'time_{:03d}'.format(index_time))
            file_data   = '{:06d}.pt'.format(index)
            file_data   = os.path.join(path_data, file_data)
            data        = torch.load(file_data)
            data        = data.unsqueeze(0)
            data_batch  = torch.cat((data_batch, data), dim=0)
        return data_batch
     
    
def normalize_mean_channel(source, target):
    batch_size  = source.shape[0]
    dim_channel = source.shape[1]
    source_mean = torch.mean(source, dim=(2,3)).reshape(batch_size,dim_channel,1,1)
    target_mean = torch.mean(target, dim=(2,3)).reshape(batch_size,dim_channel,1,1)
    source_norm = source - source_mean + target_mean
    return source_norm


# source and target should have the same mean and std
def normalize(source, target):
    batch_size  = source.shape[0]
    source_mean = torch.mean(source, dim=(1,2,3)).reshape(batch_size,1,1,1)
    target_mean = torch.mean(target, dim=(1,2,3)).reshape(batch_size,1,1,1)
    source_std  = torch.std(source, dim=(1,2,3)).reshape(batch_size,1,1,1)
    # target_std  = torch.std(target, dim=(1,2,3)).reshape(batch_size,1,1,1)
    source_norm = (source - source_mean) / source_std 
    # source_norm = (source - source_mean) / source_std * target_std
    source_norm = source_norm + target_mean
    return source_norm


def normalize_channel(source, target):
    batch_size  = source.shape[0]
    dim_channel = source.shape[1]
    source_mean = torch.mean(source, dim=(2,3)).reshape(batch_size,dim_channel,1,1)
    target_mean = torch.mean(target, dim=(2,3)).reshape(batch_size,dim_channel,1,1)
    source_std  = torch.std(source, dim=(2,3)).reshape(batch_size,dim_channel,1,1)
    target_std  = torch.std(target, dim=(2,3)).reshape(batch_size,dim_channel,1,1)

    source_norm = (source - source_mean) / source_std * target_std
    source_norm = source_norm + target_mean
    return source_norm


def normalize_mean(source, target):
    batch_size  = source.shape[0]
    source_mean = torch.mean(source, dim=(1,2,3)).reshape(batch_size,1,1,1)
    target_mean = torch.mean(target, dim=(1,2,3)).reshape(batch_size,1,1,1)
    source_norm = source - source_mean + target_mean
    return source_norm


def normalize01(data: torch.Tensor):
    batch_size  = data.shape[0]
    data_max    = torch.amax(data, dim=(1,2,3)).reshape(batch_size,1,1,1)
    data_min    = torch.amin(data, dim=(1,2,3)).reshape(batch_size,1,1,1)
    data_norm   = (data - data_min) / (data_max - data_min)
    data_norm   = torch.nan_to_num(data_norm, nan=0.0)
    return data_norm


def normalize01_global(data: torch.Tensor):
    data_max    = torch.max(data)
    data_min    = torch.min(data)
    data_norm   = (data - data_min) / (data_max - data_min)
    return data_norm


def make_mean_zero(data: torch.Tensor):
    batch_size  = data.shape[0]
    data_mean   = torch.mean(data, dim=(1,2,3)).reshape(batch_size,1,1,1)
    data_shift  = data - data_mean
    return data_shift

        
def whiten(data: torch.Tensor):
    batch_size  = data.shape[0]
    data_mean   = torch.mean(data, dim=(1,2,3)).reshape(batch_size,1,1,1)
    data_std    = torch.std(data, dim=(1,2,3)).reshape(batch_size,1,1,1)
    data_whiten = (data - data_mean) / data_std
    return data_whiten



if __name__ == '__main__':
    
    dir_work        = '/home/hong/dataset_blur'
    data_name       = 'lsun'
    data_set        = 'church'
    data_size       = 'size_{:04d}'.format(64)
    time_step       = 'time_base_{:.2f}'.format(1.2)
    dir_data        = os.path.join(dir_work, data_name, data_set, data_size, time_step)
    file_list_time  = os.path.join(dir_data, 'list_time_blur.csv')
    file_number_data= os.path.join(dir_data, 'number_data.csv')
    file_label      = os.path.join(dir_data, 'label.csv')
   
    print(file_list_time)
    
    dataset     = MyDataset(dir_data, file_list_time, file_number_data, file_label)
    dataloader  = DataLoader(dataset, batch_size=100, drop_last=True, shuffle=True, pin_memory=True)

    it = iter(dataloader)
    data_0, data_t, time_index, time_value, label = next(it)
    
    print('====================================================')
    print('label size:', label.shape) 
    print('label:', label) 
    print('====================================================')
    
    print(data_0.shape, data_t.shape, time_index, time_value)
    print(data_0.shape)
    print(data_t.shape)
    print('====================================================')
    print('mean') 
    print('====================================================')
    print('mean x_0', data_0.mean(dim=(1,2,3)))
    print('mean x_t', data_t.mean(dim=(1,2,3)))
    print('====================================================')
    print('std') 
    print('====================================================')
    print('std x_0', data_0.std(dim=(1,2,3)))
    print('std x_t', data_t.std(dim=(1,2,3)))
    print('====================================================')
    print('max') 
    print('====================================================')
    print('max x_0', data_0.max())
    print('max x_t', data_t.max())
    print('====================================================')
    print('min') 
    print('====================================================')
    print('min x_0', data_0.min())
    print('min x_t', data_t.min())
    print('====================================================')
    
    test = True
    if test == True:
        grid_data_0 = make_grid(data_0, nrow=10, normalize=True)
        grid_data_t = make_grid(data_t, nrow=10, normalize=True)
        save_image(grid_data_0, 'data0.png')
        save_image(grid_data_t, 'datat.png')
        