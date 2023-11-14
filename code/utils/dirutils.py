import os
from os.path import join
import argparse
import configparser
import sys
import datetime 


class Dir:
    def __init__(self, 
                 task: str, 
                 dir_work: str, 
                 dir_dataset: str, 
                 data_name: str,
                 data_set: str, 
                 data_size: int, 
                 date: str='', 
                 time: str='',
                 title: str=''):
        self.dir_work           = dir_work
        self.dir_dataset        = dir_dataset
        self.data_name          = data_name
        self.data_set           = data_set
        self.data_size          = data_size
        self.date               = date
        self.time               = time
        self.title              = title

        now = datetime.datetime.now()
        if date == '':
            self.date = now.strftime('%Y_%m_%d') 
        if time == '':
            self.time = now.strftime('%H_%M_%S')

        self.list_dir_sub   = self._build_dir_sub()
        self.list_dir       = self._build_dir(task)
        

    def _build_dir_sub(self):
        dir_data_name = self.data_name
        
        dir_data_set = self.data_set

        dir_data_size   = 'size_{:04d}'.format(self.data_size)
        dir_time        = '{}_{}'.format(self.date, self.time)  

        list_dir_sub = {
            'data_name' : dir_data_name,
            'data_set'  : dir_data_set,
            'data_size' : dir_data_size,
            'time'      : dir_time,
            'title'     : self.title,
            }
       
        return list_dir_sub


    def _build_dir(self, task: str):
        if task == 'train':
            list_dir = self._build_dir_train()
        elif task == 'sample':
            list_dir = self._build_dir_sample()
        
        return list_dir


    def _build_dir_train(self):
        
        save_dir    = os.path.join(self.dir_work, 'result', self.list_dir_sub['data_name'], self.list_dir_sub['time'], self.list_dir_sub['title'])
        
        dir_list = {
            'img'           : os.path.join(save_dir, 'train', 'image', 'img'),
            'train_img'     : os.path.join(save_dir, 'train', 'image', 'train_image'),
            'masked_img'    : os.path.join(save_dir, 'train', 'image', 'masked_image'),
            'noise_img'     : os.path.join(save_dir, 'train', 'image', 'noise_image'),
            'predict_img' : os.path.join(save_dir, 'train', 'image', 'predict_image'),
            'sample_img' : os.path.join(save_dir, 'train', 'image', 'sample_image'),
            'train_loss'    : os.path.join(save_dir, 'train', 'loss'),
            'train_pred'    : os.path.join(save_dir, 'train', 'prediction'),
            'log'           : os.path.join(save_dir, 'log'),
            'model'         : os.path.join(save_dir, 'model'),
            'option'        : os.path.join(save_dir, 'option'),
            'loss'          : os.path.join(save_dir, 'loss'),
            }      
        
        os.makedirs(dir_list['img'], exist_ok=True)
        os.makedirs(dir_list['train_img'], exist_ok=True)
        os.makedirs(dir_list['masked_img'], exist_ok=True)
        os.makedirs(dir_list['noise_img'], exist_ok=True)
        os.makedirs(dir_list['predict_img'], exist_ok=True)
        os.makedirs(dir_list['sample_img'], exist_ok=True)
        os.makedirs(dir_list['train_loss'], exist_ok=True)
        os.makedirs(dir_list['train_pred'], exist_ok=True)
        os.makedirs(dir_list['log'], exist_ok=True)
        os.makedirs(dir_list['model'], exist_ok=True)
        os.makedirs(dir_list['option'], exist_ok=True)
        os.makedirs(dir_list['loss'], exist_ok=True)
        
        return dir_list

        
    def _build_dir_sample(self):
        dir_list = {
            'sample'    : os.path.join(self.dir_work, 'sample'),
            'model'     : os.path.join(self.dir_work, 'model'),
            }
        os.makedirs(dir_list['sample'], exist_ok=True)

        dir_list['sample']  = os.path.join(dir_list['sample'], self.list_dir_sub['data_name'])
        dir_list['model']   = os.path.join(dir_list['model'], self.list_dir_sub['data_name'])
        os.makedirs(dir_list['sample'], exist_ok=True)

        dir_list['sample']  = os.path.join(dir_list['sample'], self.list_dir_sub['data_set'])
        dir_list['model']   = os.path.join(dir_list['model'], self.list_dir_sub['data_set'])
        os.makedirs(dir_list['sample'], exist_ok=True)
       
        dir_list['sample']  = os.path.join(dir_list['sample'], self.list_dir_sub['data_size'])
        dir_list['model']   = os.path.join(dir_list['model'], self.list_dir_sub['data_size'])
        os.makedirs(dir_list['sample'], exist_ok=True)
       
        dir_list['sample']  = os.path.join(dir_list['sample'], self.list_dir_sub['time'])
        dir_list['model']   = os.path.join(dir_list['model'], self.list_dir_sub['time'])
        os.makedirs(dir_list['sample'], exist_ok=True)
        
        return dir_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # =================  
    # input to the [dirutilis]
    parser.add_argument('--task', help='name of the task', type=str, choices=['train', 'sample'], default='train')
    parser.add_argument('--dir_work', help='path to the working directory', type=str)
    parser.add_argument('--dir_dataset', help='path to the dataset', type=str)
    parser.add_argument('--data_name', help='name of the dataset', type=str, default='mnist')
    parser.add_argument('--data_set', help='name of the subset of the dataset', type=str, choices=['train', 'test', 'eval', 'val', 'all'], default='train')
    parser.add_argument('--data_size', help='size of the data', type=int, default=32)
    parser.add_argument('--date', help='date of the program execution', type=str, default=None)
    parser.add_argument('--time', help='time of the program execution', type=str, default=None)
    # =================  
    args = parser.parse_args()

    dirs = Dir(
        task=args.task, 
        dir_work=args.dir_work,
        dir_dataset=args.dir_dataset, 
        data_name=args.data_name, 
        data_set=args.data_set, 
        data_size=args.data_size)
     
    print(dirs.list_dir_sub)
    print(dirs.list_dir)