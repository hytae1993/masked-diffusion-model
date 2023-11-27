import torch
import torchvision
from torch.utils.data import Dataset
from os import listdir
from os import walk
from os.path import join
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import datasets
from datasets import load_dataset, concatenate_datasets
import albumentations
import albumentations.pytorch
import numpy as np
import re
from PIL import Image

'''
    - huggingface load_dataset manual
    https://huggingface.co/docs/datasets/v2.15.0/en/process
    https://huggingface.co/docs/datasets/loading
    https://huggingface.co/docs/datasets/v1.8.0/loading_datasets.html    
'''

'''
    basic structure
    
    - load dataset
    >>> dataset     = load_dataset('name', split='train')
    >>> first_img   = dataset['image'][0]
    >>> first_label = dataset['label'][0]
     
    - sort dataset respect to label
    >>> sorted_dataset = dataset.sort("label")
    
    - shuffle dataset (when load_dataset, data are already shuffled: fixed index)
    >>> shuffled_dataset = dataset.shuffle(seed=42)
    
    - select data
    >>> small_dataset   = dataset.select([0, 10, 20, 30, 40, 50])
    >>> even_dataset    = dataset.filter(lambda example, idx: idx % 2 == 0, with_indices=True)
    >>> 10%_datset      = datasets.load_dataset("bookcorpus", split="train[:10%]")
    
    - split one dataset to train and test
    >>> dataset.train_test_split(test_size=0.1)
    
    - split one dataset to specific number (ex. divide to four dataset and get first one)
    >>> dataset.shard(num_shards=4, index=0)
'''

def DatasetUtils(data_path: str, data_name: str, data_set: str, data_height: int, data_width: int, data_subset: bool, data_subset_num: int):
    datasets.config.DOWNLOADED_DATASETS_PATH = Path(data_path)
    
    transform_Gray  =   albumentations.Compose([
                        albumentations.Resize(data_width, data_height), 
                        # albumentations.RandomCrop(width=256, height=256),
                        # albumentations.HorizontalFlip(p=0.5),
                        # albumentations.RandomBrightnessContrast(p=0.2),
                        albumentations.pytorch.transforms.ToTensorV2(),
                    ])
    
    transform_RGB   =   albumentations.Compose([
                        albumentations.Resize(data_width, data_height), 
                        # albumentations.RandomCrop(width=256, height=256),
                        # albumentations.HorizontalFlip(p=0.5),
                        # albumentations.RandomBrightnessContrast(p=0.2),
                        albumentations.pytorch.transforms.ToTensorV2(),
                        # albumentations.Normalize(mean= (0.485, 0.456, 0.406), std= (0.229,0.224, 0.225)),
                    ])
    
    
    
    def transforms_RGB(examples):
        images = [transform_RGB(image=image.convert("RGB")) for image in examples["image"]]
        return {"image": images}

    # ======================================================================
    # mnist 
    # ======================================================================
    if data_name.lower() == 'mnist':
        
        def transforms_Mnist(examples):
            images  = [transform_Gray(image=np.array(image))["image"] for image in examples["image"]]
            labels  = [label for label in examples["label"]]
            return {"image": images, "label": labels}
        
        if data_set.lower() == 'train':
            if data_subset:
                # dataset = load_dataset("mnist", split="train[0:{}]".format(data_subset_num), cache_dir=data_path)
                # dataset[0] = {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x7F3CCEF72B90>, 'label': 5}
                
                dataset = load_dataset("mnist", split="train", cache_dir=data_path)
                dataset = dataset.filter(lambda example: example["label"] == 7)
            else:
                dataset = load_dataset("mnist", split="train", cache_dir=data_path)
        elif data_set.lower() == 'test':
            if data_subset:
                dataset = load_dataset("mnist", split="test[0:{}]".format(data_subset_num), cache_dir=data_path)
            else:
                dataset = load_dataset("mnist", split="test", cache_dir=data_path)
        elif data_set.lower() == 'all':
            # concatnate train set and test set as one dataset. 
            dataset = datasets.load_dataset("mnist", split="train+test", cache_dir=data_path)
            # dataset_train   = load_dataset("mnist", split="train", cache_dir=data_path, num_proc=num_workers)
            # dataset_test    = load_dataset("mnist", split="test", cache_dir=data_path, num_proc=num_workers)
            # dataset         = concatenate_datasets([dataset_train, dataset_test]) 

        dataset.set_transform(transforms_Mnist)
    
    # ======================================================================
    # cifar10 
    # ======================================================================
    elif data_name.lower() == 'cifar10':
        if data_set.lower() == 'train':
            dataset = torchvision.datasets.CIFAR10(data_path, transform=transform, train=True, download=True)
        elif data_set.lower() == 'test':
            dataset = torchvision.datasets.CIFAR10(data_path, transform=transform, train=False, download=True)
        elif data_set.lower() == 'all':
            dataset_train   = torchvision.datasets.CIFAR10(data_path, transform=transform, train=True, download=True)
            dataset_test    = torchvision.datasets.CIFAR10(data_path, transform=transform, train=False, download=True)
            dataset         = torch.utils.data.ConcatDataset([dataset_train, dataset_test]) 
        
        if data_subset_use == True: 
            dataset.data    = np.array(dataset.data)
            dataset.targets = np.array(dataset.targets)
            idx_label = np.zeros_like(dataset.targets, dtype=bool)
            for label in data_subset_label:
                idx_label = np.logical_or(idx_label, dataset.targets == label)
            dataset.data    = dataset.data[idx_label]
            dataset.targets = dataset.targets[idx_label]
            
            
    # if data_subset:
    #     dataset = Subset(dataset, list(range(0,data_subset_num)))

    return dataset




def get_image_files(folder):
    image_files = {}
    for root, _, files in walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files[file] = join(root, file)
    return image_files

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

if __name__ == '__main__':
    data_path           = '/nas2/dataset/hyuntae/'
    data_name           = 'cat2000'
    data_set            = 'train' 
    data_subset_use     = True
    data_subset_label   = [5]
    data_height         = 32 
    data_width          = 32 

    dataset = DatasetUtils(data_path, data_name, data_set, data_subset_use, data_subset_label, data_height, data_width)
    
    print(len(dataset))
    print(dataset[0])
