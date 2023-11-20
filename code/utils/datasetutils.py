import torch
import torchvision
from torch.utils.data import Dataset
from os import listdir
from os import walk
from os.path import join
from torch.utils.data import DataLoader, Subset
import numpy as np
import re
from PIL import Image


def DatasetUtils(data_path: str, data_name: str, data_set: str, data_height: int, data_width: int, data_subset: bool, data_subset_num: int):
    transform_RGB = torchvision.transforms.Compose([ 
        torchvision.transforms.Resize([data_height, data_width]),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    transform_GRAY = torchvision.transforms.Compose([ 
        torchvision.transforms.Resize([data_height, data_width]),
        torchvision.transforms.ToTensor(),
        ])
    
    # ======================================================================
    # cat2000 - salinecy map benchmark dataset 
    # ======================================================================
    if data_name.lower() == 'cat2000':
        class CatDataset(Dataset):
            def __init__(self, work, transform=None):
                super(CatDataset, self).__init__()
                            
                domain_a_folders = ["Action", "Affective", "Art", "BlackWhite", "Cartoon", "Fractal",
                            "Indoor", "Inverted", "Jumbled", "LineDrawing",
                            "LowResolution", "Noisy", "Object", "OutdoorManMade", "OutdoorNatural",
                            "Pattern", "Random", "Satelite", "Sketch", "Social"]

                self.pairs = []
                
                path = join(data_path, data_name, work, 'Stimuli')
                
                for folder in domain_a_folders:
                    folder_img_path = join(path, folder)
                    folder_saliency_path = join(folder_img_path, 'Output')
                    
                    images_a = sorted(get_image_files(folder_img_path).values(), key=natural_sort_key)
                    images_b = sorted(get_image_files(folder_saliency_path).values(), key=natural_sort_key)
                    
                    for img_a, img_b in zip(images_a, images_b):
                        self.pairs.append((img_a, img_b))
                
                self.transform_RGB = transform_RGB
                self.transform_GRAY = transform_GRAY
                
            def __getitem__(self, index):
                img_a_path, img_b_path = self.pairs[index]
                
                img_a = Image.open(img_a_path).convert("RGB")
                img_b = Image.open(img_b_path).convert("L")

                if self.transform:
                    img_a = self.transform_RGB(img_a)
                    img_b = self.transform_GRAY(img_b)

                return img_a, img_b

            def __len__(self):
                return len(self.pairs)
            
        dataset_train   = CatDataset('trainSet', transform)
        dataset_test    = CatDataset('testSet', transform)
        
        # dataset         = torch.utils.data.ConcatDataset([dataset_train, dataset_test]) 
        # dataset         = [dataset_train, dataset_test]
        dataset         = dataset_train
        
    # ======================================================================
    # synthetic 
    # ======================================================================
    elif data_name.lower() == 'synthetic':
        class SyntheticDataset(Dataset):
            def __init__(self, transform=None):
                super(SyntheticDataset, self).__init__()
                            
                self.pairs = []
                
                path = join(data_path, data_name)
                
                folder_img_path = join(path, 'original')
                folder_saliency_path = join(path, 'saliency')
                folder_noisy_path = join(path, 'noisy')
                
                images_a = sorted(get_image_files(folder_img_path).values(), key=natural_sort_key)
                images_b = sorted(get_image_files(folder_saliency_path).values(), key=natural_sort_key)
                images_c = sorted(get_image_files(folder_noisy_path).values(), key=natural_sort_key)
                    
                for img_a, img_b, img_c in zip(images_a, images_b, images_c):
                    self.pairs.append((img_a, img_b, img_c))
                
                self.transform = transform
                
            def __getitem__(self, index):
                img_a_path, img_b_path, img_c_path = self.pairs[index]
                
                img_a = Image.open(img_a_path).convert("L")
                img_b = Image.open(img_b_path).convert("L")
                img_c = Image.open(img_c_path).convert("L")

                if self.transform:
                    img_a = self.transform(img_a)
                    img_b = self.transform(img_b)
                    img_c = self.transform(img_c)

                return img_a, img_b, img_c

            def __len__(self):
                return len(self.pairs)
            
        dataset_train   = SyntheticDataset(transform)
        
        # dataset         = torch.utils.data.ConcatDataset([dataset_train, dataset_test]) 
        # dataset         = [dataset_train, dataset_test]
        dataset         = dataset_train
        
        
    # ======================================================================
    # DUTS 
    # ======================================================================
    elif data_name.lower() == 'duts':
        class DutsDataset(Dataset):
            def __init__(self, work, transform_RGB=None, transform_GRAY=None):
                super(DutsDataset, self).__init__()
                            
                self.pairs = []
                
                path = join(data_path, data_name, work)
                
                if 'TR' in work:
                    folder_img_path = join(path, 'DUTS-TR-Image')
                    folder_saliency_path = join(path, 'DUTS-TR-Mask')
                elif 'TE' in work:
                    folder_img_path = join(path, 'DUTS-TE-Image')
                    folder_saliency_path = join(path, 'DUTS-TE-Mask')
                
                images_a = sorted(get_image_files(folder_img_path).values(), key=natural_sort_key)
                images_b = sorted(get_image_files(folder_saliency_path).values(), key=natural_sort_key)
                    
                for img_a, img_b in zip(images_a, images_b):
                    self.pairs.append((img_a, img_b))
                
                self.transform_RGB = transform_RGB
                self.transform_GRAY = transform_GRAY
                
            def __getitem__(self, index):
                img_a_path, img_b_path = self.pairs[index]
                
                img_a = Image.open(img_a_path).convert("RGB")
                img_b = Image.open(img_b_path).convert("L")

                if self.transform_RGB:
                    img_a = self.transform_RGB(img_a)
                    img_b = self.transform_GRAY(img_b)

                return img_a, img_b

            def __len__(self):
                return len(self.pairs)
            
        dataset_train   = DutsDataset('DUTS-TR', transform_RGB, transform_GRAY)
        dataset_test    = DutsDataset('DUTS-TE', transform_RGB, transform_GRAY)
        
        # dataset         = torch.utils.data.ConcatDataset([dataset_train, dataset_test]) 
        # dataset         = [dataset_train, dataset_test]
        dataset         = dataset_train
        
    elif data_name.lower() == 'oxford-flower':
        class flowerDataset(Dataset):
            def __init__(self, transform=None):
                super(flowerDataset, self).__init__()
                            
                path = join(data_path, data_name, 'train')
                self.images = sorted(get_image_files(path).values(), key=natural_sort_key)
                    
                self.transform = transform
                
            def __getitem__(self, index):
                img_path = self.images[index]
                img = Image.open(img_path).convert("RGB")
                img = self.transform(img)

                return img

            def __len__(self):
                return len(self.images)
        
        dataset_train = flowerDataset(transform_RGB)
        dataset = dataset_train

    # ======================================================================
    # celeba 
    # ======================================================================
    elif data_name.lower() == 'celeba':
        transform = torchvision.transforms.Compose([ 
            torchvision.transforms.CenterCrop([160, 160]),
            transform
            ])
        if data_set.lower() == 'train':
            dataset = torchvision.datasets.CelebA(data_path, transform=transform, split='train', download=True)
            if num_data > 0:
                subset  = list(range(0, num_data))
                dataset = torch.utils.data.Subset(dataset, subset)
        elif data_set.lower() == 'test':
            dataset = torchvision.datasets.CelebA(data_path, transform=transform, split='test', download=True)
        elif data_set.lower() == 'all':
            dataset_train   = torchvision.datasets.CelebA(data_path, transform=transform, split='train', download=True)
            dataset_test    = torchvision.datasets.CelebA(data_path, transform=transform, split='test', download=True)
            dataset         = torch.utils.data.ConcatDataset([dataset_train, dataset_test]) 
    # ======================================================================
    # mnist 
    # ======================================================================
    elif data_name.lower() == 'mnist':
        if data_set.lower() == 'train':
            dataset = torchvision.datasets.MNIST(data_path, transform=transform, train=True, download=True)
        elif data_set.lower() == 'test':
            dataset = torchvision.datasets.MNIST(data_path, transform=transform, train=False, download=True)
        elif data_set.lower() == 'all':
            dataset_train   = torchvision.datasets.MNIST(data_path, transform=transform, train=True, download=True)
            dataset_test    = torchvision.datasets.MNIST(data_path, transform=transform, train=False, download=True)
            dataset         = torch.utils.data.ConcatDataset([dataset_train, dataset_test]) 

        if data_subset_use == True:
            idx_label = torch.zeros_like(dataset.targets).bool()
            for label in data_subset_label:
                idx_label = torch.logical_or(idx_label, dataset.targets == label)
            dataset.data    = dataset.data[idx_label]
            dataset.targets = dataset.targets[idx_label]
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
            
            
    if data_subset:
        dataset = Subset(dataset, list(range(0,data_subset_num)))

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
