import torch
import numpy as np
from PIL import Image
import os

import torch.nn as nn
from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils.datautils import normalize01


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        
        if image_tensor.dim() != 3:
            nrow            = int(np.ceil(np.sqrt(image_tensor.shape[0])))
            image_tensor    = make_grid(image_tensor, nrow=nrow, normalize=True)
        
        # image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
            
        image_numpy  = image_tensor.float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy.astype(imtype)
    

    img = Image.fromarray(image_numpy)
    return img


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def make_multi_grid(grid_list, nrow, ncol):
    if len(grid_list[0].shape) != 3:
        nrow_    = int(np.ceil(np.sqrt(grid_list[0].shape[0])))
        for i in range(len(grid_list)):
            grid_list[i]    = normalize01(grid_list[i])
            grid_list[i]    = make_grid(grid_list[i], nrow=nrow_, normalize=True)
    
    width = grid_list[0].shape[1]
    height = grid_list[0].shape[2]
    padd = 10
    k = 0
    back = torch.ones(grid_list[0].shape[0], width * nrow + padd * (nrow-1), height * ncol + padd * (ncol-1))

    for i in range(0,nrow):
        for j in range(0,ncol):
            back[:,width*i+padd*i:width*(i+1)+padd*i,height*j+padd*j:height*(j+1)+padd*j] = grid_list[k]
            k += 1
            
    return back