import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('..')

from .ResNet import ResNet_models as resnet

def Model(work: str, args: dict):
    if args.architecture == 'ResNet':

        if args.method == 'from_latent':
            if work == 'generator':
                model   = resnet.GeneratorLatent(channel=args.channel_reduced_gen, latent_dim=args.latent_dim, device=args.cuda_device)
            
            elif work == 'descriptor':
                model   = resnet.Descriptor(channel=args.channel_reduced_des)
                
        elif args.method == 'from_image':
            if work == 'generator':
                model   = resnet.GeneratorBaseLine(channel=args.channel_reduced_gen, latent_dim=args.latent_dim, device=args.cuda_device)
            
            elif work == 'descriptor':
                model   = resnet.Descriptor(channel=args.channel_reduced_des)
        
    else:
        raise NotImplementedError('model selection error') 
    return model
 

if __name__ == '__main__':
    cuda_device     = 0
    model_name      = 'unet1'
    data_channel    = 3
    data_height     = 256
    data_width      = 256
    device          = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'mps')
    
    model           = Model(model_name, data_channel, data_height, data_width)
    model           = model.to(device)

    total_params = sum(param.numel() for param in model.parameters())
    print('total number of parameters:', total_params, flush=True)

