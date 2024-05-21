import torch
from torch import nn
import numpy as np
import torchvision.models as models

class Discriminator(nn.Module):
    def __init__(self, in_channels: int, dim_features: int):
        super(Discriminator, self).__init__()
        self.conv1      = nn.Conv2d(in_channels=in_channels, out_channels=dim_features*1, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2      = nn.Conv2d(in_channels=dim_features*1, out_channels=dim_features*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3      = nn.Conv2d(in_channels=dim_features*2, out_channels=dim_features*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4      = nn.Conv2d(in_channels=dim_features*4, out_channels=dim_features*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv5      = nn.Conv2d(in_channels=dim_features*8, out_channels=dim_features*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1        = nn.BatchNorm2d(dim_features*1) 
        self.bn2        = nn.BatchNorm2d(dim_features*2) 
        self.bn3        = nn.BatchNorm2d(dim_features*4) 
        self.bn4        = nn.BatchNorm2d(dim_features*8) 
        self.linear1    = nn.Linear(in_features=dim_features*16, out_features=dim_features*8, bias=False) 
        self.linear2    = nn.Linear(in_features=dim_features*8, out_features=1, bias=False) 
        self.activate   = nn.LeakyReLU(inplace=True)
        self.flatten    = nn.Flatten()
             
    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.activate(x)
        
        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.activate(x)
        
        x = self.conv3(x)
        #x = self.bn3(x)
        x = self.activate(x)
        
        x = self.conv4(x)
        #x = self.bn4(x)
        x = self.activate(x)
        
        x = self.conv5(x)
        x = self.activate(x)
       
        x = self.flatten(x)
         
        x = self.linear1(x)
        x = self.activate(x)
        
        x = self.linear2(x)
        x = torch.squeeze(x, dim=-1)
        return x
    
    
class Generator(nn.Module):
    def __init__(self, dim_latent: int, dim_features: int, out_channels: int):
        super(Generator, self).__init__()
        self.linear     = nn.Linear(in_features=dim_latent, out_features=dim_features*16, bias=False) 
        self.unflatten  = nn.Unflatten(1, (dim_features*16, 1, 1))
        self.upsample   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.activate   = nn.LeakyReLU(inplace=True)
        self.output     = nn.Sigmoid()
        
        self.conv1      = nn.Conv2d(in_channels=dim_features*16, out_channels=dim_features*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2      = nn.Conv2d(in_channels=dim_features*8, out_channels=dim_features*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3      = nn.Conv2d(in_channels=dim_features*4, out_channels=dim_features*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4      = nn.Conv2d(in_channels=dim_features*2, out_channels=dim_features*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5      = nn.Conv2d(in_channels=dim_features*1, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1        = nn.BatchNorm2d(dim_features*8) 
        self.bn2        = nn.BatchNorm2d(dim_features*4) 
        self.bn3        = nn.BatchNorm2d(dim_features*2) 
        self.bn4        = nn.BatchNorm2d(dim_features*1) 
        
    def forward(self, x):
        
        x = self.linear(x)
        x = self.unflatten(x)
        x = self.upsample(x)
        
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.activate(x)
        x = self.upsample(x)
        
        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.activate(x)
        x = self.upsample(x)
         
        x = self.conv3(x)
        #x = self.bn3(x)
        x = self.activate(x)
        x = self.upsample(x)
        
        x = self.conv4(x)
        #x = self.bn4(x)
        x = self.activate(x)
        x = self.upsample(x)
        
        x = self.conv5(x)
        x = self.output(x) 
        return x


if __name__ == '__main__':
    dim_latent      = 100
    dim_features    = 32
    in_channels     = 1
    out_channels    = 1
    dim_height      = 32
    dim_width       = 32
     
    G = Generator(dim_latent, dim_features, out_channels) 
    D = Discriminator(in_channels, dim_features) 

    z = torch.randn(100, dim_latent)
    x = torch.randn(100, in_channels, dim_height, dim_width)

    y = G(z)
    h = D(y)
    
    print(y.shape)
    print(h.shape)