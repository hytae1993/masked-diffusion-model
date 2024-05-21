import numpy as np
import torch.nn as nn

# =============================================================================================
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/ebgan/ebgan.py
# =============================================================================================

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(62, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(2, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = 32 // 2
        down_dim = 64 * (32 // 2) ** 2
        down_dim = 12544
        
        self.embedding = nn.Linear(down_dim, 32)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, 1, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        embedding = self.embedding(out.view(out.size(0), -1))
        out = self.fc(embedding)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out, embedding


class AutoEncoder(nn.Module):
    def __init__(self, z_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, z_dim),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64 * 7 * 7),
            nn.Unflatten(1, (64, 7, 7)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1, output_padding=1, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1, output_padding=1, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        print(x.shape)
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        return x