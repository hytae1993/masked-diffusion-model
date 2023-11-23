import torch
from torch import nn
import numpy as np
import random

# ======================================================================
#  mask class
# ======================================================================
class Mask(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args   = args
        
        self.batch_size         = args.batch_size
        self.height             = args.data_size
        self.width              = args.data_size
        
    
    def get_mask(self, black_area_num):
        """
        Generate random masks with black areas for each channel in the batch.

        Parameters:
        - black_area_ratios: List of ratios for the mask area to be black for each channel.

        Returns:
        - masks: Binary masks with black areas, shape (batch_size, 1, height, width).
        """
        masks = torch.ones((len(black_area_num), 1, self.height, self.width), dtype=torch.float32)

        for i in range(len(black_area_num)):
            num_black_pixels = black_area_num[i].int()

            black_pixels = random.sample(range(self.height * self.width), num_black_pixels)

            black_pixels = [(idx // self.width, idx % self.width) for idx in black_pixels]

            for j, k in black_pixels:
                masks[i, 0, j, k] = 0.0

        return masks
    
    
    def get_list_black_area_ratios(self, time):
        """
        Generate a scheduled time list. Each time means the ratio to be removed from original input image.

        Parameters:
        - time: Max time fo ddpm steps
        
        Returns:
        - black_area_ratio: scheduled time list.
        """
        
        if self.args.ddpm_schedule == 'linear':
            # black_area_ratio    = list(map(int, timesteps / (self.args.ddpm_num_steps+1)))
            ratio    = time / self.args.updated_ddpm_num_steps
            
        elif self.args.ddpm_schedule == 'log_scale':
            base     = 1.1
            ratio    = np.power(base, time.cpu())
            ratio    = ratio / np.power(base, self.args.updated_ddpm_num_steps)
        else:
            raise ValueError("Invalid mask ratio scheduler")
        
        black_area_ratio    = self.height * self.width * ratio
        black_area_ratio    = np.ceil(black_area_ratio)
        
        return black_area_ratio