from diffusers import UNet2DModel

def MyModel(dim_channel: int, dim_height: int, dim_width: int, num_attention: int=1):
    block_out_channels = (128, 128, 256, 256, 512, 512)
    
    if num_attention == 1:
        down_block_types    = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
        up_block_types      = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    elif num_attention == 2:
        down_block_types    = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D")
        up_block_types      = ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    elif num_attention == 3:
        down_block_types    = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D")
        up_block_types      = ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    elif num_attention == 4:
        down_block_types    = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D")
        up_block_types      = ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    elif num_attention == 5:
        down_block_types    = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
        up_block_types      = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    else:
        raise NotImplementedError('not implemented')

    model = UNet2DModel(
        sample_size         = dim_height,
        in_channels         = dim_channel,
        out_channels        = dim_channel,
        layers_per_block    = 2,
        block_out_channels  = block_out_channels,
        down_block_types    = down_block_types,
        up_block_types      = up_block_types,
    )
    return model