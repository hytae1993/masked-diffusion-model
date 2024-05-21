import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('..')


from .unet import unet1 as model1
from .unet import unet2 as model2
from .unet import unet3 as model3
from .unet import unet4 as model4
from .unet import unet5 as model5
from .unet import unet6 as model6

def Model(name: str, data_channel: int, data_height: int, data_width: int, out_channel: int):
    if name == 'unet1':    # unet from unet1.py  
        model_c_in              = data_channel
        model_c_out             = data_channel 
        model_time_dim          = 256
        model_remove_deep_conv  = False

        model = model1.UNet(
            model_c_in,
            model_c_out,
            model_time_dim,
            model_remove_deep_conv)
        
    elif name == 'unet2':  # unet from unet2.py  
        model_image_channels    = data_channel 
        model_n_channels        = 64
        model_ch_mults          = (1, 2, 2, 4)
        model_is_attn           = (False, False, True, True)
        model_n_blocks          = 2

        model = model2.UNet(
            model_image_channels, 
            model_n_channels, 
            model_ch_mults, 
            model_is_attn, 
            model_n_blocks)
    
    elif name == 'unet3':  # unet from unet3.py  
        model_dim           = data_height
        model_init_dim      = None
        model_out_dim       = None 
        model_dim_mults     = (1, 2, 4, 8)
        model_channels      = data_channel
        with_time_emb       = True
        resnet_block_groups = 8
        use_convnext        = True
        convnext_mult       = 2
        
        model = model3.UNet(
            model_dim,
            model_init_dim,
            model_out_dim,
            model_dim_mults,
            model_channels,
            with_time_emb,
            resnet_block_groups,
            use_convnext,
            convnext_mult)
            
    elif name == 'unet4':  # unet from unet4.py  
        image_size              = data_height
        in_channels             = data_channel
        model_channels          = 128
        out_channels            = data_channel
        num_res_blocks          = 2
        attention_resolutions   = 16,8
        dropout                 = 0
        channel_mult            = (1, 2, 4, 8)
        conv_resample           = True
        dims                    = 2
        num_classes             = None
        use_checkpoint          = False
        use_fp16                = False
        num_heads               = 1
        num_head_channels       = -1
        num_heads_upsample      = -1
        use_scale_shift_norm    = False
        resblock_updown         = False
        use_new_attention_order = False
    
        model = model4.UNetModel(
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            num_classes,
            use_checkpoint,
            use_fp16,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            resblock_updown,
            use_new_attention_order)
    
    elif name == 'unet5':  # unet from unet5.py  
        in_channel              = data_channel
        out_channel             = data_channel
        inner_channel           = 32
        norm_groups             = 32
        channel_mults           = (1, 2, 4, 8, 8)
        attn_res                = [8]
        res_blocks              = 3
        dropout                 = 0
        with_noise_level_emb    = True
        image_size              = data_height

        model = model5.UNet(
            in_channel,
            out_channel,
            inner_channel,
            norm_groups,
            channel_mults,
            attn_res,
            res_blocks,
            dropout,
            with_noise_level_emb,
            image_size)
    
    elif name == 'unet6':  # unet from unet6.py
        image_size          = data_height
        in_channels         = data_channel
        hid_channels        = 128
        out_channels        = out_channel
        num_res_blocks      = 2
        time_embedding_dim  = None
        drop_rate           = 0.0
        resample_with_conv  = True

        if image_size == 32:
            ch_multipliers  = [1, 2, 2, 2]
            # ch_multipliers  = [1, 2, 4, 8]
            # apply_attn      = [False, True, False, False]
            # apply_attn      = [True, True, True, True]
            apply_attn      = [False, False, True, False]
            
        elif image_size == 64:
            ch_multipliers  = [1, 2, 2, 2]
            apply_attn      = [False, False, True, False]
            
        elif image_size == 128:
            ch_multipliers  = [1, 1, 2, 2, 4, 4]
            apply_attn      = [False, False, False, False, True, False]
            
        elif image_size == 256:
            ch_multipliers  = [1, 1, 2, 2, 4, 4]
            apply_attn      = [False, False, False, False, True, False]
            
        
        model = model6.UNet(
            in_channels=in_channels,
            hid_channels=hid_channels,
            out_channels=out_channel,
            ch_multipliers=ch_multipliers,
            num_res_blocks=num_res_blocks,
            apply_attn=apply_attn,
            time_embedding_dim=time_embedding_dim,
            drop_rate=drop_rate,
            resample_with_conv=resample_with_conv)
    
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

