from __future__ import annotations
import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import timm
from mamba_ssm import Mamba
from torch.fft import fft, ifft

class DWConv(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, bias=True, groups=dim)

    def forward(self, x, H=None, W=None):
        if x.ndim == 4:  
            x = self.dwconv(x)
        elif x.ndim == 3: 
            B, N, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.dwconv(x)
            x = x.flatten(2).transpose(1, 2)
        else:
            raise ValueError(f"Unexpected input shape {x.shape}, expected 3D or 4D tensor.")
        return x

class MSFIBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dwconv3 = DWConv(dim, kernel_size=3)  
        self.dwconv5 = DWConv(dim, kernel_size=5) 
        self.dwconv7 = DWConv(dim, kernel_size=7)  
        self.ln1 = nn.LayerNorm(dim)  
        self.relu = nn.ReLU(inplace=True)  
        self.ln2 = nn.LayerNorm(dim) 
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0) 
        self.sigmoid = nn.Sigmoid()  
        self.channel_attn = ChannelAttention(dim)

    def forward(self, x, H, W):
        x_skip = x 

        x = self.dwconv3(x, H, W)
        x = self.channel_attn(x)
        
        x = x.transpose(1, -1)  
        x = self.ln1(x)
        x = x.transpose(1, -1) 
        x = x.view(x.shape[0], self.dim, H, W) 
        x = self.proj(x) 
        x = self.relu(x)

        x_fft = torch.fft.fft(x, dim=1)
        x_fft_real, x_fft_imag = x_fft.real, x_fft.imag

        x_fft_real = x_fft_real.transpose(1, -1)
        x_fft_real = self.ln1(x_fft_real)
        x_fft_real = x_fft_real.transpose(1, -1)
        x_fft_real = self.relu(x_fft_real)

        x_fft_imag = x_fft_imag.transpose(1, -1)
        x_fft_imag = self.ln1(x_fft_imag)
        x_fft_imag = x_fft_imag.transpose(1, -1)
        x_fft_imag = self.relu(x_fft_imag)

        x_fft_real_3 = self.dwconv3(x_fft_real, H, W)
        x_fft_real_5 = self.dwconv5(x_fft_real, H, W)
        x_fft_real_7 = self.dwconv7(x_fft_real, H, W)
        x_fft_real = x_fft_real_3 + x_fft_real_5 + x_fft_real_7

        x_fft_imag_3 = self.dwconv3(x_fft_imag, H, W)
        x_fft_imag_5 = self.dwconv5(x_fft_imag, H, W)
        x_fft_imag_7 = self.dwconv7(x_fft_imag, H, W)
        x_fft_imag = x_fft_imag_3 + x_fft_imag_5 + x_fft_imag_7

        
        x_fft_out = torch.complex(x_fft_real, x_fft_imag)
        x_ifft = torch.fft.ifft(x_fft_out, dim=1).real

        x_ifft = x_ifft.transpose(1, -1)
        x_ifft = self.ln2(x_ifft)
        x_ifft = x_ifft.transpose(1, -1)
        x_ifft = self.relu(x_ifft)

        x_ifft = x_ifft.view(x_ifft.shape[0], self.dim, H, W)  
        x_ifft = self.proj(x_ifft) 
        #x = x_skip + x_ifft
        x_ifft = self.sigmoid(x_ifft)  

        x = x_skip * x_ifft
        
        return x


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y

class MultiHeadChannelAttentionDW(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0,        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  
        
        self.query = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)  
        self.key = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.value = nn.Conv2d(dim, dim, kernel_size=1, padding=0,bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        B, C, H, W = x.shape  # [B, C, H, W]
        
        Q = self.query(x)  # [B, C, H, W]
        K = self.key(x)    # [B, C, H, W]
        V = self.value(x)  # [B, C, H, W]

        Q = Q.view(B, C, -1)
        K = K.view(B, C, -1)
        V = V.view(B, C, -1)

        Q = Q.view(B, self.num_heads, self.head_dim, -1)
        K = K.view(B, self.num_heads, self.head_dim, -1)
        V = V.view(B, self.num_heads, self.head_dim, -1)

        attn = torch.matmul(Q.transpose(2, 3), K) / (self.head_dim ** 0.5)  # [B, num_heads, H*W, H*W]
        attn = self.softmax(attn)

        out = torch.matmul(attn, V.transpose(2, 3))  # [B, num_heads, H*W, head_dim]
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)  #[B, C, H, W]
        out = out.view(B, C, H, W)
        out = self.proj(out)

        out = self.gamma * out + x  
        return out


class FMABlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, drop=0., num_slices=None, act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        
        self.ln1 = nn.LayerNorm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        
        self.proj0 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_ifft = nn.Conv2d(dim, dim, 1, 1, 0)
        
        self.channel_attn = MultiHeadChannelAttentionDW(dim)
        
        # Depthwise 3x3 convolution branch
        self.dwconv3 = DWConv(dim, kernel_size=3)
        
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=num_slices,
        )
    
    def forward(self, x, H, W):
        B, C = x.shape[0], x.shape[1]
        x_skip = x
        
        # LayerNorm + ReLU
        x = x.transpose(1, -1)
        x = self.ln1(x)
        x = x.transpose(1, -1)
        x = self.relu(x)
        
        x_fft = fft(x, dim=1)
        x_fft_real, x_fft_imag = x_fft.real, x_fft.imag
        
        x_fft_real_attn = self.channel_attn(x_fft_real)
        x_fft_imag_attn = self.channel_attn(x_fft_imag)
        
        x_fft_real_dw = self.dwconv3(x_fft_real)
        x_fft_imag_dw = self.dwconv3(x_fft_imag)
        
        #x_fft_real = x_fft_real_attn + x_fft_real_dw
        #x_fft_imag = x_fft_imag_attn + x_fft_imag_dw
        x_fft_real = x_fft_real_attn + x_fft_real_dw
        x_fft_imag = x_fft_imag_attn + x_fft_imag_dw
        
        # IFFT
        x_fft_out = torch.complex(x_fft_real, x_fft_imag)
        x_ifft = ifft(x_fft_out, dim=1).real
        x_ifft = x_ifft.transpose(1, -1)
        x_ifft = self.ln3(x_ifft)
        x_ifft = x_ifft.transpose(1, -1)
        x_ifft = self.relu(x_ifft)
        x_ifft = x_ifft.view(B, C, H, W)
        x_ifft = self.proj_ifft(x_ifft)
        
        x_mamba = x.reshape(B, C, H * W).transpose(1, 2)
        x_mamba = self.mamba(x_mamba)
        x_mamba = x_mamba.transpose(1, 2).reshape(B, C, H, W)
        
        x_mamba = x_mamba.transpose(1, -1)
        x_mamba = self.ln3(x_mamba)
        x_mamba = x_mamba.transpose(1, -1)
        x_mamba = self.relu(x_mamba)
        x_mamba = x_mamba.view(B, C, H, W)
        x_mamba = self.proj0(x_mamba)
        
        x = x_skip + x_ifft + x_mamba
      
        return x

class HMConv(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()
       
        self.proj = nn.Conv2d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv2d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv2d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm2d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv2d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm2d(in_channles)
        self.nonliner4 = nn.ReLU() 
        
        self.nonliner5 = nn.SiLU()
        self.fc1 = nn.Linear(in_channles, in_channles) 
        self.act = nn.GELU()
        self.dwconv = DWConv(in_channles) 
           

    def forward(self, x):

        x_residual = x 
        x_act = self.act(x)
        B, C, H, W = x.shape

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)
     
        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x2 = self.proj3(x2)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)
        
        x = x1 + x2
        
        
        x = self.norm(x)
        x = self.nonliner(x)
        x = self.proj(x)
        
        x = self.norm4(x)
        x = self.nonliner4(x)
        x = self.proj4(x)
        return x + x_act

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        # Downsample layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        for i in range(4):
            gsc = HMConv(dims[i])
            if i < 2:
               
                stage = nn.Sequential(
                    *[MSFIBlock(dim=dims[i]) for _ in range(depths[i])]
                )
            else:
                
                stage = nn.Sequential(
                    *[FMABlock(dim=dims[i], num_slices=num_slices_list[i]) for _ in range(depths[i])]
                )
            self.stages.append(stage)
            self.gscs.append(gsc)

        self.out_indices = out_indices

    def forward_features(self, x):
        outs = []
        for i in range(4):
            # Downsample layer
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            B, C, H, W = x.shape  

            for block in self.stages[i]:
                x = block(x, H, W)  

            # Append output for specified indices
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)


class UNetUpBlockWithInterpolation(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UNetUpBlockWithInterpolation, self).__init__()
        
        self.reduce_channels = nn.Conv2d(in_channels + skip_channels, in_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x, skip), dim=1)
        x = self.reduce_channels(x)  
        x = self.conv(x)
        return x
        
class FDImamba(nn.Module):
    def __init__(self, 
                 in_chans=3,
                 out_chans=1,
                 depths=[2, 2, 2, 2],
                 feat_size=[48, 96, 192, 384],
                 hidden_size=768,
                 norm_name="instance",
                 res_block=True,
                 spatial_dims=2) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.vit = MambaEncoder(
            in_chans=in_chans, 
            depths=depths,
            dims=feat_size,
            drop_path_rate=0,
            layer_scale_init_value=1e-6
        )
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
  
        
        self.decoder5 = UNetUpBlockWithInterpolation(hidden_size, feat_size[3], feat_size[3])
        self.decoder4 = UNetUpBlockWithInterpolation(feat_size[3], feat_size[2], feat_size[2])
        self.decoder3 = UNetUpBlockWithInterpolation(feat_size[2], feat_size[1], feat_size[1])
        self.decoder2 = UNetUpBlockWithInterpolation(feat_size[1], feat_size[0], feat_size[0])

       
        self.decoder_blocks = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        for i in range(4):
            if i > 2:  
                stage = nn.Sequential(
                    *[FMABlock(dim=feat_size[i], num_slices=num_slices_list[i]) for _ in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                    *[MSFIBlock(dim=feat_size[i]) for _ in range(depths[i])]
                )
            self.decoder_blocks.append(stage)

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=out_chans)

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])

        dec3 = self.decoder5(enc_hidden, enc4)
        dec3 = self.apply_decoder_blocks(dec3, 3)          
        dec2 = self.decoder4(dec3, enc3)
        dec2 = self.apply_decoder_blocks(dec2, 2)  

        dec1 = self.decoder3(dec2, enc2)
        dec1 = self.apply_decoder_blocks(dec1, 1) 

        dec0 = self.decoder2(dec1, enc1)
        dec0 = self.apply_decoder_blocks(dec0, 0)  
        out = self.out(dec0)

        return out

    def apply_decoder_blocks(self, x, stage_idx):
        for block in self.decoder_blocks[stage_idx]:
            B, C, H, W = x.shape
            x = block(x, H, W)  
        return x
