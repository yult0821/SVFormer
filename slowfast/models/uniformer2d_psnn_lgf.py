# -*- coding: utf-8 -*-
from math import ceil, sqrt
from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from functools import partial
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from .build import MODEL_REGISTRY
import os
import pickle as pkl

try:
    from spikingjelly.activation_based import layer, neuron, surrogate  # spikingjelly13, spikingjelly14
except:
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode  # spikingjelly12

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

model_path = '/userhome/transformer/UniFormer/video_classification/exp/downloaded_ckpts' #'path_to_models'
model_path = {
    'uniformer_small_in1k': os.path.join(model_path, 'uniformer_small_in1k.pth'),
    'uniformer_small_k400_8x8': os.path.join(model_path, 'uniformer_small_k400_8x8.pth'),
    'uniformer_small_k400_16x4': os.path.join(model_path, 'uniformer_small_k400_16x4.pth'),
    'uniformer_small_k600_16x4': os.path.join(model_path, 'uniformer_small_k600_16x4.pth'),
    'uniformer_base_in1k': os.path.join(model_path, 'uniformer_base_in1k.pth'),
    'uniformer_base_k400_8x8': os.path.join(model_path, 'uniformer_base_k400_8x8.pth'),
    'uniformer_base_k400_16x4': os.path.join(model_path, 'uniformer_base_k400_16x4.pth'),
    'uniformer_base_k600_16x4': os.path.join(model_path, 'uniformer_base_k600_16x4.pth'),
    'uniformer_small_sthv2_16_prek600': os.path.join(model_path, 'uniformer_small_sthv2_16_prek600.pth'),
    'uniformer2d_psnn_tri8_xs16a_ucf101_best': os.path.join(model_path, 'uniformer2d_psnn_tri8_xs16a_ucf101_best.pyth'),
    'uniformer2d_psnn_tri22_xs16a_ucf101_best': os.path.join(model_path, 'uniformer2d_psnn_tri22_xs16a_ucf101_best.pyth'),
    'uniformer2d_psnn_tri28_xs16a_ucf101_best': os.path.join(model_path, 'uniformer2d_psnn_tri28_xs16a_ucf101_best.pyth'),
    'uniformer2d_psnn_tri40_lgf_ucf101_best': os.path.join(model_path, 'uniformer2d_psnn_tri40_lgf_ucf101_best.pyth'),
}


class BatchNorm1d_eT(nn.BatchNorm1d):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True
    ):
        """
        将T与C维度合并, 再利用nn.BatchNorm1d操作, 相当于每个时间步单独计算均值和方差。输入应为[T, B, C, N], 多步模式下应用。单步模式用nn.BatchNorm1d即可
        Refer to :class:`torch.nn.BatchNorm1d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def extra_repr(self):
        return super().extra_repr()

    def forward(self, x: Tensor):
        if x.dim() != 4:
            raise ValueError(f'expected x with shape [T, B, C, N], but got x with shape {x.shape}!')
        T, B, C, N = x.shape
        x = x.permute(1, 0, 2, 3).contiguous()
        x = x.reshape(B, T*C, N).contiguous()
        x = super().forward(x)
        x = x.reshape(B, T, C, N).contiguous()
        x = x.permute(1, 0, 2, 3).contiguous()
        return x


class BatchNorm2d_eT(nn.BatchNorm2d):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True
    ):
        """
        将T与C维度合并, 再利用nn.BatchNorm2d操作, 相当于每个时间步单独计算均值和方差。输入应为[T, B, C, H, W], 多步模式下应用。单步模式用nn.BatchNorm2d即可
        Refer to :class:`torch.nn.BatchNorm2d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def extra_repr(self):
        return super().extra_repr()

    def forward(self, x: Tensor):
        if x.dim() != 5:
            raise ValueError(f'expected x with shape [T, B, C, H, W], but got x with shape {x.shape}!')
        T, B, C, H, W = x.shape
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        x = x.reshape(B, T*C, H, W).contiguous()
        x = super().forward(x)
        x = x.reshape(B, T, C, H, W).contiguous()
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        return x


# def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
#     return layer.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups, step_mode='m')

# def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
#     return layer.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups, step_mode='m')

# def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
#     return layer.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups, step_mode='m')


def conv_nxn(inp, oup, kernel_size=3, stride=3, groups=1):  # 0904, to use overlapping conv2d in patch embedding
    return layer.Conv2d(inp, oup, (kernel_size, kernel_size), (stride, stride), (1, 1), groups=groups, step_mode='m')

# def conv_nxn(inp, oup, kernel_size=3, stride=3, groups=1):
#     return layer.Conv2d(inp, oup, (kernel_size, kernel_size), (stride, stride), (0, 0), groups=groups, step_mode='m')

def conv_1x1(inp, oup, groups=1):
    return layer.Conv2d(inp, oup, (1, 1), (1, 1), (0, 0), groups=groups, step_mode='m')

def conv_3x3(inp, oup, groups=1):  # postiton encoding
    return layer.Conv2d(inp, oup, (3, 3), (1, 1), (1, 1), groups=groups, step_mode='m')

def conv_5x5(inp, oup, groups=1):
    return layer.Conv2d(inp, oup, (5, 5), (1, 1), (2, 2), groups=groups, step_mode='m')

def bn_2d(dim):
    return BatchNorm2d_eT(dim) # 将T与C维度合并, 再利用nn.BatchNorm2d操作, 相当于每个时间步单独计算均值和方差
    # return layer.BatchNorm2d(dim, step_mode='m')

def bn_1d(dim):
    return BatchNorm1d_eT(dim) # 将T与C维度合并, 再利用nn.BatchNorm1d操作, 相当于每个时间步单独计算均值和方差
    # return layer.BatchNorm1d(dim, step_mode='m')

def act_layer(): # try different neuron models
#     return neuron.KLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', backend='torch') #surrogate.Sigmoid()
    #return neuron.GatedLIFNode(T=16, surrogate_function=surrogate.Sigmoid(), step_mode='m', backend='torch') #modify line 2296 in modiUniFormer/video_classification/spikingjelly/spikingjelly/activation_based/neuron.py (failed!)
    return neuron.ParametricLIFNode(v_threshold=0.5, surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', backend='cupy')
    #return neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', backend='cupy') # backend='cupy' 'torch'

# 1102 try all BNT
TN = 16 #4 #16  # cfg.UNIFORMER.T   cfg.DATA.NUM_FRAMES

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=act_layer, drop=0.):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act1 = act_layer()
        self.fc1 = layer.Linear(in_features, hidden_features)
        # self.norm1 = norm_layer(hidden_features)  # 0901 added # [T B N C]
        self.norm1 = bn_1d(hidden_features * TN) # 1023 try all BN
#         self.drop1 = layer.Dropout(drop)  # SNN, no need to dropout? 0829
        self.act2 = act_layer()
        self.fc2 = layer.Linear(hidden_features, out_features)
        # self.norm2 = norm_layer(out_features)  # 0901 added
        self.norm2 = bn_1d(out_features * TN) # 1023 try all BN
#         self.drop2 = layer.Dropout(drop)  # SNN, no need to dropout? 0829

    def forward(self, x): # [T B N C]
        x = self.act1(x)
        if isinstance(self.act1, neuron.GatedLIFNode):
            x = torch.squeeze(x)
        # calculate firing rate
        # fr = (x.sum() / x.numel()).item()
        #print("MLP act1", fr)
        x = self.fc1(x)
        x = x.transpose(-2, -1) # 1023 try all BN
        x = self.norm1(x)  # 0901 added # 1023 try all BN
        x = x.transpose(-2, -1) # 1023 try all BN
#         x = self.drop1(x)  # SNN, no need to dropout? 0829
        x = self.act2(x)
        if isinstance(self.act2, neuron.GatedLIFNode):
            x = torch.squeeze(x)
        # calculate firing rate
        # fr = (x.sum() / x.numel()).item()
        #print("MLP act2", fr)
        x = self.fc2(x)
        x = x.transpose(-2, -1) # 1023 try all BN
        x = self.norm2(x)  # 0901 added # 1023 try all BN
        x = x.transpose(-2, -1) # 1023 try all BN
#         x = self.drop2(x)  # SNN, no need to dropout? 0829
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        
        self.in_lif = act_layer()
        self.qkv = layer.Linear(dim, dim * 3) # bias=qkv_bias (0818)
        
        # add batchnorm 0818 before input to lif  (0830: try to only use bn to q k v)
        self.norm1 = bn_1d(dim * 3 * TN) # 1023 try all BN
        # self.norm1 = norm_layer(dim * 3)  # 1013 try layernorm in ssa
        
        # self.attn_drop = layer.Dropout(attn_drop)
        self.q_lif = act_layer()
        self.k_lif = act_layer()
        self.v_lif = act_layer()

#         # add batchnorm 0818 before input to lif
#         self.norm2 = bn_1d(dim)
        self.proj_lif = act_layer()
        self.proj = layer.Linear(dim, dim)
        
        # 1013: add layernorm after proj
        # self.norm2 = norm_layer(dim)
        self.norm2 = bn_1d(dim * TN) # 1023 try all BN

#         self.proj_drop = layer.Dropout(proj_drop)  # SNN, no need to dropout? 0829

    def forward(self, x):
        #print(x.shape)
        x = self.in_lif(x)
        #print(x.shape)
        if isinstance(self.in_lif, neuron.GatedLIFNode):
            x = torch.squeeze(x)
        # calculate firing rate
        # fr = (x.sum() / x.numel()).item()
        #print("Attn in_lif", fr)
        T, B, N, C = x.shape

        # add batchnorm 0818 before input to lif  (0830: try to only use bn to q k v)
        qkv_ = self.norm1(self.qkv(x).transpose(-2, -1)).transpose(-2, -1).flatten(0, 1)
        # qkv_ = self.norm1(self.qkv(x)).flatten(0, 1)   # 1013 try layernorm in ssa
        
        qkv = qkv_.reshape(T*B, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode='trunc')).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

#         x = x.flatten(0, 1) # torch.Size([1, 1568, 320])
#         # TB, N, C = x.shape 
#         # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # torch.Size([3, 1, 5, 1568, 64])
#         qkv = self.qkv(x).reshape(T*B, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode='trunc')).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # torch.Size([1, 5, 1568, 64])

        # transfer into spikes
        q_sk = self.q_lif(q.reshape(T, B, self.num_heads, N, torch.div(C, self.num_heads, rounding_mode='trunc')).contiguous())
        k_sk = self.k_lif(k.reshape(T, B, self.num_heads, N, torch.div(C, self.num_heads, rounding_mode='trunc')).contiguous())
        v_sk = self.v_lif(v.reshape(T, B, self.num_heads, N, torch.div(C, self.num_heads, rounding_mode='trunc')).contiguous())
        #print(q_sk.shape)
        
        # # calculate firing rate
        # fr = (q_sk.sum() / q_sk.numel()).item()
        # #print("Attn q_sk", fr)
        # fr = (k_sk.sum() / k_sk.numel()).item()
        # #print("Attn k_sk", fr)
        # fr = (v_sk.sum() / v_sk.numel()).item()
        # #print("Attn v_sk", fr)

        # attn = (q @ k.transpose(-2, -1)) * self.scale # torch.Size([1, 5, 1568, 1568])
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C) # torch.Size([1, 5, 1568, 64]) -> torch.Size([1, 1568, 320])
        
        x = k_sk.transpose(-2,-1) @ v_sk
        x = (q_sk @ x) * self.scale  # (T, B, num_heads, N, C//num_heads)
        #print(x.shape)
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()  # (T, B, N, C)

#         # add batchnorm 0818 before input to lif   (bad effects, 0901)
#         x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)

        #print(x.shape)
        x = self.proj_lif(x)
        if isinstance(self.proj_lif, neuron.GatedLIFNode):
            x = torch.squeeze(x)
        # calculate firing rate
        # fr = (x.sum() / x.numel()).item()
        #print("Attn proj_lif", fr)
        #print(x.shape)
        x = self.proj(x) # torch.Size([1, 1568, 320])
#         x = self.proj_drop(x) # torch.Size([1, 1568, 320])  # SNN, no need to dropout? 0829

        # x = self.norm2(x) # 1013: add layernorm after proj
        x = x.transpose(-2, -1) # 1023 try all BN
        x = self.norm2(x) # 1023 try all BN
        x = x.transpose(-2, -1) # 1023 try all BN

#         # add batchnorm 0818 before input to lif   (no need, due to the layernorm after attention)
#         x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)       

        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=act_layer, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act1 = act_layer()
        self.fc1 = conv_1x1(in_features, hidden_features)
        self.norm1 = bn_2d(hidden_features * TN)  # 0901 added
#         self.drop1 = layer.Dropout(drop)  # SNN, no need to dropout? 0829
        self.act2 = act_layer()
        self.fc2 = conv_1x1(hidden_features, out_features)
        self.norm2 = bn_2d(out_features * TN)  # 0901 added
#         self.drop2 = layer.Dropout(drop)  # SNN, no need to dropout? 0829

    def forward(self, x):
        x = self.act1(x)
        # calculate firing rate
        # fr = (x.sum() / x.numel()).item()
        #print("CMLP act1", fr)
        x = self.fc1(x)
        x = self.norm1(x)  # 0901 added
#         x = self.drop1(x)  # SNN, no need to dropout? 0829
        x = self.act2(x)
        # calculate firing rate
        # fr = (x.sum() / x.numel()).item()
        #print("CMLP act2", fr)
        x = self.fc2(x)
        x = self.norm2(x)  # 0901 added
#         x = self.drop2(x)  # SNN, no need to dropout? 0829
        return x


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=act_layer, norm_layer=nn.LayerNorm):
        super().__init__()

#         # excluding position embedding module 1116
#         self.in_lif = act_layer()
#         self.pos_embed = conv_3x3(dim, dim, groups=dim)
#         self.norm1 = bn_2d(dim * TN)

        self.act1 = act_layer()
        # 0821 - apply less lif layer, exclude act2 and act3
        # 0914 tri10 - try to add these two lif layers based on tri8, to test its effect
        self.conv1 = conv_1x1(dim, dim, 1) # not change size
#         self.act2 = act_layer()
        self.attn = conv_5x5(dim, dim, groups=dim) # not change size
#         self.act3 = act_layer()
        self.conv2 = conv_1x1(dim, dim, 1) # not change size
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_2d(dim * TN)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = self.in_lif(x)
#         x = x + self.pos_embed(self.in_lif(x)) # torch.Size([1, 64, 8, 56, 56]) -> torch.Size([1, 64, 8, 56, 56])

#         # excluding position embedding module 1116
#         x = x + self.norm1(self.pos_embed(self.in_lif(x))) # 1013 adjust bn position 

        # x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x))))) # local MHRA # not change size # should be decomposed to add lif-neuron
        x_pe = x
#         x = self.norm1(x) # 1013 adjust bn position 
        #print('line358 plif input', x.shape)
        x = self.act1(x)
        #print('line358 plif output', x.shape)
        # calculate firing rate
        # fr = (x.sum() / x.numel()).item()
        #print("CBlock act1", fr)
        # 0821 - apply less lif layer, exclude act2 and act3
        # 0914 tri10 - try to add these two lif layers based on tri8, to test its effect
        x = self.conv1(x)
#         x = self.act2(x)
        x = self.attn(x)
#         x = self.act3(x)
        x = self.conv2(x)
        #x = x_pe + self.drop_path(x)  # 1013 adjust bn position 
        x = x_pe + self.norm2(self.drop_path(x))  # 1013 adjust bn position 
        
        # x_attn = x
        # x = self.norm2(x)
        # x = self.mlp(x)
        # x = x_attn + self.drop_path(x) # not change size
        
        # 1013 adjust bn position 
#         x = x + self.drop_path(self.mlp(self.norm2(x))) 
        x = x + self.drop_path(self.mlp(x))
        
        return x   


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=act_layer, norm_layer=nn.LayerNorm):
        super().__init__()

#         # excluding position embedding module 1116
#         self.in_lif = act_layer()
#         self.pos_embed = conv_3x3(dim, dim, groups=dim)

#         # 1013: try batchnorm before ssa
#         # self.norm1 = norm_layer(dim)
#         self.norm1 = bn_2d(dim * TN)

        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop) # global MHRA
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 1013: try layernorm after ssa
        # 1013: adjust this layernorm into ssa
#         self.norm2 = norm_layer(dim)
        #self.norm2 = bn_1d(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
#         # x = self.in_lif(x)
#         x = x + self.pos_embed(self.in_lif(x))
#         T, B, C, H, W = x.shape
#         x = x.flatten(3).transpose(2, 3) # torch.Size([1, 320, 8, 14, 14]) -> torch.Size([1, 1568, 320])
#         x = x + self.drop_path(self.attn(self.norm1(x))) # torch.Size([1, 1568, 320])
#         x = x + self.drop_path(self.mlp(self.norm2(x))) # torch.Size([1, 1568, 320])
#         x = x.transpose(2, 3).reshape(T, B, C, H, W) # torch.Size([1, 320, 8, 14, 14])


#         # excluding position embedding module 1116
#         # 1013: adjust bn/ln position  ( batchnorm before ssa and ln after ssa)
#         x = x + self.norm1(self.pos_embed(self.in_lif(x)))
        T, B, C, H, W = x.shape
        x = x.flatten(3).transpose(2, 3) # (T B N C) N=H*W
        x = x + self.drop_path(self.attn(x)) # (T B N C)
        x = x + self.drop_path(self.mlp(x)) # (T B N C)
        x = x.transpose(2, 3).reshape(T, B, C, H, W)
        
        return x    


class SplitSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=act_layer, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_lif = act_layer()
        self.pos_embed = conv_3x3(dim, dim, groups=dim)
        self.t_norm = norm_layer(dim)
        self.t_attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = self.in_lif(x)
        x = x + self.pos_embed(self.in_lif(x))
        B, C, D, H, W = x.shape
        attn = x.view(B, C, D, H * W).permute(0, 3, 2, 1).contiguous()
        attn = attn.view(B * H * W, D, C)
        attn = attn + self.drop_path(self.t_attn(self.t_norm(attn)))
        attn = attn.view(B, H * W, D, C).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(B * D, H * W, C)
        residual = x.view(B, C, D, H * W).permute(0, 2, 3, 1).contiguous()
        residual = residual.view(B * D, H * W, C)
        attn = residual + self.drop_path(self.attn(self.norm1(attn)))
        attn = attn.view(B, D * H * W, C)
        out = attn + self.drop_path(self.mlp(self.norm2(attn)))
        out = out.transpose(1, 2).reshape(B, C, D, H, W)
        return out


class SpeicalPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size) # 224 -> (224, 224)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # self.norm = nn.LayerNorm(embed_dim)
        # self.in_lif = act_layer()
        self.proj = conv_nxn(in_chans, embed_dim, kernel_size=patch_size[0]+2, stride=patch_size[0])# 0904, to use overlapping conv2d in patch embedding
#         self.proj = conv_nxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        
        # use batchnorm instead 0818
        # self.norm = nn.LayerNorm(embed_dim)
        self.norm = bn_2d(embed_dim * TN)

    def forward(self, x):
        T, B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.in_lif(x) # SpeicalPatchEmbed directly receive external inputs, act as input encoding layer
        x = self.proj(x)
        
        # use batchnorm instead 0818
        x = self.norm(x)
        # T, B, C, H, W = x.shape
        # x = x.flatten(3).transpose(2, 3)
        # x = self.norm(x)
        # x = x.reshape(T, B, H, W, -1).permute(0, 1, 4, 2, 3).contiguous()
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, std=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # self.norm = nn.LayerNorm(embed_dim)
        self.in_lif = act_layer()
#         self.proj = conv_nxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        self.proj = conv_nxn(in_chans, embed_dim, kernel_size=patch_size[0]+2, stride=patch_size[0]) # 0904, to use overlapping conv2d in patch embedding
        # if std:
        #     self.proj = conv_3xnxn_std(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        # else:
        #     self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        
        # use batchnorm instead 0818
        # self.norm = nn.LayerNorm(embed_dim)
        self.norm = bn_2d(embed_dim * TN)

    def forward(self, x):
        T, B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.in_lif(x)
        # calculate firing rate
        # fr = (x.sum() / x.numel()).item()
        #print("PatchEmbed in_lif", fr)
        x = self.proj(x) # torch.Size([1, 64, 8, 56, 56]) -> torch.Size([1, 128, 8, 28, 28])
        
        # use batchnorm instead 0818
        x = self.norm(x)
        # T, B, C, H, W = x.shape # [1, 128, 8, 28, 28]
        # x = x.flatten(3).transpose(2, 3) # torch.Size([1, 64, 8, 56, 56]) -> torch.Size([1, 128, 6272]) -> torch.Size([1, 6272, 128])
        # x = self.norm(x) # torch.Size([1, 6272, 128])
        # x = x.reshape(T, B, H, W, -1).permute(0, 1, 4, 2, 3).contiguous() # torch.Size([1, 8, 28, 28, 128]) -> torch.Size([1, 128, 8, 28, 28])
        return x


@MODEL_REGISTRY.register()
class Uniformer2d_psnn_lgf(nn.Module): # uniformer2d_psnn_try28_allBNT_headV2.py
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, cfg):
        super().__init__()

        depth = cfg.UNIFORMER.DEPTH
        num_classes = cfg.MODEL.NUM_CLASSES 
        img_size = cfg.DATA.TRAIN_CROP_SIZE
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        embed_dim = cfg.UNIFORMER.EMBED_DIM
        head_dim = cfg.UNIFORMER.HEAD_DIM
        mlp_ratio = cfg.UNIFORMER.MLP_RATIO
        qkv_bias = cfg.UNIFORMER.QKV_BIAS
        qk_scale = cfg.UNIFORMER.QKV_SCALE
        representation_size = cfg.UNIFORMER.REPRESENTATION_SIZE
        drop_rate = cfg.UNIFORMER.DROPOUT_RATE
        attn_drop_rate = cfg.UNIFORMER.ATTENTION_DROPOUT_RATE
        drop_path_rate = cfg.UNIFORMER.DROP_DEPTH_RATE
        split = cfg.UNIFORMER.SPLIT
        std = cfg.UNIFORMER.STD # False (spatial-temporal downsample)
        self.use_checkpoint = cfg.MODEL.USE_CHECKPOINT # Checkpointing works by trading compute for memory. (/home/yult/miniconda3/envs/uniformer/lib/python3.8/site-packages/torch/utils/checkpoint.py)
        self.checkpoint_num = cfg.MODEL.CHECKPOINT_NUM
        self.T = cfg.UNIFORMER.T # time steps for uniformer_snn

        logger.info(f'Use checkpoint: {self.use_checkpoint}')
        logger.info(f'Checkpoint number: {self.checkpoint_num}')

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.patch_embed1 = SpeicalPatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        # self.patch_embed2 = PatchEmbed(
        #     img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1], std=std)
        # self.patch_embed3 = PatchEmbed(
        #     img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], std=std)
        # self.patch_embed4 = PatchEmbed(
        #     img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], std=std)

        self.pos_drop = layer.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim] # uniformer_small [1, 2, 5, 8]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1], std=std)
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], std=std)
        if split:
            self.blocks3 = nn.ModuleList([
                SplitSABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
                for i in range(depth[2])])
            self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], std=std)
            self.blocks4 = nn.ModuleList([
                SplitSABlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
            for i in range(depth[3])])
        else:
            self.blocks3 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
                for i in range(depth[2])])
            self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], std=std)
            self.blocks4 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
            for i in range(depth[3])])
        # self.norm = bn_2d(embed_dim[-1])  # no need? 1016
        
        #  local-global fusion: local path 20240116
        # self.lpDWConv1 = layer.Conv2d(embed_dim[1], embed_dim[1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=embed_dim[1], step_mode='m')
        # self.lpDWConv2 = layer.Conv2d(embed_dim[1], embed_dim[1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=embed_dim[1], step_mode='m')
        self.lplif = act_layer() # try add plif in the local-path 2024.2.29
        self.lpDWConv = layer.Conv2d(embed_dim[2], embed_dim[2], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=embed_dim[2], step_mode='m')
        self.lpPWConv = layer.Conv2d(embed_dim[2], embed_dim[-1], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, step_mode='m')
        self.lpBN = bn_2d(embed_dim[-1] * TN)
        
        # # V1: try to use the temporal info in the classification head, not just average them 1031
        # self.dwconv_bf_head = layer.Conv2d(embed_dim[-1], embed_dim[-1], (img_size // 32, img_size // 32), groups=embed_dim[-1], step_mode='m')
        # self.dwconv_bn = bn_2d(embed_dim[-1])
        # # self.pre_logits = nn.Identity()
        # self.head = layer.Linear(embed_dim[-1]*cfg.DATA.NUM_FRAMES, num_classes) if num_classes > 0 else nn.Identity()
        
        # V2: try to use the temporal info in the classification head, not just average them 1031
        # backbone_outdim = embed_dim[-1] # only main path
        feature_outdim = embed_dim[-1] * 2 # local-global fusion 20240116
        self.lif_bf_head = act_layer() # try add plif before the head 2024.2.29
        self.dwconv_bf_head = nn.Conv3d(feature_outdim, feature_outdim, (cfg.DATA.NUM_FRAMES, img_size // 32, img_size // 32), groups=embed_dim[-1])
        self.dwconv_bn = nn.BatchNorm3d(feature_outdim)
        # self.pre_logits = nn.Identity()
        self.head = layer.Linear(feature_outdim, num_classes) if num_classes > 0 else nn.Identity()
        
        # # Representation layer
        # if representation_size:
        #     self.num_features = representation_size
        #     self.pre_logits = nn.Sequential(OrderedDict([
        #         ('fc', layer.Linear(embed_dim, representation_size)),
        #         ('act', nn.Tanh())
        #     ]))
        # else:
        #     self.pre_logits = nn.Identity()
        
        # # Classifier head
        # self.head = layer.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = layer.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def inflate_weight(self, weight_2d, time_dim, center=False):
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def get_pretrained_model(self, cfg):
        if cfg.UNIFORMER.PRETRAIN_NAME:
            checkpoint = torch.load(model_path[cfg.UNIFORMER.PRETRAIN_NAME], map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            elif 'model_state' in checkpoint:
                checkpoint = checkpoint['model_state']

            state_dict_3d = self.state_dict()
            for k in checkpoint.keys():
                if k not in state_dict_3d.keys():  # added 9.12 (to ignore unmatched nn.modules)
                    logger.info(f'Ignore: {k}')
                    continue
                if checkpoint[k].shape != state_dict_3d[k].shape:
                    if len(state_dict_3d[k].shape) <= 2:
                        logger.info(f'Ignore: {k}')
                        continue
                    logger.info(f'Inflate: {k}, {checkpoint[k].shape} => {state_dict_3d[k].shape}')
                    time_dim = state_dict_3d[k].shape[2]
                    checkpoint[k] = self.inflate_weight(checkpoint[k], time_dim)

            if self.num_classes != checkpoint['head.weight'].shape[0]:
                del checkpoint['head.weight'] 
                del checkpoint['head.bias'] 
            return checkpoint
        else:
            return None
            
    def forward_features(self, x):
        x = self.patch_embed1(x) # torch.Size([1, 3, 16, 224, 224]) -> torch.Size([1, 64, 8, 56, 56])
        x = self.pos_drop(x) # torch.Size([1, 64, 8, 56, 56]) -> torch.Size([1, 64, 8, 56, 56])
        for i, blk in enumerate(self.blocks1):
            if self.use_checkpoint and i < self.checkpoint_num[0]:
                x = checkpoint.checkpoint(blk, x)  # Checkpointing works by trading compute for memory. (/home/yult/miniconda3/envs/uniformer/lib/python3.8/site-packages/torch/utils/checkpoint.py)
            else:
                x = blk(x) # torch.Size([1, 64, 8, 56, 56]) -> torch.Size([1, 64, 8, 56, 56])
        x = self.patch_embed2(x) # torch.Size([1, 64, 8, 56, 56]) -> torch.Size([1, 128, 8, 28, 28])
        for i, blk in enumerate(self.blocks2):
            if self.use_checkpoint and i < self.checkpoint_num[1]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
                
        x = self.patch_embed3(x) # torch.Size([1, 128, 8, 28, 28]) -> torch.Size([1, 320, 8, 14, 14])  #T, B, C, H, W = x.shape
        
        # local_path 0116
        #x_lp_in = x  #T, B, C, H, W = x.shape
        #print('Shape of local path input:', x.shape)
        x_lp = self.lplif(x) # try add plif in the local-path 2024.2.29
        x_lp = self.lpDWConv(x_lp) # try add plif in the local-path 2024.2.29
#         x_lp = self.lpDWConv(x)
        x_lp = self.lpPWConv(x_lp)
        x_lp = self.lpBN(x_lp)
        #print('Shape of local path output:', x_lp.shape)
        
        for i, blk in enumerate(self.blocks3):
            if self.use_checkpoint and i < self.checkpoint_num[2]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.patch_embed4(x) # torch.Size([1, 320, 8, 14, 14]) -> torch.Size([1, 512, 8, 7, 7])
        for i, blk in enumerate(self.blocks4):
            if self.use_checkpoint and i < self.checkpoint_num[3]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
#         x = self.norm(x)  # no need? 1016  # torch.Size([1, 512, 8, 7, 7]) -> torch.Size([1, 512, 8, 7, 7])
        # x = self.pre_logits(x)
        
        # local_global_fusion 0116
        x = torch.cat((x_lp, x), 2) # concate along the channel dimension
        #print('Shape of feature output:', x.shape)  #T, B, C, H, W = x.shape
        return x

    def forward(self, x):
        # x = x[0] # tuple/list -> tensor (B, C, D, H, W) torch.Size([1, 3, 16, 224, 224])
        # print(type(x), len(x)) # <class 'list'> 1
        if isinstance(x, (tuple, list)):
            x = x[0]
        # print(type(x), x.shape) # <class 'torch.Tensor'> torch.Size([2, 3, 32, 224, 224]) (实际batchsize设为1，因为有augmentation的作用，变成了2)
        
        # T = self.T
        # # reorganize the input to be (T, B, C, D, H, W)
        # # x = (x.unsqueeze(0)).repeat(T, 1, 1, 1, 1, 1) # T=1
        # B, C, F, H, W = x.shape
        # D = torch.div(F, T, rounding_mode='trunc')
        # x = x.reshape(B, C, T, D, H, W).contiguous()
        # x = x.permute(2, 0, 1, 3, 4, 5).contiguous() # (T, B, C, D, H, W)
        # # print(x.shape) # torch.Size([4, 2, 3, 8, 224, 224])
        # # # save an example to check its temporal order
        # # x_save = '/home/yult/transformer/UniFormer/video_classification/exp/uniformer_snn_s4x8_sthv2_pressv2/x_TBCDHW.pkl'
        # # if not os.path.exists(x_save):
        # #     print(f'Save an input example to check: {x_save}')
        # #     with open(x_save, 'wb') as xf:
        # #         pkl.dump(x, xf)
        # #     xf.close()
        
#         print('original input, ', type(x), x.shape)
#         print('input(B, C, T, H, W) average across TCHW dimension(B): ', torch.mean(x, dim=[1,2,3,4]))
        x = x.permute(2, 0, 1, 3, 4).contiguous() # tensor (B, C, T, H, W) -> tensor (T, B, C, H, W)  # T = D
#         print('after permute, ', type(x), x.shape)
#         #print('input(T, B, C, H, W) average(T,B) across CHW dimension: ', torch.mean(x, dim=[2,3,4]))
#         print('input(T, B, C, H, W) average across BCHW dimension(T): ', torch.mean(x, dim=[1,2,3,4]))
        
        # # V1: try to use the temporal info in the classification head, not just average them 1031
        # x = self.forward_features(x) # tensor (T, B, C, H, W)
        # print(x.shape)
        # x = self.dwconv_bf_head(x)
        # print(x.shape) # tensor (T, B, C, 1, 1)
        # x = self.dwconv_bn(x)
        # print(x.shape) # tensor (T, B, C, 1, 1)
        # x = torch.squeeze(x) # tensor (T, B, C)
        # print(x.shape) 
        # x = x.permute(1, 0, 2) # tensor (B, T, C)
        # print(x.shape) 
        # x = x.flatten(1)  # tensor (B, T*C)
        # print(x.shape) 
        # x = self.head(x) # tensor (B, #classes)
        # print(x.shape) 
        
        # # x = self.forward_features(x) # torch.Size([T, 1, 512, 8, 7, 7])
        # # # print(x.shape)
        # # x = x.flatten(3).mean(-1) # torch.Size([T, 1, 512])
        # # # print(x.shape)
        # # x = self.head(x.mean(0)) # torch.Size([1, 174])
        # # # print(x.shape)
        
        # V2: try to use the temporal info in the classification head, not just average them 1031
        x = self.forward_features(x) # tensor (T, B, C, H, W)
        #print(x.shape)
        
        x = self.lif_bf_head(x) # try add plif before the head 2024.2.29
        
        x = x.permute(1, 2, 0, 3, 4) # tensor (B, C, T, H, W)
        x = self.dwconv_bf_head(x)
        # print(x.shape) # tensor (B, C, 1, 1, 1)
        x = self.dwconv_bn(x)
        # print(x.shape) # tensor (B, C, 1, 1, 1)
        #x = torch.squeeze(x) # tensor (B, C)
        #x = torch.squeeze(x, (1,2,3,4)) # tensor (B, C) # 2024.1.2 torch 2.0
        B, C, _, _, _ = x.shape
        x = torch.squeeze(x) # tensor (B, C, 1, 1, 1) -> (B, C)
        if B == 1: # 2024.1.2
            x = torch.unsqueeze(x, 0)
        # print(x.shape) 
        x = self.head(x) # tensor (B, #classes)
        # print(x.shape)
        
        return x
