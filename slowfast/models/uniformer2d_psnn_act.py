# -*- coding: utf-8 -*-
from math import ceil, sqrt
from collections import OrderedDict
import torch
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
}


# def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
#     return layer.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups, step_mode='m')

# def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
#     return layer.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups, step_mode='m')

# def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
#     return layer.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups, step_mode='m')

def conv_nxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return layer.Conv2d(inp, oup, (kernel_size, kernel_size), (stride, stride), (0, 0), groups=groups, step_mode='m')

def conv_1x1(inp, oup, groups=1):
    return layer.Conv2d(inp, oup, (1, 1), (1, 1), (0, 0), groups=groups, step_mode='m')

def conv_3x3(inp, oup, groups=1):
    return layer.Conv2d(inp, oup, (3, 3), (1, 1), (1, 1), groups=groups, step_mode='m')

def conv_5x5(inp, oup, groups=1):
    return layer.Conv2d(inp, oup, (5, 5), (1, 1), (2, 2), groups=groups, step_mode='m')

def bn_2d(dim):
    return layer.BatchNorm2d(dim, step_mode='m')

def act_layer():
    return neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', backend='cupy') # backend='cupy' 'torch'


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=act_layer, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act1 = act_layer()
        self.fc1 = layer.Linear(in_features, hidden_features)
        self.drop1 = layer.Dropout(drop)
        self.act2 = act_layer()
        self.fc2 = layer.Linear(hidden_features, out_features)
        self.drop2 = layer.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.act2(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        
        self.in_lif = act_layer()
        self.qkv = layer.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = layer.Dropout(attn_drop)
        self.q_lif = act_layer()
        self.k_lif = act_layer()
        self.v_lif = act_layer()
        self.proj_lif = act_layer()
        self.proj = layer.Linear(dim, dim)
        self.proj_drop = layer.Dropout(proj_drop)

    def forward(self, x):
        
        x = self.in_lif(x)
        T, B, N, C = x.shape
        x = x.flatten(0, 1) # torch.Size([1, 1568, 320])
        # TB, N, C = x.shape 
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # torch.Size([3, 1, 5, 1568, 64])
        qkv = self.qkv(x).reshape(T*B, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode='trunc')).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # torch.Size([1, 5, 1568, 64])
        
        # transfer into spikes
        q_sk = self.q_lif(q.reshape(T, B, self.num_heads, N, torch.div(C, self.num_heads, rounding_mode='trunc')).contiguous())
        k_sk = self.k_lif(k.reshape(T, B, self.num_heads, N, torch.div(C, self.num_heads, rounding_mode='trunc')).contiguous())
        v_sk = self.v_lif(v.reshape(T, B, self.num_heads, N, torch.div(C, self.num_heads, rounding_mode='trunc')).contiguous())

        # attn = (q @ k.transpose(-2, -1)) * self.scale # torch.Size([1, 5, 1568, 1568])
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C) # torch.Size([1, 5, 1568, 64]) -> torch.Size([1, 1568, 320])
        
        x = k_sk.transpose(-2,-1) @ v_sk
        x = (q_sk @ x) * self.scale
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        
        x = self.proj_lif(x)
        x = self.proj(x) # torch.Size([1, 1568, 320])
        x = self.proj_drop(x) # torch.Size([1, 1568, 320])
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=act_layer, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act1 = act_layer()
        self.fc1 = conv_1x1(in_features, hidden_features)
        self.drop1 = layer.Dropout(drop)
        self.act2 = act_layer()
        self.fc2 = conv_1x1(hidden_features, out_features)
        self.drop2 = layer.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.act2(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=act_layer, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_lif = act_layer()
        self.pos_embed = conv_3x3(dim, dim, groups=dim)
        self.norm1 = bn_2d(dim)
        self.act1 = act_layer()
        # 0821 - apply less lif layer, exclude act2 and act3
        self.conv1 = conv_1x1(dim, dim, 1) # not change size
        # self.conv2 = conv_1x1(dim, dim, 1) # not change size
#         self.act2 = act_layer()
        self.attn = conv_5x5(dim, dim, groups=dim) # not change size
#         self.act3 = act_layer()
        self.conv2 = conv_1x1(dim, dim, 1) # not change size
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = self.in_lif(x)
        x = x + self.pos_embed(self.in_lif(x)) # torch.Size([1, 64, 8, 56, 56]) -> torch.Size([1, 64, 8, 56, 56])
        
        # x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x))))) # local MHRA # not change size # should be decomposed to add lif-neuron
        x_pe = x
        x = self.norm1(x)
        x = self.act1(x)
        # 0821 - apply less lif layer, exclude act2 and act3
        x = self.conv1(x)
#         x = self.act2(x)
        x = self.attn(x)
#         x = self.act3(x)
        x = self.conv2(x)
        x = x_pe + self.drop_path(x)
        
        # x_attn = x
        # x = self.norm2(x)
        # x = self.mlp(x)
        # x = x_attn + self.drop_path(x) # not change size
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x   


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=act_layer, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_lif = act_layer()
        self.pos_embed = conv_3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop) # global MHRA
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = self.in_lif(x)
        x = x + self.pos_embed(self.in_lif(x))
        T, B, C, H, W = x.shape
        x = x.flatten(3).transpose(2, 3) # torch.Size([1, 320, 8, 14, 14]) -> torch.Size([1, 1568, 320])
        x = x + self.drop_path(self.attn(self.norm1(x))) # torch.Size([1, 1568, 320])
        x = x + self.drop_path(self.mlp(self.norm2(x))) # torch.Size([1, 1568, 320])
        x = x.transpose(2, 3).reshape(T, B, C, H, W) # torch.Size([1, 320, 8, 14, 14])
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
        self.proj = conv_nxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.in_lif(x) # SpeicalPatchEmbed directly receive external inputs, act as input encoding layer
        x = self.proj(x)
        T, B, C, H, W = x.shape
        x = x.flatten(3).transpose(2, 3)
        x = self.norm(x)
        x = x.reshape(T, B, H, W, -1).permute(0, 1, 4, 2, 3).contiguous()
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
        self.proj = conv_nxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        # if std:
        #     self.proj = conv_3xnxn_std(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        # else:
        #     self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.in_lif(x)
        x = self.proj(x) # torch.Size([1, 64, 8, 56, 56]) -> torch.Size([1, 128, 8, 28, 28])
        T, B, C, H, W = x.shape # [1, 128, 8, 28, 28]
        x = x.flatten(3).transpose(2, 3) # torch.Size([1, 64, 8, 56, 56]) -> torch.Size([1, 128, 6272]) -> torch.Size([1, 6272, 128])
        x = self.norm(x) # torch.Size([1, 6272, 128])
        x = x.reshape(T, B, H, W, -1).permute(0, 1, 4, 2, 3).contiguous() # torch.Size([1, 8, 28, 28, 128]) -> torch.Size([1, 128, 8, 28, 28])
        return x


@MODEL_REGISTRY.register()
class Uniformer2d_psnn_act(nn.Module):
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
        self.norm = bn_2d(embed_dim[-1])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', layer.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier head
        self.head = layer.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        
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
        x = self.patch_embed3(x) # torch.Size([1, 128, 8, 28, 28]) -> torch.Size([1, 320, 8, 14, 14])
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
        x = self.norm(x)  # torch.Size([1, 512, 8, 7, 7]) -> torch.Size([1, 512, 8, 7, 7])
        x = self.pre_logits(x)
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
        
        x = x.permute(2, 0, 1, 3, 4).contiguous() # (T, B, C, H, W)  # T = D
        
        x = self.forward_features(x) # torch.Size([T, 1, 512, 8, 7, 7])
        # print(x.shape)
        x = x.flatten(3).mean(-1) # torch.Size([T, 1, 512])
        # print(x.shape)
        x = self.head(x.mean(0)) # torch.Size([1, 174])
        # print(x.shape)
        return x
