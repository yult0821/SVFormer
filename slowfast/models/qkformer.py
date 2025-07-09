from math import ceil, sqrt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
# import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import _cfg
# from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
from .build import MODEL_REGISTRY
import os

try:
    from spikingjelly.activation_based import layer, neuron, surrogate  # spikingjelly13, spikingjelly14
except:
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode  # spikingjelly12
# from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from timm.models.layers import to_2tuple, trunc_normal_, DropPath
# from timm.models.registry import register_model
import torch.nn.functional as F

__all__ = ['qkformer']

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

model_path = '/home/yult/transformer/UniFormer/video_classification/exp/downloaded_ckpts' #'path_to_models'
model_path = {
    'spikingformer_cml_8x384_in1k': os.path.join(model_path, 'spikingformer_cml_8x384_in1k.pth'),
    'spikingformer_8x512_in1k': os.path.join(model_path, 'spikingformer_8x512_in1k.pth'),
    # 'uniformer_small_in1k': os.path.join(model_path, 'uniformer_small_in1k.pth'),
    # 'uniformer_small_k400_8x8': os.path.join(model_path, 'uniformer_small_k400_8x8.pth'),
    # 'uniformer_small_k400_16x4': os.path.join(model_path, 'uniformer_small_k400_16x4.pth'),
    # 'uniformer_small_k600_16x4': os.path.join(model_path, 'uniformer_small_k600_16x4.pth'),
    # 'uniformer_base_in1k': os.path.join(model_path, 'uniformer_base_in1k.pth'),
    # 'uniformer_base_k400_8x8': os.path.join(model_path, 'uniformer_base_k400_8x8.pth'),
    # 'uniformer_base_k400_16x4': os.path.join(model_path, 'uniformer_base_k400_16x4.pth'),
    # 'uniformer_base_k600_16x4': os.path.join(model_path, 'uniformer_base_k600_16x4.pth'),
}

def act_layer(): # try different neuron models
#     return neuron.KLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', backend='torch') #surrogate.Sigmoid()
    #return neuron.GatedLIFNode(T=16, surrogate_function=surrogate.Sigmoid(), step_mode='m', backend='torch') #modify line 2296 in modiUniFormer/video_classification/spikingjelly/spikingjelly/activation_based/neuron.py (failed!)
    return neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', backend='cupy') # backend='cupy' 'torch'
#     return neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', backend='cupy')


def compute_non_zero_rate(x):
    x_shape = torch.tensor(list(x.shape))
    all_neural = torch.prod(x_shape)
    z = torch.nonzero(x)
    print("After attention proj the none zero rate is", z.shape[0]/all_neural)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features
    def forward(self, x):
        T,B,C,W,H = x.shape
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T,B,self.c_hidden,W,H).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T,B,C,W,H).contiguous()
        x = self.fc2_lif(x)
        return x

class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = act_layer() #MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')


    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        q = torch.sum(q, dim = 3, keepdim = True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)

        return x


class Spiking_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = act_layer() #MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        x_feat = x
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T,B,C,W,H)

        return x

class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.attn = Spiking_Self_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        # Downsampling + Res 0
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims)
        self.proj1_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj2_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims)
        self.proj2_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')


    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x)
        x = self.proj_maxpool(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.proj1_maxpool(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.proj2_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//4, W//4).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x

class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = act_layer() #MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.proj3_maxpool(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//2, W//2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x

@MODEL_REGISTRY.register()
class Qkformer(nn.Module):
    def __init__(self, cfg
                #  T=4,
                #  img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                #  embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                #  drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                #  depths=[6, 8, 6], sr_ratios=[8, 4, 2]
                 ):
        super().__init__()
        
        num_classes = cfg.MODEL.NUM_CLASSES 
        img_size_h, img_size_w = cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE
        in_channels = cfg.DATA.INPUT_CHANNEL_NUM[0]
        
        depths = cfg.QKFORMER.DEPTH
        patch_size = cfg.QKFORMER.PATCH_SIZE
        embed_dims = cfg.QKFORMER.EMBED_DIM
        num_heads = cfg.QKFORMER.NUM_HEADS
        mlp_ratios = cfg.QKFORMER.MLP_RATIO
        qkv_bias = cfg.QKFORMER.QKV_BIAS
        qk_scale = cfg.QKFORMER.QKV_SCALE
        drop_rate = cfg.QKFORMER.DROPOUT_RATE
        attn_drop_rate = cfg.QKFORMER.ATTENTION_DROPOUT_RATE
        drop_path_rate = cfg.QKFORMER.DROP_DEPTH_RATE
        sr_ratios = cfg.QKFORMER.SR_RATIOS
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.num_classes = num_classes
        self.depths = depths
        # self.T = T
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims // 4)

        stage1 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 4, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])

        patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 2)


        stage2 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 2, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(2)])


        patch_embed3 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims)

        stage3 = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths - 3)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        # classification head 这里不需要脉冲，因为输入的是在T时长平均发射值
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed3, H, W):
        if H * W == self.patch_embed3.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed3.H, patch_embed3.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    def get_pretrained_model(self, cfg):
        if cfg.QKFORMER.PRETRAIN_NAME:
            checkpoint = torch.load(model_path[cfg.QKFORMER.PRETRAIN_NAME], map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            elif 'model_state' in checkpoint:
                checkpoint = checkpoint['model_state']
            elif 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            # state_dict_3d = self.state_dict()
            # for k in checkpoint.keys():
            #     if checkpoint[k].shape != state_dict_3d[k].shape:
            #         if len(state_dict_3d[k].shape) <= 2:
            #             logger.info(f'Ignore: {k}')
            #             continue
            #         logger.info(f'Inflate: {k}, {checkpoint[k].shape} => {state_dict_3d[k].shape}')
            #         time_dim = state_dict_3d[k].shape[2]
            #         checkpoint[k] = self.inflate_weight(checkpoint[k], time_dim)

            if self.num_classes != checkpoint['head.weight'].shape[0]:
                del checkpoint['head.weight'] 
                del checkpoint['head.bias'] 
            return checkpoint
        else:
            return None

        
    def forward_features(self, x):

        stage1 = getattr(self, f"stage1")
        stage2 = getattr(self, f"stage2")
        stage3 = getattr(self, f"stage3")
        patch_embed1 = getattr(self, f"patch_embed1")
        patch_embed2 = getattr(self, f"patch_embed2")
        patch_embed3 = getattr(self, f"patch_embed3")

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        x = patch_embed3(x)
        for blk in stage3:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        # T = self.T
        # x = (x.unsqueeze(0)).repeat(T, 1, 1, 1, 1)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.permute(2, 0, 1, 3, 4).contiguous() # (B, C, T, H, W) --> (T, B, C, H, W)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

