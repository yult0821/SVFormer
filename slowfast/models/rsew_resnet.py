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

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, layer, neuron, surrogate # spikingjelly13, spikingjelly14

# from timm.models.layers import to_2tuple, trunc_normal_, DropPath
# from timm.models.registry import register_model

# __all__ = ['RecurrentSEWResNet', 'ConvRecurrentContainer', 'r_sew_resnet18', 'r_sew_resnet34', 'r_sew_resnet50', 'r_sew_resnet101', 'r_sew_resnet152']
__all__ = ['RecurrentSEWResNet', 'ConvRecurrentContainer']


import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

# model_path = '/home/yult/transformer/UniFormer/video_classification/exp/downloaded_ckpts' #'path_to_models'
# model_path = '/userhome/transformer/UniFormer/video_classification/exp/downloaded_ckpts' #'path_to_models'
model_path = '/home/yult/projects/uniformer250704/UniFormer/video_classification/exp/downloaded_ckpts'
model_path = {
    'spikingformer_cml_8x384_in1k': os.path.join(model_path, 'spikingformer_cml_8x384_in1k.pth'),
    'spikingformer_8x512_in1k': os.path.join(model_path, 'spikingformer_8x512_in1k.pth'),
    'sew_resnet101_in1k': os.path.join(model_path, 'sew_resnet101_in1k.pth'),
    'sew_resnet50_in1k': os.path.join(model_path, 'sew_resnet50_in1k.pth'),
    'sew_resnet18_in1k': os.path.join(model_path, 'sew_resnet18_in1k.pth'),
    # 'uniformer_small_in1k': os.path.join(model_path, 'uniformer_small_in1k.pth'),
    # 'uniformer_small_k400_8x8': os.path.join(model_path, 'uniformer_small_k400_8x8.pth'),
    # 'uniformer_small_k400_16x4': os.path.join(model_path, 'uniformer_small_k400_16x4.pth'),
    # 'uniformer_small_k600_16x4': os.path.join(model_path, 'uniformer_small_k600_16x4.pth'),
    # 'uniformer_base_in1k': os.path.join(model_path, 'uniformer_base_in1k.pth'),
    # 'uniformer_base_k400_8x8': os.path.join(model_path, 'uniformer_base_k400_8x8.pth'),
    # 'uniformer_base_k400_16x4': os.path.join(model_path, 'uniformer_base_k400_16x4.pth'),
    # 'uniformer_base_k600_16x4': os.path.join(model_path, 'uniformer_base_k600_16x4.pth'),
}


class ConvRecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module, in_channels, out_channels, stride, step_mode="s"):
        super().__init__()
        self.step_mode = step_mode
        assert not hasattr(sub_module, "step_mode") or sub_module.step_mode == "s"
        self.sub_module_out_channels = out_channels
        self.sub_module = sub_module
        
        if stride > 1:
            self.dwconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, 
                                             padding=1, output_padding=1, groups=out_channels, bias=False)
            self.bn1 = layer.BatchNorm2d(out_channels)
            self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        else:
            self.dwconv = None
        self.pwconv = nn.Conv2d(in_channels + out_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn = layer.BatchNorm2d(in_channels)
        self.sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        
        self.register_memory("y", None)
    
    def single_step_forward(self, x):
        if self.y is None:
            if self.dwconv is None:
                self.y = torch.zeros(x.size(0), self.sub_module_out_channels, x.size(2), x.size(3)).to(x)
            else:
                h = (x.size(2) + 2 - (3 - 1) - 1) // 2 + 1
                w = (x.size(3) + 2 - (3 - 1) - 1) // 2 + 1
                self.y = torch.zeros(x.size(0), self.sub_module_out_channels, h, w).to(x)
        if self.dwconv is None:
            out = self.y
        else:
            out = self.sn1(self.bn1(self.dwconv(self.y)))
        out = torch.cat((x, out), dim=1)
        out = self.bn(self.pwconv(out))
        out = self.sn(out)
        self.y = self.sub_module(out)
        
        return self.y


def sew_function(x, y, cnf):
    if cnf == "ADD":
        return x + y
    elif cnf == "AND":
        return x * y
    elif cnf == "OR":
        return x + y - x * y
    else:
        raise NotImplementedError


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, cnf=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.stride = stride
        self.cnf = cnf
    
    def forward(self, x):
        identity = x
        
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.sn2(self.bn2(self.conv2(out)))
        
        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))
        
        out = sew_function(out, identity, self.cnf)
        
        return out
    
    def extra_repr(self):
        return super().extra_repr() + f"cnf={self.cnf}"


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, cnf=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.stride = stride
        self.cnf = cnf
    
    def forward(self, x):
        identity = x
        
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.sn2(self.bn2(self.conv2(out)))
        out = self.sn3(self.bn3(self.conv3(out)))
        
        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))
        
        out = sew_function(out, identity, self.cnf)
        
        return out
    
    def extra_repr(self):
        return super().extra_repr() + f"cnf={self.cnf}"


@MODEL_REGISTRY.register()
class RecurrentSEWResNet(nn.Module):
    def __init__(self, cfg
                #  block, layers, num_classes=101, groups=1, width_per_groups=64, 
                #  norm_layer=None, cnf=None, zero_init_residual=False
                 ):
        super(RecurrentSEWResNet, self).__init__()
        # if norm_layer is None:
        #     norm_layer = layer.BatchNorm2d
        # self._norm_layer = norm_layer
        
        self._norm_layer = layer.BatchNorm2d
        self.in_planes = 64
        self.groups = 1
        self.base_width = 64
        zero_init_residual=False
        
        # block = Bottleneck  # (sew_resnet50 [3, 4, 6, 3], sew_resnet101 [3, 4, 23, 3], sew_resnet152 [3, 8, 36, 3])  
        # block = BasicBlock  # (sew_resnet18 [2, 2, 2, 2], sew_resnet34 [3, 4, 6, 3])
        layers = cfg.SEWRESNET.LAYERS  # [3, 4, 23, 3] for sew_resnet101
        cnf = cfg.SEWRESNET.CNF
        if sum(cfg.SEWRESNET.LAYERS) < 10: # ignore sew_resnet34 [3, 4, 6, 3]
            block = BasicBlock  # sew_resnet18
        else:
            block = Bottleneck  # (sew_resnet50 [3, 4, 6, 3], sew_resnet101 [3, 4, 23, 3], sew_resnet152 [3, 8, 36, 3]) 
        
        self.num_classes = cfg.MODEL.NUM_CLASSES 
        self.use_checkpoint = cfg.MODEL.USE_CHECKPOINT # Checkpointing works by trading compute for memory. (/home/yult/miniconda3/envs/uniformer/lib/python3.8/site-packages/torch/utils/checkpoint.py)
        self.checkpoint_num = cfg.MODEL.CHECKPOINT_NUM
        
        
        self.conv1 = layer.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.in_planes)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer1 = self._make_layer(block, 64, layers[0], cnf=cnf)
        self.recurrent_layer1 = layer1 # ConvRecurrentContainer(layer1, in_channels=64, out_channels=64 * block.expansion, stride=1)
        layer2 = self._make_layer(block, 128, layers[1], stride=2, cnf=cnf)
        self.recurrent_layer2 = layer2 # ConvRecurrentContainer(layer2, in_channels=64 * block.expansion, out_channels=128 * block.expansion, stride=2)
        layer3 = self._make_layer(block, 256, layers[2], stride=2, cnf=cnf)
        self.recurrent_layer3 = layer3 # ConvRecurrentContainer(layer3, in_channels=128 * block.expansion, out_channels=256 * block.expansion, stride=2)
        layer4 = self._make_layer(block, 512, layers[3], stride=2, cnf=cnf)
        # only keep recurrent connections for the last layer
        self.recurrent_layer4 = ConvRecurrentContainer(layer4, in_channels=256 * block.expansion, out_channels=512 * block.expansion, stride=2)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, self.num_classes)
        
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, num_blocks, stride=1, cnf=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups, self.base_width, self._norm_layer, cnf))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, groups=self.groups, base_width=self.base_width, norm_layer=self._norm_layer, cnf=cnf))
        
        return nn.Sequential(*layers)
    
    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    def get_pretrained_model(self, cfg):
        if cfg.SEWRESNET.PRETRAIN_NAME:
            checkpoint = torch.load(model_path[cfg.SEWRESNET.PRETRAIN_NAME], map_location='cpu')
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

            if self.num_classes != checkpoint['fc.weight'].shape[0]: # ['head.weight']
                del checkpoint['fc.weight'] 
                del checkpoint['fc.bias'] 
            return checkpoint
        else:
            return None
    
    
    def forward(self, x):
        
        # prepare the correct-format input for snn
        # print(type(x))
        if isinstance(x, (tuple, list)):
            x = x[0]
        # print("Original input shape:", x.shape)
        x = x.permute(2, 0, 1, 3, 4).contiguous() # tensor (B, C, T, H, W) -> tensor (T, B, C, H, W)
        # print("Rearranged input shape:", x.shape)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)
        
        x = self.recurrent_layer1(x)
        x = self.recurrent_layer2(x)
        x = self.recurrent_layer3(x)
        x = self.recurrent_layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        # x = self.fc(x)
        x = self.fc(x.mean(0)) # average the T dimension
        
        return x

        
# def r_sew_resnet18(**kwargs):
#     return RecurrentSEWResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


# def r_sew_resnet34(**kwargs):
#     return RecurrentSEWResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


# def r_sew_resnet50(**kwargs):
#     return RecurrentSEWResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def r_sew_resnet101(**kwargs):
#     return RecurrentSEWResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


# def r_sew_resnet152(**kwargs):
#     return RecurrentSEWResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

