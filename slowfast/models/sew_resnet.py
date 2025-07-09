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

try:
    from spikingjelly.activation_based import layer, neuron, surrogate  # spikingjelly13, spikingjelly14
except:
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode  # spikingjelly12

from timm.models.layers import to_2tuple, trunc_normal_, DropPath
# from timm.models.registry import register_model
import torch.nn.functional as F


# __all__ = ['SEWResNet', 'sew_resnet18', 'sew_resnet34', 'sew_resnet50', 'sew_resnet101', 'sew_resnet152']
__all__ = ['SEWResNet']

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

model_path = '/userhome/transformer/UniFormer/video_classification/exp/downloaded_ckpts' #'path_to_models'
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
class SEWResNet(nn.Module):
    def __init__(self, cfg,
                #  block, layers, num_classes=1000, groups=1, width_per_groups=64, 
                #  norm_layer=None, cnf=None, zero_init_residual=False
                 ):
        super(SEWResNet, self).__init__()
        # if norm_layer is None:
        #     norm_layer = layer.BatchNorm2d
        # self._norm_layer = norm_layer
        
        self._norm_layer = layer.BatchNorm2d
        self.in_planes = 64
        self.groups = 1
        self.base_width = 64
        zero_init_residual = False
        
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
        self.layer1 = self._make_layer(block, 64, layers[0], cnf=cnf)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, cnf=cnf)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, cnf=cnf)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, cnf=cnf)
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
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.permute(2, 0, 1, 3, 4).contiguous() # tensor (B, C, T, H, W) -> tensor (T, B, C, H, W)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        # x = self.fc(x)
        x = self.fc(x.mean(0)) # average the T dimension
        
        return x

        
# def sew_resnet18(**kwargs):
#     return SEWResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

# def sew_resnet34(**kwargs):
#     return SEWResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

# def sew_resnet50(**kwargs):
#     return SEWResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

# def sew_resnet101(**kwargs):
#     return SEWResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

# def sew_resnet152(**kwargs):
#     return SEWResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

