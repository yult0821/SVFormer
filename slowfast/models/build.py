#!/usr/bin/env python3
# modified from https://github.com/facebookresearch/SlowFast
"""Model construction functions."""

import torch
from fvcore.common.registry import Registry

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    
    if cfg.MODEL.ARCH in ['uniformer', 'spikingformer', 'spikingformer_cml', 'sew_resnet', 'uniformer_snn', 'uniformer_psnn', 'uniformer2d_snn', 'uniformer2d_psnn', 'uniformer2d_psnn_act', 'uniformer2d_psnn_norm', 'uniformer2d_psnn_bn', 'uniformer2d_psnn_try', 'uniformer2d_psnn_tri', 'uniformer2d_prsnn', 'rsew_resnet', 'uniformer2d_psnn_convBnLif', 'uniformer2d_psnn_head', 'uniformer2d_psnn_lgf', 'uniformer2d_psnn_qkf', 'uniformer2d_psnn_convqkf', 'uniformer2d_if_lgf', 'uniformer2d_psnn_lgfT', 'qkformer', 'uniformer3d_psnn', 'uniformer3d_psnn_lgf', 'sglformer', 'metaspikformer']:
        checkpoint = model.get_pretrained_model(cfg)
        if checkpoint:
            logger.info('load pretrained model')
            model.load_state_dict(checkpoint, strict=False)
            
    if cfg.MODEL.ARCH in ['sew_resnet']:
        from spikingjelly.activation_based import functional, neuron # spikingjelly > 0.0.0.0.12
        functional.set_step_mode(model, 'm') # 将 net 中所有模块的步进模式设置为 'm' 
        functional.set_backend(model, 'cupy', neuron.BaseNode) # 将 net 中 所有类型为 instance 的模块后端更改为 'cupy'
        
    if cfg.MODEL.ARCH in ['rsew_resnet', 'uniformer2d_prsnn']:
        from . import rsew_resnet_helper
        if cfg.MODEL.ARCH in ['rsew_resnet']:
            from .rsew_resnet import ConvRecurrentContainer
        elif cfg.MODEL.ARCH in ['uniformer2d_prsnn']:
            from .uniformer2d_prsnn import ConvRecurrentContainer
        from spikingjelly.activation_based import neuron
        print("set_step_mode and set_backend for recurrent models")
        rsew_resnet_helper.set_step_mode(model, step_mode='m', keep_instance=(ConvRecurrentContainer,)) # 将 net 中除了ConvRecurrentContainer之外的模块的步进模式设置为 'm'
        rsew_resnet_helper.set_backend(model, backend='cupy', instance=neuron.BaseNode, keep_instance=(ConvRecurrentContainer,))
        
    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
#         # Make model replica operate on the current device
#         model = torch.nn.parallel.DistributedDataParallel(
#             module=model, device_ids=[cur_device], output_device=cur_device,
#             find_unused_parameters=False  # modified by yult 0529 to solve the error info when using multiple gpus in spikingformer (RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.) (for uniformer2d_psnn_tri41_base36_w16a_qkf_ucf101_scratch, set find_unused_parameters=True, 2024.1.19)
#         )  # original version: find_unused_parameters=False

        if cfg.MODEL.ARCH in ['spikingformer_cml', 'spikingformer', 'qkformer', 'sglformer']:
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device, 
                find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device, 
                find_unused_parameters=False)        
            
    return model
