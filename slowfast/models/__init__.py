#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .custom_video_model_builder import *  # noqa
from .ptv_model_builder import (
    PTVCSN,
    PTVX3D,
    PTVR2plus1D,
    PTVResNet,
    PTVSlowFast,
)  # noqa
from .video_model_builder import ResNet, SlowFast  # noqa
from .uniformer import Uniformer  # noqa
# from .spikingformer import Spikingformer  # spikingjelly 0.0.0.0.12
from .spikingformer_cml import Spikingformer_cml  # spikingjelly 0.0.0.0.12 (modified to be consistant with sj13)
from .sew_resnet import SEWResNet
from .uniformer_snn import Uniformer_snn
from .uniformer_psnn import Uniformer_psnn # as tri8 in uniformer2d_psnn for unif3d
from .uniformer2d_snn import Uniformer2d_snn # LIFNode
from .uniformer2d_psnn import Uniformer2d_psnn # ParametricLIFNode
from .uniformer2d_psnn_act import Uniformer2d_psnn_act # ParametricLIFNode
from .uniformer2d_psnn_norm import Uniformer2d_psnn_norm # ParametricLIFNode
from .uniformer2d_psnn_bn import Uniformer2d_psnn_bn # ParametricLIFNode
from .uniformer2d_psnn_try import Uniformer2d_psnn_try # ParametricLIFNode
from .uniformer2d_psnn_tri import Uniformer2d_psnn_tri # ParametricLIFNode
from .uniformer2d_prsnn import Uniformer2d_prsnn # add a recurrent layer before the classification head
from .rsew_resnet import RecurrentSEWResNet
from .uniformer2d_psnn_convBnLif import Uniformer2d_psnn_convBnLif # ParametricLIFNode
from .uniformer2d_psnn_head import Uniformer2d_psnn_head # ParametricLIFNode
from .uniformer2d_psnn_lgf import Uniformer2d_psnn_lgf # ParametricLIFNode
from .uniformer2d_psnn_qkf import Uniformer2d_psnn_qkf # ParametricLIFNode
from .uniformer2d_psnn_convqkf import Uniformer2d_psnn_convqkf # ParametricLIFNode
from .uniformer2d_if_lgf import Uniformer2d_if_lgf # IFNode
from .uniformer2d_psnn_lgfT import Uniformer2d_psnn_lgfT # ParametricLIFNode, T=24
from .qkformer import Qkformer
from .uniformer3d_psnn import Uniformer3d_psnn # 
from .uniformer3d_psnn_lgf import Uniformer3d_psnn_lgf
from .sglformer import Sglformer
from .metaspikformer import Metaspikformer
