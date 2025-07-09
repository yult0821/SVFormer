from torch import Tensor
import torch.nn as nn

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

    