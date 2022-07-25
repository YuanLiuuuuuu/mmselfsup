from typing import Tuple
from mmcls.models import ResNet
from torch import nn
import torch
from mmselfsup.registry import MODELS
from mmcv.cnn import (build_conv_layer, build_norm_layer)
from mmcv.cnn.utils.weight_init import trunc_normal_


@MODELS.register_module()
class SimMIMResNet(ResNet):

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 stem_patch_size=4,
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 mask=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0):
        self.stem_patch_size = stem_patch_size
        super().__init__(
            depth=depth,
            in_channels=in_channels,
            stem_channels=stem_channels,
            base_channels=base_channels,
            expansion=expansion,
            num_stages=num_stages,
            strides=strides,
            dilations=dilations,
            out_indices=out_indices,
            style=style,
            deep_stem=deep_stem,
            avg_down=avg_down,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual,
            init_cfg=init_cfg,
            drop_path_rate=drop_path_rate)

        self._make_stem_layer(in_channels, stem_channels)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.stem_channels))
        self.mask = mask

        trunc_normal_(self.mask_token, mean=0, std=.02)

    def _make_stem_layer(self, in_channels, stem_channels):

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=self.stem_patch_size,
            stride=self.stem_patch_size)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor]:
        # Used to patchify image into non-overlapped tokens
        x = self.conv1(x)
        x = self.norm1(x)
        x: torch.Tensor = self.relu(x)
        # if mask is True, apply mask on input images
        if self.mask:
            # now x is the patchfied image
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)

            assert mask is not None
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1. - w) + mask_token * w

            # x is the masked image
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
