from mmcls.models import ConvNeXt
from mmcv.cnn.utils.weight_init import trunc_normal_
from torch import nn
import torch
from ..builder import BACKBONES


@BACKBONES.register_module()
class SimMIMConvNext(ConvNeXt):

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 init_cfg=None):

        super().__init__(arch, in_channels, stem_patch_size, norm_cfg, act_cfg,
                         linear_pw_conv, drop_path_rate,
                         layer_scale_init_value, out_indices, frozen_stages,
                         gap_before_final_norm, init_cfg)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.channels[0]))

        trunc_normal_(self.mask_token, mean=0, std=.02)

    def forward(self, x, mask):
        # Used to patchify image into non-overlapped tokens
        x = self.downsample_layers[0](x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)

        assert mask is not None
        B, L, _ = x.shape
        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1. - w) + mask_token * w

        x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())
        return tuple(outs)
