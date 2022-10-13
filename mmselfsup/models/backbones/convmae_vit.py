# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch

from .mae_vit import MAEViT


class ConvMAEViT(MAEViT):
    """Convolution Vision Transformer for ConvMAE pre-training.

    This module is based on the implementation of MAEViT, but make some
    custom changes, e.g. adding convolution blocks and fusing multi-scale
    features before feeding into the decoder. For more details, please refer
    to the paper `ConvMAE: Masked Convolution Meets Masked Autoencoders
     <https://arxiv.org/pdf/2205.03892.pdf>`_.

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        depth (list): The depth of each block. Defaults to [2, 2, 11].
        embed_dims (list): The feature dimension of each block. Defaults to
            [256, 384, 768].
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'b',
                 depth: List[int] = [2, 2, 11],
                 embed_dims: List[int] = [256, 384, 768],
                 img_size: int = 224,
                 patch_size: int = 16,
                 out_indices: Union[Sequence, int] = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 output_cls_token: bool = True,
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 mask_ratio: float = 0.75,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        # do not use cls_token in ConvMAE
        del self.cls_token
        # TODO: create all kinds of blocks here, e.g. conv blocks and
        #  transformer blocks.
        # Note: only implement these components related to the encoder, you
        #  can refer to MAEViT for more details

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: implement ConvMAE custom random masking here
        pass

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # TODO: implement ConvMAE custom forward here
        # Note: only implement the forward of encoder, you can refer to
        #  MAEViT for more details
        pass
