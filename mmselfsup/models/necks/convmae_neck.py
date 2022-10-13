# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .mae_neck import MAEPretrainDecoder


class ConvMAEPretrainDecoder(MAEPretrainDecoder):
    """Decoder for ConvMAE Pre-training.

    This module is based on MAEPretrainDecoder, but with a different forward
    function. Compared to MAEPretrainDecoder, this module does not process the
    cls_token.
    """

    def forward(self, x: torch.Tensor,
                ids_restore: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward function, related to the decoder.
        # You can refer to the forward function in MAEPretrainDecoder.
        # Compared to MAEPretrainDecoder, this module does not process the
        # cls_token.
        pass
