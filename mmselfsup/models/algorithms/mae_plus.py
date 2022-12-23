# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .mae import MAE


@MODELS.register_module()
class MAEPlus(MAE):

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        low_freq_targets = self.target_generator(inputs[0])
        latent, mask, ids_restore, weights = self.backbone(inputs[0])
        pred = self.neck(latent, ids_restore)
        loss = self.head(pred, low_freq_targets, mask)
        weight_params = {
            f'weight_{i}': weights[i]
            for i in range(weights.size(0))
        }
        losses = dict(loss=loss)
        losses.update(weight_params)
        return losses
