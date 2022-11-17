from mmselfsup.registry import TRANSFORMS
from mmcv.transforms.base import BaseTransform
from PIL import Image
import torch
from torchvision import transforms
import math
import torchvision.transforms.functional as F
import numpy as np


class RandomResizedCrop(transforms.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F.get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1, )).item()
        j = torch.randint(0, width - w + 1, size=(1, )).item()

        return i, j, h, w


@TRANSFORMS.register_module()
class PytorchRandomResizedCrop(BaseTransform):

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=3) -> None:
        self.resize_crop = RandomResizedCrop(
            size=size, scale=scale, ratio=ratio, interpolation=interpolation)

    def transform(self, results: dict) -> dict:
        img = results['img']
        img_pil = Image.fromarray(img)
        img_pil = self.resize_crop(img_pil)
        img_array = np.array(img_pil)
        results['img'] = img_array
        results['img_shape'] = img_array.shape[:2]

        return results


@TRANSFORMS.register_module()
class PytorchRandomHorizontalFlip(BaseTransform):

    def __init__(self, p=0.5) -> None:
        self.flip = transforms.RandomHorizontalFlip(p=p)

    def transform(self, results: dict) -> dict:
        img = results['img']
        img_pil = Image.fromarray(img)
        img_pil = self.flip(img_pil)
        img_array = np.array(img_pil)
        results['img'] = img_array
        results['img_shape'] = img_array.shape[:2]

        return results


@TRANSFORMS.register_module()
class PytorchResize(BaseTransform):

    def __init__(self, size, interpolation=3) -> None:
        self.resize = transforms.Resize(size=size, interpolation=interpolation)

    def transform(self, results: dict) -> dict:
        img = results['img']
        img_pil = Image.fromarray(img)
        img_pil = self.resize(img_pil)
        img_array = np.array(img_pil)
        results['img'] = img_array
        results['img_shape'] = img_array.shape[:2]

        return results


@TRANSFORMS.register_module()
class PytorchCenterCrop(BaseTransform):

    def __init__(self, size) -> None:
        self.crop = transforms.CenterCrop(size=size)

    def transform(self, results: dict) -> dict:
        img = results['img']
        img_pil = Image.fromarray(img)
        img_pil = self.crop(img_pil)
        img_array = np.array(img_pil)
        results['img'] = img_array
        results['img_shape'] = img_array.shape[:2]

        return results