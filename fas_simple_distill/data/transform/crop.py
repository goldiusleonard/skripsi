import random
from typing import Tuple, Sequence

import torch
from PIL import Image
from torchvision.transforms import functional as TF


class SquareCrop:
    def __call__(self, x):
        if isinstance(x, Image.Image):
            w, h = x.size
        elif isinstance(x, torch.Tensor):
            _, h, w = x.shape
        else:
            raise TypeError(f"Cannot handle type {type(x)}.")

        min_side = min(h, w)
        return TF.center_crop(x, (min_side, min_side))


class AspectCrop:
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, x):
        width, height = TF.get_image_size(x)
        aspect = width / float(height)

        ideal_aspect = self.ratio

        if aspect > ideal_aspect:
            new_w = int(ideal_aspect * height)
            new_h = height
        else:
            new_h = int(width / ideal_aspect)
            new_w = width

        return TF.center_crop(x, (int(new_h), int(new_w)))
    
    
def zoom_crop(img, perc):
    if isinstance(img, Image.Image):
        w, h = img.size
    elif isinstance(img, torch.Tensor):
        _, h, w = img.shape
    else:
        raise TypeError(f"Cannot handle type {type(img)}.")

    return TF.center_crop(img, (int(h * perc), int(w * perc)))


class ShrinkCrop:
    def __init__(self, perc: float):
        if perc >= 1.0 or perc <= 0.0:
            raise ValueError("perc must be a value between 0 and 1.")
        self.perc = perc

    def __call__(self, img):
        return zoom_crop(img, self.perc)
    
    
class RandomZoomCrop:
    def __init__(self, perc_range: Tuple[float, float]):
        if not isinstance(perc_range, Sequence) or len(perc_range) != 2:
            raise TypeError("perc_range must be a 2 element sequence.")
        self.perc_range = perc_range

    def __call__(self, img):
        perc = random.uniform(*self.perc_range)
        return zoom_crop(img, perc)
