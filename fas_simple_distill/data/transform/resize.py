from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image


class ResizeOCV:
    def __init__(self, size):
        self.size = self._to_tuple(size)

    def _to_tuple(self, size):
        if isinstance(size, (int, float)):
            return (size, size)
        elif isinstance(size, (tuple, list)):
            return size
        else:
            raise ValueError("Invalid type", type(size))

    def __call__(self, x: Union[torch.Tensor, np.ndarray, Image.Image]):
        return_type = "numpy"

        if isinstance(x, Image.Image):
            x = np.asarray(x)
            return_type = "pil"
        elif isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            return_type = "tensor"
        elif not isinstance(x, np.ndarray):
            raise RuntimeError(f"Cannot handle {type(x)}.")

        if len(x.shape) != 3:
            raise RuntimeError(f"Cannot handle data in shape {x.shape}")

        try:
            index_dim = list(x.shape).index(3)
        except ValueError as e:
            raise RuntimeError(
                "Channel dim not found, make sure there is color dim with shape 3."
            ) from e

        if index_dim == 0:
            x = np.transpose(x, (1, 2, 0))
        elif index_dim != 2:
            raise RuntimeError("Invalid dim order, channel dim is in middle.")

        return_float = False
        if x.dtype == "float32":
            x = np.clip(x, 0, 1)
            x *= 255.0
            x = x.astype(np.uint8)
            return_float = True
        elif x.dtype != "uint8":
            raise RuntimeError(f"Cannot handle {x.dtype} arrays.")

        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        x = cv2.resize(x, self.size, interpolation=cv2.INTER_CUBIC)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.ascontiguousarray(x)

        if return_float:
            x = x.astype(np.float32)
            x /= 255.0

        if index_dim == 0:
            x = np.transpose(x, (2, 0, 1))
            x = np.ascontiguousarray(x)

        if return_type == "pil":
            return Image.fromarray(x)
        elif return_type == "tensor":
            return torch.from_numpy(x)

        return x
