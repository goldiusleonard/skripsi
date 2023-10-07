from random import randint
from torch.utils.data import IterDataPipe
from PIL import Image
import numpy as np
import cv2

_under_size_handling_methods = ["skip", "raise"]

class ExtractRandomPatchesIterDataPipe(IterDataPipe):
    def __init__(
        self,
        src_datapipe: IterDataPipe,
        img_key: str = ".png",
        num_patches: int = 9,
        window_size: int = 160,
        under_size_handling: str = "skip",

    ) -> None:
        self.src_datapipe = src_datapipe
        self.img_key = img_key
        self.num_patches = num_patches
        self.window_size = window_size
        self.x1_min = 0
        self.y1_min = 0

        if not under_size_handling in _under_size_handling_methods:
            raise RuntimeError(f"under_size_handling not in {_under_size_handling_methods}")
        
        self.under_size_handling = under_size_handling

    def __iter__(self):
        for data in self.src_datapipe:
            img = data[self.img_key]
            if isinstance(img, Image.Image):
                src_is_pil = True
                img = np.asarray(img)
            elif isinstance(img, np.ndarray):
                src_is_pil = False
            else:
                raise TypeError(f"Cannot handle image type {type(img)}")

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            width = img.shape[1]
            height = img.shape[0]

            if width < self.window_size or height < self.window_size:
                if self.under_size_handling == "skip":
                    continue
                elif self.under_size_handling == "raise":
                    raise RuntimeError(f"input image size is smaller than window size!")

            x1_max = width - self.window_size
            y1_max = height - self.window_size

            patches = []
            no_patch = False

            for i in range(self.num_patches):
                x1_patch = randint(self.x1_min, x1_max)
                y1_patch = randint(self.y1_min, y1_max)
                
                patch = img_bgr[y1_patch:y1_patch+self.window_size, x1_patch:x1_patch+self.window_size]

                if patch.shape[0] < self.window_size or patch.shape[0] < self.window_size:
                    if self.under_size_handling == "skip":
                        no_patch = True
                        break
                    elif self.under_size_handling == "raise":
                        raise RuntimeError(f"patch size is not the same with window size!")
                
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch = Image.fromarray(patch)

                patches.append(patch)

            if no_patch and self.under_size_handling == "skip":
                continue

            data[self.img_key] = patches
            yield data
