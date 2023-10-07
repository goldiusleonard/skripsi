from torch.utils.data import IterDataPipe
import torchvision
from random import randint

class AddResizedImgIterDataPipe(IterDataPipe):
    def __init__(
        self,
        src_datapipe: IterDataPipe,
        img_key: str = ".png",
        size: int = 112,
    ) -> None:
        self.src_datapipe = src_datapipe
        self.img_key = img_key
        self._resize = torchvision.transforms.Resize(size)
    
    def __iter__(self):
        for data in self.src_datapipe:
            if not f"{self.img_key}_resized" in data.keys():
                data[f"{self.img_key}_resized"] = self._resize(data[self.img_key])
            yield data