from typing import Any, Dict
import cv2
from torch.utils.data import IterDataPipe
import numpy as np
from PIL import Image

# pylint: disable=abstract-method
class DictBlankImageSkipperDP(IterDataPipe):
    def __init__(self, src_datapipe, img_key=".png") -> None:
        super().__init__()
        self.src_datapipe = src_datapipe
        self.img_key = img_key

    def __iter__(self):
        for data in self.src_datapipe:
            if np.asarray(data[self.img_key]).std() < 1e-3:
                continue

            yield data


# pylint: disable=abstract-method
class DictToPILIterDataPipe(IterDataPipe):
    def __init__(self, src_datapipe, img_key=".png") -> None:
        super().__init__()
        self.src_datapipe = src_datapipe
        self.img_key = img_key

    def __iter__(self):
        for data in self.src_datapipe:
            img = cv2.cvtColor(data[self.img_key], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            data[self.img_key] = img

            yield data


class DictExtractSubjectIterDataPipe(IterDataPipe):
    def __init__(self, src_datapipe, lbl_key=".pickle") -> None:
        super().__init__()
        self.src_datapipe = src_datapipe
        self.lbl_key = lbl_key

    def __iter__(self):
        for data in self.src_datapipe:
            annotations: Dict[str, Any] = data[self.lbl_key]
            subject = annotations["subject"]
            data["subject"] = int(subject)

            yield data


class DictExtractSpoofTypeIterDataPipe(IterDataPipe):
    def __init__(self, src_datapipe, lbl_key=".pickle") -> None:
        super().__init__()
        self.src_datapipe = src_datapipe
        self.lbl_key = lbl_key

    def __iter__(self):
        for data in self.src_datapipe:
            annotations: Dict[str, Any] = data[self.lbl_key]
            spoof_type = annotations["spoof_type"]
            data["spoof_type"] = int(spoof_type)

            yield data
