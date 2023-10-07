import torch
from typing import Optional
from PIL import Image
import numpy as np
import os
from .model.resnet_ssdg.ssdg_resnet_baseline_no_relu import DG_model
import torchvision.transforms.functional as TF

class SpoofDetection():
    def __init__(self, checkpoint):
        self.model = DG_model()
        self.load_checkpoint(checkpoint + ".pth")

    def load_checkpoint(self, checkpoint: Optional[str] = None):
        ckpt_dict = torch.load(
            "./face_anti_spoofing/weights/" + checkpoint, map_location="cpu"
        )     

        self.model.load_state_dict(ckpt_dict["model"])
    
    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        img_transformed = TF.to_tensor(img)
        img_transformed = TF.normalize(img_transformed,
                                    mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225))
        return img_transformed

    def preprocess_image_alter(self, img: np.ndarray) -> torch.Tensor:
        img_transformed = TF.to_tensor(img)
        img_transformed = TF.normalize(img_transformed,
                                    mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5))
        return img_transformed
    
    def predict(self, img):
        classifier_out, _ = self.model(img.unsqueeze(0))
        preds = classifier_out.softmax(dim=1)[..., 1]
        return preds