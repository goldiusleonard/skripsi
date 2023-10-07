from face_detection.main import FaceDetection
from PIL import Image
import numpy as np
import cv2

class AlignFace:
    def __init__(self, size, scale, use_cuda=True, use_onnx=False, noface_fail=False) -> None:
        self.fd = FaceDetection(use_cuda=use_cuda, use_onnx=use_onnx)
        self.size = size
        self.scale = scale
        self.noface_fail = noface_fail
    
    def __call__(self, input_img):
        input_is_pil = False
        
        x = input_img
        if isinstance(x, Image.Image):
            input_is_pil = True
            x = np.asarray(x, dtype=np.uint8)

        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        dets, ang = self.fd.predict(x)
        
        if len(dets) < 1:
            if self.noface_fail:
                raise RuntimeError("No face detected!")

            return input_img
        
        x_aligned, _ = self.fd.align_single_face(
            x, dets, ang, crop_size=self.size, scale=self.scale
        )

        
        x_aligned = cv2.cvtColor(x_aligned, cv2.COLOR_BGR2RGB)
        if input_is_pil:
            return Image.fromarray(x_aligned)

        return x_aligned