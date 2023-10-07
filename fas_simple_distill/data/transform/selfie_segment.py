from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
from enum import Enum

mp_selfie_segmentation = mp.solutions.selfie_segmentation


class InputType(Enum):
    PILLOW = 0
    ARRAY_RGB = 1
    ARRAY_BGR = 2


class SelfieSegmentation:
    def __init__(
        self, input_type = "ARRAY_RGB", model_selection=0, remove_outliers=True
    ) -> None:
        if input_type == "PILLOW":
            self.input_type = InputType.PILLOW
        elif input_type == "ARRAY_RGB":
            self.input_type = InputType.ARRAY_RGB
        elif input_type == "ARRAY_BGR":
            self.input_type = InputType.ARRAY_BGR
        else:
            raise TypeError("input_type should be 'PILLOW', 'ARRAY_RGB', or 'ARRAY_BGR'")
        
        self.model = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )
        self.remove_outliers = remove_outliers

    def __call__(self, input_img):
        input_img_copy = input_img
        
        input_img = np.asarray(input_img, dtype=np.uint8)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        
        try:
            _, seg_img = self.get_person_mask(input_img)
        except Exception as e:
            return input_img_copy
        
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(seg_img)

    def crop_person_mask(self, image: np.ndarray, seg_mask):
        BG_COLOR = (0, 0, 0, 0.0)
        condition = np.stack((seg_mask,) * 4, axis=-1) > 0.1
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        seg_image = np.where(condition, image, bg_image)
        return seg_image

    def get_person_mask(self, image: np.ndarray):
        image = self._handle_input(image)
        seg_mask = self._predict_seg_mask(image)
        seg_mask = self._postprocess_seg_mask(seg_mask)
        
        if not self.remove_outliers:
            seg_image = self.crop_person_mask(image, seg_mask)
            return seg_mask, seg_image

        seg_mask = self._remove_small_contours(seg_mask)
        seg_image = self.crop_person_mask(image, seg_mask)
        return seg_mask, seg_image

    def _remove_small_contours(self, seg_mask):
        seg_mask_uint8 = seg_mask * 255
        seg_mask_uint8 = seg_mask_uint8.astype(np.uint8)
        contours, _ = cv2.findContours(
            seg_mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        max_area = max([cv2.contourArea(c) for c in contours])

        for c in contours:
            area = cv2.contourArea(c)
            if area < max_area:
                cv2.drawContours(seg_mask, [c], -1, 0, -1)
            else:
                cv2.drawContours(seg_mask, [c], -1, 1, -1)

        return seg_mask

    def _postprocess_seg_mask(self, seg_mask):
        seg_mask = cv2.GaussianBlur(seg_mask, (19, 19), 0)
        _, seg_mask = cv2.threshold(seg_mask, 0.7, 1.0, cv2.THRESH_BINARY)
        seg_mask = cv2.morphologyEx(
            seg_mask, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8)
        )

        return seg_mask

    def _predict_seg_mask(self, image):
        image.flags.writeable = False
        seg_mask = self.model.process(image).segmentation_mask
        image.flags.writeable = True
        return seg_mask

    def _handle_input(self, image):
        if self.input_type == InputType.PILLOW:
            image = np.asarray(image)
        elif self.input_type == InputType.ARRAY_BGR:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
