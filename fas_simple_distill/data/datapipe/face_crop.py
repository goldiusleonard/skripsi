from typing import Union

import cv2
import numpy as np
from face_detection import FaceDetection, FaceSelectionMethod
from PIL import Image
from torch.utils.data import IterDataPipe
from torchvision.transforms import RandomHorizontalFlip

PICKLE_KEY = ".pickle"


_no_face_handling_methods = ["skip", "raise"]


class FaceCropFromDetsIterDataPipe(IterDataPipe):
    def __init__(
        self,
        src_datapipe: IterDataPipe,
        anno_key: str = "faces",
        img_key: str = ".png",
        loose_factor: float = 1.0,
        select_method: Union[str, FaceSelectionMethod] = FaceSelectionMethod.AREA,
        no_face_handling: str = "skip",
    ) -> None:
        self.src_datapipe = src_datapipe
        self.anno_key = anno_key
        self.img_key = img_key
        self.loose_factor = loose_factor

        if isinstance(select_method, str):
            select_method = FaceSelectionMethod[select_method]
        self.select_method = select_method

        if no_face_handling not in _no_face_handling_methods:
            raise ValueError(
                f"Got {no_face_handling}, no_face_handling must "
                f"be one of these values: {_no_face_handling_methods}"
            )
        self.no_face_handling = no_face_handling

    def __iter__(self):
        for data in self.src_datapipe:
            anno = data[PICKLE_KEY][self.anno_key]
            dets = anno["dets"]
            angle = anno["ang"]

            if len(dets) < 1:
                if self.no_face_handling == "skip":
                    continue
                else:
                    raise RuntimeError("No face detected!")

            img = data[self.img_key]
            if isinstance(img, Image.Image):
                src_is_pil = True
                img = np.asarray(img)
            elif isinstance(img, np.ndarray):
                src_is_pil = False
            else:
                raise TypeError(f"Cannot handle image type {type(img)}")

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img_cropped_bgr, angle = FaceDetection.crop_single_face(
                img=img_bgr,
                dets=dets,
                curr_angle=angle,
                loose_factor=self.loose_factor,
                select_method=self.select_method,
            )

            # img_cropped = cv2.cvtColor(img_cropped_bgr, cv2.COLOR_BGR2RGB)
            # img_cropped = Image.fromarray(img_cropped)

            data[self.img_key] = img_cropped_bgr
            yield data


class FaceCropFromDetsFRIterDataPipe(IterDataPipe):
    def __init__(
        self,
        src_datapipe: IterDataPipe,
        anno_key: str = "faces",
        img_key: str = ".png",
        loose_factor: float = 1.0,
        select_method: Union[str, FaceSelectionMethod] = FaceSelectionMethod.AREA,
        no_face_handling: str = "skip",
    ) -> None:
        self.src_datapipe = src_datapipe
        self.anno_key = anno_key
        self.img_key = img_key
        self.loose_factor = loose_factor

        if isinstance(select_method, str):
            select_method = FaceSelectionMethod[select_method]
        self.select_method = select_method

        if no_face_handling not in _no_face_handling_methods:
            raise ValueError(
                f"Got {no_face_handling}, no_face_handling must "
                f"be one of these values: {_no_face_handling_methods}"
            )
        self.no_face_handling = no_face_handling

    def __iter__(self):
        for data in self.src_datapipe:
            anno = data[PICKLE_KEY][self.anno_key]
            dets = anno["dets"]
            angle = anno["ang"]

            if len(dets) < 1:
                if self.no_face_handling == "skip":
                    continue
                else:
                    raise RuntimeError("No face detected!")

            img = data[self.img_key]
            # if isinstance(img, Image.Image):
            #     src_is_pil = True
            #     img = np.asarray(img)
            # elif isinstance(img, np.ndarray):
            #     src_is_pil = False
            # else:
            #     raise TypeError(f"Cannot handle image type {type(img)}")

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img_cropped_bgr_fas, angle = FaceDetection.crop_single_face(
                img=img_bgr,
                dets=dets,
                curr_angle=angle,
                loose_factor=self.loose_factor,
                select_method=self.select_method,
            )

            img_cropped_bgr_fr, angle = FaceDetection.crop_single_face(
                img=img_bgr,
                dets=dets,
                curr_angle=angle,
                loose_factor=1.0,
                select_method=self.select_method,
            )

            img_cropped_bgr_fas = cv2.cvtColor(img_cropped_bgr_fas, cv2.COLOR_BGR2RGB)
            img_cropped_bgr_fas = Image.fromarray(img_cropped_bgr_fas)
            img_cropped_bgr_fr = cv2.cvtColor(img_cropped_bgr_fr, cv2.COLOR_BGR2RGB)
            img_cropped_bgr_fr = Image.fromarray(img_cropped_bgr_fr)

            # data[self.img_key] = img_cropped_bgr
            del data[self.img_key]
            data[f"{self.img_key}_fas"] = img_cropped_bgr_fas
            data[f"{self.img_key}_fr"] = img_cropped_bgr_fr
            yield data


class FaceCropFromDetsRandomHorizontalFlipIterDataPipe(IterDataPipe):
    def __init__(
        self,
        src_datapipe: IterDataPipe,
        anno_key: str = "faces",
        img_key: str = ".png",
        loose_factor: float = 1.0,
        select_method: Union[str, FaceSelectionMethod] = FaceSelectionMethod.AREA,
        no_face_handling: str = "skip",
        prob: float = 0.5,
    ) -> None:
        self.src_datapipe = src_datapipe
        self.anno_key = anno_key
        self.img_key = img_key
        self.loose_factor = loose_factor

        if isinstance(select_method, str):
            select_method = FaceSelectionMethod[select_method]
        self.select_method = select_method

        if no_face_handling not in _no_face_handling_methods:
            raise ValueError(
                f"Got {no_face_handling}, no_face_handling must "
                f"be one of these values: {_no_face_handling_methods}"
            )
        self.no_face_handling = no_face_handling
        self._rhf = RandomHorizontalFlip(prob)

    def __iter__(self):
        for data in self.src_datapipe:
            anno = data[PICKLE_KEY][self.anno_key]
            dets = anno["dets"]
            angle = anno["ang"]

            if len(dets) < 1:
                if self.no_face_handling == "skip":
                    continue
                else:
                    raise RuntimeError("No face detected!")

            img = data[self.img_key]
            # if isinstance(img, Image.Image):
            #     src_is_pil = True
            #     img = np.asarray(img)
            # elif isinstance(img, np.ndarray):
            #     src_is_pil = False
            # else:
            #     raise TypeError(f"Cannot handle image type {type(img)}")

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            img_pil = self._rhf(img_pil)
            img_bgr = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

            img_cropped_bgr_fas, angle = FaceDetection.crop_single_face(
                img=img_bgr,
                dets=dets,
                curr_angle=angle,
                loose_factor=self.loose_factor,
                select_method=self.select_method,
            )

            img_cropped_bgr_fr, angle = FaceDetection.crop_single_face(
                img=img_bgr,
                dets=dets,
                curr_angle=angle,
                loose_factor=1.0,
                select_method=self.select_method,
            )

            img_cropped_bgr_fas = cv2.cvtColor(img_cropped_bgr_fas, cv2.COLOR_BGR2RGB)
            img_cropped_bgr_fas = Image.fromarray(img_cropped_bgr_fas)
            img_cropped_bgr_fr = cv2.cvtColor(img_cropped_bgr_fr, cv2.COLOR_BGR2RGB)
            img_cropped_bgr_fr = Image.fromarray(img_cropped_bgr_fr)

            # data[self.img_key] = img_cropped_bgr
            del data[self.img_key]
            data[f"{self.img_key}_fas"] = img_cropped_bgr_fas
            data[f"{self.img_key}_fr"] = img_cropped_bgr_fr
            yield data


class FaceAlignFromDetsIterDataPipe(IterDataPipe):
    def __init__(
        self,
        src_datapipe: IterDataPipe,
        anno_key: str = "faces",
        img_key: str = ".png",
        scale: float = 1.0,
        crop_size: int = 224,
        select_method: Union[str, FaceSelectionMethod] = FaceSelectionMethod.CENTER,
        no_face_handling: str = "skip",
    ) -> None:
        self.src_datapipe = src_datapipe
        self.anno_key = anno_key
        self.img_key = img_key
        self.scale = scale
        self.crop_size = crop_size

        if isinstance(select_method, str):
            select_method = FaceSelectionMethod[select_method]
        self.select_method = select_method

        if no_face_handling not in _no_face_handling_methods:
            raise ValueError(
                f"Got {no_face_handling}, no_face_handling must "
                f"be one of these values: {_no_face_handling_methods}"
            )
        self.no_face_handling = no_face_handling

    def __iter__(self):
        for data in self.src_datapipe:
            anno = data[PICKLE_KEY][self.anno_key]
            dets = anno["dets"]
            angle = anno["ang"]

            if len(dets) < 1:
                if self.no_face_handling == "skip":
                    continue
                else:
                    raise RuntimeError("No face detected!")

            img = data[self.img_key]
            if isinstance(img, Image.Image):
                src_is_pil = True
                img = np.asarray(img)
            elif isinstance(img, np.ndarray):
                src_is_pil = False
            else:
                raise TypeError(f"Cannot handle image type {type(img)}")

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img_aligned_bgr, angle = FaceDetection.align_single_face(
                img=img_bgr,
                dets=dets,
                curr_angle=angle,
                crop_size=self.crop_size,
                scale=self.scale,
                select_method=self.select_method,
            )

            # img_aligned = cv2.cvtColor(img_aligned_bgr, cv2.COLOR_BGR2RGB)
            # img_aligned = Image.fromarray(img_aligned)

            data[self.img_key] = img_aligned_bgr
            yield data


class FaceAlignFromDetsFRIterDataPipe(IterDataPipe):
    def __init__(
        self,
        src_datapipe: IterDataPipe,
        anno_key: str = "faces",
        img_key: str = ".png",
        scale: float = 1.0,
        crop_size: int = 224,
        select_method: Union[str, FaceSelectionMethod] = FaceSelectionMethod.CENTER,
        no_face_handling: str = "skip",
    ) -> None:
        self.src_datapipe = src_datapipe
        self.anno_key = anno_key
        self.img_key = img_key
        self.scale = scale
        self.crop_size = crop_size

        if isinstance(select_method, str):
            select_method = FaceSelectionMethod[select_method]
        self.select_method = select_method

        if no_face_handling not in _no_face_handling_methods:
            raise ValueError(
                f"Got {no_face_handling}, no_face_handling must "
                f"be one of these values: {_no_face_handling_methods}"
            )
        self.no_face_handling = no_face_handling

    def __iter__(self):
        for data in self.src_datapipe:
            anno = data[PICKLE_KEY][self.anno_key]
            dets = anno["dets"]
            angle = anno["ang"]

            if len(dets) < 1:
                if self.no_face_handling == "skip":
                    continue
                else:
                    raise RuntimeError("No face detected!")

            img = data[self.img_key]
            # if isinstance(img, Image.Image):
            #     src_is_pil = True
            #     img = np.asarray(img)
            # elif isinstance(img, np.ndarray):
            #     src_is_pil = False
            # else:
            #     raise TypeError(f"Cannot handle image type {type(img)}")

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img_aligned_bgr_fas, _ = FaceDetection.align_single_face(
                img=img_bgr,
                dets=dets,
                curr_angle=angle,
                crop_size=self.crop_size,
                scale=self.scale,
                select_method=self.select_method,
            )

            img_aligned_bgr_fr, _ = FaceDetection.align_single_face(
                img=img_bgr,
                dets=dets,
                curr_angle=angle,
                crop_size=112,
                scale=1.0,
                select_method=self.select_method,
            )

            img_aligned_bgr_fas = cv2.cvtColor(img_aligned_bgr_fas, cv2.COLOR_BGR2RGB)
            img_aligned_bgr_fas = Image.fromarray(img_aligned_bgr_fas)
            img_aligned_bgr_fr = cv2.cvtColor(img_aligned_bgr_fr, cv2.COLOR_BGR2RGB)
            img_aligned_bgr_fr = Image.fromarray(img_aligned_bgr_fr)

            # data[self.img_key] = img_aligned_bgr
            del data[self.img_key]
            data[f"{self.img_key}_fas"] = img_aligned_bgr_fas
            data[f"{self.img_key}_fr"] = img_aligned_bgr_fr
            yield data


class FaceAlignFromDetsRandomHorizontalFlipIterDataPipe(IterDataPipe):
    def __init__(
        self,
        src_datapipe: IterDataPipe,
        anno_key: str = "faces",
        img_key: str = ".png",
        scale: float = 1.0,
        crop_size: int = 224,
        select_method: Union[str, FaceSelectionMethod] = FaceSelectionMethod.CENTER,
        no_face_handling: str = "skip",
        prob: float = 0.5,
    ) -> None:
        self.src_datapipe = src_datapipe
        self.anno_key = anno_key
        self.img_key = img_key
        self.scale = scale
        self.crop_size = crop_size

        if isinstance(select_method, str):
            select_method = FaceSelectionMethod[select_method]
        self.select_method = select_method

        if no_face_handling not in _no_face_handling_methods:
            raise ValueError(
                f"Got {no_face_handling}, no_face_handling must "
                f"be one of these values: {_no_face_handling_methods}"
            )
        self.no_face_handling = no_face_handling
        self._rhf = RandomHorizontalFlip(prob)

    def __iter__(self):
        for data in self.src_datapipe:
            anno = data[PICKLE_KEY][self.anno_key]
            dets = anno["dets"]
            angle = anno["ang"]

            if len(dets) < 1:
                if self.no_face_handling == "skip":
                    continue
                else:
                    raise RuntimeError("No face detected!")

            img = data[self.img_key]
            # if isinstance(img, Image.Image):
            #     src_is_pil = True
            #     img = np.asarray(img)
            # elif isinstance(img, np.ndarray):
            #     src_is_pil = False
            # else:
            #     raise TypeError(f"Cannot handle image type {type(img)}")

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            img_pil = self._rhf(img_pil)
            img_bgr = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

            img_aligned_bgr_fas, _ = FaceDetection.align_single_face(
                img=img_bgr,
                dets=dets,
                curr_angle=angle,
                crop_size=self.crop_size,
                scale=self.scale,
                select_method=self.select_method,
            )

            img_aligned_bgr_fr, _ = FaceDetection.align_single_face(
                img=img_bgr,
                dets=dets,
                curr_angle=angle,
                crop_size=112,
                scale=1.0,
                select_method=self.select_method,
            )

            img_aligned_bgr_fas = cv2.cvtColor(img_aligned_bgr_fas, cv2.COLOR_BGR2RGB)
            img_aligned_bgr_fas = Image.fromarray(img_aligned_bgr_fas)
            img_aligned_bgr_fr = cv2.cvtColor(img_aligned_bgr_fr, cv2.COLOR_BGR2RGB)
            img_aligned_bgr_fr = Image.fromarray(img_aligned_bgr_fr)

            # data[self.img_key] = img_aligned_bgr
            del data[self.img_key]
            data[f"{self.img_key}_fas"] = img_aligned_bgr_fas
            data[f"{self.img_key}_fr"] = img_aligned_bgr_fr
            yield data
