import sys


import sys

sys.path.append("F:/skripsi/FAS-Skripsi-4")

import pickle
from pathlib import Path
import cv2
import numpy as np
import PIL.Image as Image
from fas_simple_distill.model.divt.divt_mobilevit_v2 import DG_model
from face_detection.main import FaceDetection, FaceSelectionMethod
from torchvision.transforms import ToTensor, Normalize
import torch
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.remove("F:/skripsi/FAS-Skripsi-4")

VALID_SPOOF_LABELS = {
    "spoof",
    "print",
    "screen",
    "cutout",
    "half_face",
    "mask",
    "paper",
    "printed",
    "Face Mask",
    "Full Person",
    "Half Bottom",
    "Half Top",
    "Printed Glossy",
    "Printed Laserjet",
    "Printed Matte",
    "Printed Matte (Wrong Side)",
    "Printed Plain",
    "Printed Silky",
    "Screen",
    "printed_plain",
    "printed_matte",
    "printed_matte_ws",
    "printed_glossy",
    "face_mask",
    "printed_matte_(wrong_side)",
    "printed_silky",
    "full_person",
    "half_bottom",
    "half_top",
    "printed_laserjet",
}

VALID_LIVE_LABELS = {"live", "real"}


class crop_align_face:
    def __init__(
        self,
        use_cuda: bool = True,
        no_rotate: bool = True,
        crop_size: int = 256,
        scale: float = 0.75,
        select_method=FaceSelectionMethod.AREA,
    ) -> None:
        use_onnx = not use_cuda
        self.fd = FaceDetection(use_cuda, no_rotate, use_onnx)
        self.crop_size = crop_size
        self.scale = scale
        self.select_method = select_method

    def __call__(self, x):
        if isinstance(x, Image.Image):
            input_is_pil = True
            x = np.array(x)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        else:
            input_is_pil = False

        dets, angle = self.fd.predict(x)
        x_crop, _ = self.fd.align_single_face(
            x, dets, angle, self.crop_size, self.scale, self.select_method
        )

        if input_is_pil:
            x_crop = cv2.cvtColor(x_crop, cv2.COLOR_BGR2RGB)
            x_crop = Image.fromarray(x_crop)

        return x_crop


class LivenessModel:
    def __init__(self, ckpt_path, device: str = "cuda"):
        model_ckpt = torch.load(ckpt_path, map_location="cpu")

        self.device = device
        self.dg_model = DG_model(
            "F:/skripsi/FAS-Skripsi-4/fas_simple_distill/model/mobilevit_config/mobilevitv2-1.0.yaml",
            num_classes=2,
        )

        model_state_dict = model_ckpt["model"]

        self.dg_model.load_state_dict(model_state_dict)
        self.dg_model.eval()
        self.dg_model.requires_grad_(False)
        self.dg_model.to(self.device)

        self._crop_face = crop_align_face()
        self._to_tensor = ToTensor()
        self._normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            # mean=[0.5, 0.5, 0.5],
            # std=[0.5, 0.5, 0.5],
        )

    def _preprocess_data(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a numpy image to be input tensor

        Args:
            image (np.ndarray): A BGR image (H, W, C)

        Returns:
            np.ndarray: Preprocessed input tensor (C, H, W)
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_data = Image.fromarray(rgb_image)
        processed_data = self._crop_face(processed_data)
        processed_data = self._to_tensor(processed_data)
        processed_data = self._normalize(processed_data)
        return processed_data

    def __call__(self, image: np.ndarray):
        try:
            input_tensor = self._preprocess_data(image)[None, ...].to(self.device)
        except Exception as e:
            return -1, -1
        cls_out, feat = self.dg_model(input_tensor)
        score = cls_out.softmax(dim=1)[..., 1]

        return score.detach().cpu().numpy(), feat.detach().cpu().numpy()


def _main():
    data_name = "oulu_npu"
    # data_name = "replay_attack"
    # data_name = "msu_mfsd"
    # data_name = "casia_mfsd"
    data_root = Path(f"F:/skripsi/datasets/1_frame_datasets/{data_name}")

    liveness_model = LivenessModel(
        "evaluator/weights/divt_bs_60_binary_less_tight_OCItoM.pth"
    )
    embedding_folder = "divt_bs_60_binary_less_tight_OCItoM"

    labels = []
    main_embs = []
    main_scores = []
    img_path_embed = []
    image_paths = (
        list(data_root.rglob("*.png"))
        + list(data_root.rglob("*.jpg"))
        + list(data_root.rglob("*.jpeg"))
    )

    # i = 0
    for img_path in tqdm(image_paths):
        # if i == 500:
        #     break
        if "live" in str(img_path).lower() or "real" in str(img_path).lower():
            label = 1
        else:
            label = 0

        img = cv2.imread(str(img_path))

        main_score, main_emb = liveness_model(img)

        if main_score == -1 and main_emb == -1:
            continue

        main_embs.append(main_emb)
        main_scores.append(main_score)
        labels.append(label)
        img_path_embed.append(img_path)

        # i += 1

    main_embs = np.concatenate(main_embs, axis=0)
    main_scores = np.concatenate(main_scores, axis=0)

    save_data = {
        "main_embs": main_embs,
        "main_scores": main_scores,
        "labels": labels,
        "img_paths": img_path_embed,
    }
    dst_path = f"./embedding_data/{embedding_folder}/"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    with open(dst_path + f"{data_name}.pkl", "wb") as pklfile:
        pickle.dump(save_data, pklfile)


if __name__ == "__main__":
    _main()
