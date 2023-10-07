import sys

sys.path.append("F:/skripsi/FAS-Skripsi-4")

import pickle
from pathlib import Path
import cv2
import numpy as np
import PIL.Image as Image
from fas_simple_distill.model.maddg.DGFANet import FeatEmbedder, FeatExtractor
from fas_simple_distill.model.iresnet import iresnet18
from face_detection.main import FaceDetection, FaceSelectionMethod
from torchvision.transforms import ToTensor, Normalize, Resize
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
        scale: float = 1.0,
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
    def __init__(self, ckpt_path, device: str = "cuda", t=0.00001):
        model_ckpt = torch.load(ckpt_path, map_location="cpu")

        self.device = device
        self.feat_extractor = FeatExtractor(in_channels=6)
        self.feat_embedder = FeatEmbedder(embed_size=512, in_channels=128)

        self.feat_extractor.load_state_dict(model_ckpt["feat_extractor"])
        self.feat_embedder.load_state_dict(model_ckpt["feat_embedder"])

        self.feat_extractor.requires_grad_(False)
        self.feat_embedder.requires_grad_(False)
        self.feat_extractor.eval()
        self.feat_embedder.eval()
        self.feat_extractor.to(self.device)
        self.feat_embedder.to(self.device)

        self.fr_model = iresnet18()
        self.fr_model.load_state_dict(
            torch.load(
                "F:/skripsi/FAS-Skripsi-4/fas_simple_distill/weights/cosface_iresnet18.pth"
            )
        )
        self.fr_model.eval()
        self.fr_model.to(self.device)

        self._crop_face = crop_align_face()
        self._to_tensor = ToTensor()
        self._normalize = Normalize(
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225],
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        self.t = t
        self._resize = Resize(112)
        self.min_dist = []

    def remove_feat(self, fas_embs, fr_embs):
        embedding_size = fas_embs.size(1)
        batch = fas_embs.size(0)

        dist_matrices = None
        temp_dist_matrices = None

        for batch_index in range(int(batch)):
            if dist_matrices is None:
                dist_matrices = (
                    fas_embs[batch_index]
                    .expand(embedding_size, embedding_size)
                    .cuda()
                    .unsqueeze(0)
                )
            else:
                dist_matrices = torch.cat(
                    (
                        dist_matrices,
                        fas_embs[batch_index]
                        .expand(embedding_size, embedding_size)
                        .cuda()
                        .unsqueeze(0),
                    ),
                    dim=0,
                )

        for batch_index in range(int(batch)):
            if temp_dist_matrices is None:
                temp_dist_matrices = dist_matrices[batch_index].T.unsqueeze(0)
            else:
                temp_dist_matrices = torch.cat(
                    (temp_dist_matrices, dist_matrices[batch_index].T.unsqueeze(0)),
                    dim=0,
                )
        del dist_matrices

        dist_matrices = None
        for batch_index in range(int(batch)):
            if dist_matrices is None:
                dist_matrices = (
                    torch.sub(temp_dist_matrices[batch_index], fr_embs[batch_index])
                    .abs()
                    .unsqueeze(0)
                )
            else:
                dist_matrices = torch.cat(
                    (
                        dist_matrices,
                        torch.sub(temp_dist_matrices[batch_index], fr_embs[batch_index])
                        .abs()
                        .unsqueeze(0),
                    ),
                    dim=0,
                )

        min_dists = dist_matrices.min(dim=1)[0]
        self.min_dist.extend(min_dists.detach().cpu().numpy().tolist())

        new_fas_embs = fas_embs.clone().cuda()
        for batch_index in range(int(fas_embs.size(0))):
            new_fas_embs[batch_index] = torch.where(
                min_dists[batch_index] < self.t,
                fas_embs[batch_index] * min_dists[batch_index],
                fas_embs[batch_index],
            )
        return new_fas_embs

    def rgb2hsv_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        assert (
            rgb.dim() == 4 and rgb.shape[1] == 3
        ), "tensor shape should be like B x 3 x H x W"
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[
            cmax_idx == 0
        ]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[
            cmax_idx == 1
        ]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[
            cmax_idx == 2
        ]
        hsv_h[cmax_idx == 3] = 0.0
        hsv_h /= 6.0
        hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

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
        normalized_processed_data = self._resize(processed_data)
        normalized_processed_data = self._normalize(normalized_processed_data)
        return processed_data, normalized_processed_data

    def __call__(self, image: np.ndarray):
        try:
            input_tensor, normalized_input_tensor = self._preprocess_data(image)
            input_tensor = input_tensor[None, ...].to(self.device)
            normalized_input_tensor = normalized_input_tensor[None, ...].to(self.device)
        except Exception as e:
            return -1, -1
        hsv_input_tensor = self.rgb2hsv_torch(input_tensor).to(self.device)
        input_tensor = torch.cat((input_tensor, hsv_input_tensor), dim=1).to(
            self.device
        )
        _, featmap = self.feat_extractor(input_tensor)
        feat, _ = self.feat_embedder(featmap)
        fr_feat = self.fr_model(normalized_input_tensor)
        feat = self.remove_feat(feat, fr_feat)
        cls_out = self.feat_embedder.classifier(feat)
        score = cls_out.softmax(dim=1)[..., 1]

        return score.detach().cpu().numpy(), feat.detach().cpu().numpy()


def _main():
    # data_name = "oulu_npu"
    data_name = "replay_attack"
    # data_name = "msu_mfsd"
    # data_name = "casia_mfsd"
    data_root = Path(f"./{data_name}")

    liveness_model = LivenessModel(
        "evaluator/weights/maddg_no_triplet_feat_mining_OCMtoI.pth"
    )
    embedding_folder = "maddg_no_triplet_feat_mining_OCMtoI_remove_feat"

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

    # print(f"minimum distance: {min(liveness_model.min_dist[0])}")
    # print(f"maximum distance: {max(liveness_model.min_dist[0])}")
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
