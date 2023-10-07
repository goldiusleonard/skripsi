from pathlib import Path
from tqdm import tqdm
import cv2
import os

root_path1 = "F:/skripsi/FAS-Skripsi-4/results/divt_mobilevits_OMItoC/casia_mfsd/FP"
root_path2 = (
    "F:/skripsi/FAS-Skripsi-4/results/divt_mobilevits_oulu_msu_OMItoC/casia_mfsd/FP"
)

img_paths1 = list(Path(root_path1).rglob("*.png"))
img_paths2 = list(Path(root_path2).rglob("*.png"))

save_path = "F:/skripsi/diff/FP"

if len(img_paths1) > len(img_paths2):
    img_name2_list = []
    for img_path2 in img_paths2:
        img_name2_list.append(Path(img_path2).stem)

    for img_path1 in tqdm(img_paths1):
        img = cv2.imread(str(img_path1))
        img_name1 = Path(img_path1).stem
        if img_name1 in img_name2_list:
            continue
        cv2.imwrite(os.path.join(save_path, f"{img_name1}.png"), img)
