import os
from pathlib import Path
import pickle
import cv2
from tqdm import tqdm

TYPES = ["TP", "TN", "FP", "FN"]

pkl_path = "embedding_data/divt_mobilevits_OMItoC/casia_mfsd.pkl"
threshold = 0.008838022

f = open(pkl_path, "rb")

pkl_data = pickle.load(f)

f.close()

scores = pkl_data["main_scores"]
img_paths = pkl_data["img_paths"]
labels = pkl_data["labels"]

model_name = pkl_path.split("/")[1]
dataset_name = Path(pkl_path).stem

for TYPE in TYPES:
    if not os.path.exists(f"./results/{model_name}/{dataset_name}/{TYPE}"):
        os.makedirs(f"./results/{model_name}/{dataset_name}/{TYPE}")

i = 1

for score, img_path, label in tqdm(zip(scores, img_paths, labels)):
    img = cv2.imread(str(img_path))
    if label == 1:
        if score >= threshold:
            cv2.imwrite(
                os.path.join(
                    f"./results/{model_name}/{dataset_name}/TP",
                    f"{str(img_path.stem)}.png",
                ),
                img,
            )
        else:
            cv2.imwrite(
                os.path.join(
                    f"./results/{model_name}/{dataset_name}/FN",
                    f"{str(img_path.stem)}.png",
                ),
                img,
            )
    else:
        if score < threshold:
            cv2.imwrite(
                os.path.join(
                    f"./results/{model_name}/{dataset_name}/TN",
                    f"{str(img_path.stem)}.png",
                ),
                img,
            )
        else:
            cv2.imwrite(
                os.path.join(
                    f"./results/{model_name}/{dataset_name}/FP",
                    f"{str(img_path.stem)}.png",
                ),
                img,
            )
    i += 1
