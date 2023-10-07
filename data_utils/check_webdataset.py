from argparse import ArgumentParser
from typing import Any, Dict, Union
from pathlib import Path
from glob import glob
import io

import cv2
import numpy as np
import face_detection
from datatools.torch_data.utils import decode_webdataset_pil
from datatools.torch_data.webdataset import get_webdataset_datapipe
from PIL import Image
import webdataset as wds


def _parse_args():
    parser = ArgumentParser(
        description="Utility to unpack webdataset files to original folder structure."
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Url to webdataset shards, supports braceexpand.",
    )

    args = parser.parse_args()

    return args


def get_original_tarkey(key: str):
    key_parts = Path(key).parts
    start_idx = -1
    for i, part in enumerate(key_parts, 1):
        if part.endswith(".tar"):
            start_idx = i
            break

    if start_idx == -1:
        raise RuntimeError(f"Error getting original tarkey from key {key}")

    orig_tarkey = str(Path().joinpath(*key_parts[start_idx:]))
    return orig_tarkey


def process_webdataset(
    url: str,
):
    wds_datapipe = get_webdataset_datapipe(
        url,
        decoder_fun=decode_webdataset_pil,
        seed=None,
        shuffle_per_epoch=False,
        split_by_rank=False,
        split_by_worker=False,
    )

    total_data = 0
    real_data = 0
    photo_data = 0
    cutout_data = 0
    video_data = 0

    for data in wds_datapipe:
        orig_tarkey = get_original_tarkey(data["__key__"])
        annotations: Dict[str, Any] = data[".pickle"]
        # img: Image.Image = data[".png"]
        attack_type = annotations["spoof_type"]

        if attack_type == "0":
            real_data += 1
        elif attack_type == "1":
            photo_data += 1
        elif attack_type == "2":
            cutout_data += 1
        elif attack_type == "3":
            video_data += 1
        else:
            print(attack_type)
            raise RuntimeError("Unknown attack type!")

        # img_bytes = io.BytesIO()
        # img.save(img_bytes, format="PNG")
        # to_write = {"__key__": orig_tarkey, "png": img, "pickle": annotations}
        # shard_writer.write(to_write)

        total_data += 1

    print(f"Real Data: {real_data}")
    print(f"Photo Data: {photo_data}")
    print(f"Cutout Data: {cutout_data}")
    print(f"Video Data: {video_data}")
    print(f"Total data: {total_data}")


def _main():
    args = _parse_args()

    print(args)
    urls = args.url
    if urls is None:
        urls = glob(args.glob, recursive=args.glob_recursive)

    process_webdataset(
        urls,
    )


if __name__ == "__main__":
    _main()
