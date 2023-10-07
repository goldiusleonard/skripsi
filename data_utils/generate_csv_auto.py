import csv
import os
from pathlib import Path
import argparse

exts = ['.png', '.jpg', '.jpeg']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--csv-save-path", required=True, type=Path)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    root_path = Path(str(args.data_root))
    class_list = ["live", "spoof"]

    for cls in class_list:
        f = open(os.path.join(args.csv_save_path.parent, args.csv_save_path.stem + f"_{cls}.csv"), 'w', newline='', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(["path", "label"])

        for img_path in root_path.rglob("*"):
            if not img_path.suffix in exts:
                continue

            if "live" in str(img_path).lower() or "real" in str(img_path).lower():
                label = "1"
            else:
                label = "0"

            if cls == "live" and label == "0" or cls == "spoof" and label == "1":
                continue

            img_path = Path(*img_path.parts[1:])
            writer.writerow([str(img_path).replace("\\", "/"), label])

        f.close()

if __name__ == "__main__":
    main()