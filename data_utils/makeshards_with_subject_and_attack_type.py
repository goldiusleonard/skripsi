import argparse
from pathlib import Path
from typing import List, Union
import pandas as pd
import webdataset as wds


class InvalidProportion(Exception):
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-csv", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--maxcount", default=10, type=int)
    parser.add_argument("--start-subject", required=True, type=int)
    parser.add_argument(
        "--resample",
        nargs="?",
        default="off",
        const="off",
        choices=["off", "oversample", "stratified"],
    )
    parser.add_argument("--equal-shard", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--flip-label", action="store_true")
    parser.add_argument("pattern", type=str)
    parser.add_argument("dest", type=Path, help="destination path to write shards")
    args = parser.parse_args()

    if args.resample == "off" and args.equal_shard:
        parser.error("Must resample if shard is to be equal!")

    return args


def readfile(fname):
    with open(fname, "rb") as stream:
        return stream.read()


def stratify(
    df_data: pd.DataFrame,
    stratify_column_name: List[str],
    stratify_values: List[Union[int, str]],
    stratify_proportions: List[float],
    random_state=None,
):
    """Stratifies data according to the values and proportions passed in

    Args:
        df_data (DataFrame): source data
        stratify_column_name (str): The name of the single column in the dataframe
            that holds the data values that will be used to stratify the data
        stratify_values (list of int or str): A list of all of the potential values for stratifying
        stratify_proportions (list of float): A list of numbers representing the
            desired propotions for stratifying. The list values must add up to 1
            and must match the number of values in stratify_values.
        random_state (int, optional): sets the random_state. Defaults to None.

    Returns:
        DataFrame: a new dataframe based on df_data that has the
            new proportions representing the desired strategy for stratifying
    """
    df_stratified = pd.DataFrame(columns=df_data.columns)

    if sum(stratify_proportions) != 1:
        raise InvalidProportion("Proportion must add up to 1")

    pos = -1
    for i in range(len(stratify_values)):
        pos += 1
        if pos == len(stratify_values) - 1:
            ratio_len = len(df_data) - len(df_stratified)
        else:
            ratio_len = int(len(df_data) * stratify_proportions[i])

        df_filtered = df_data[df_data[stratify_column_name] == stratify_values[i]]
        df_temp = df_filtered.sample(
            replace=True, n=ratio_len, random_state=random_state
        )

        df_stratified = pd.concat([df_stratified, df_temp])

    return df_stratified


def oversample(data_df):
    max_size = data_df["label"].value_counts().max()
    lst = [data_df]
    for _, group in data_df.groupby("label"):
        lst.append(group.sample(max_size - len(group), replace=True))
    data_df_new = pd.concat(lst)

    return data_df_new


def spoof_type_to_int(spoof_type: str):
    spoof_type = spoof_type.lower()
    if spoof_type in ["real", "live"]:
        return 0

    if spoof_type in ["print", "print1", "print2", "printed", "paper", "photo"]:
        return 1

    if spoof_type in ["cutout"]:
        return 2

    if spoof_type in ["screen", "video", "replay", "video-replay1", "video-replay2"]:
        return 3

    raise RuntimeError(f"Unknown Spoof Type: {spoof_type}!")


def write_equal_shards(
    label_df: pd.DataFrame,
    pattern: str,
    data_root: Path,
    maxcount: int,
    start_subject: int,
):
    pos_df = label_df[label_df["label"] == 1]
    neg_df = label_df[label_df["label"] == 0]

    assert len(pos_df) == len(neg_df)
    assert maxcount % 2 == 0

    def _write_shard(_writer, _idx, _img, _lbldict):
        _key = f"{_idx:010d}"
        _sample = {"__key__": _key, "png": _img, "pickle": _lbldict}
        _writer.write(_sample)

    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for idx, (pos_row_dict, neg_row_dict) in enumerate(
            zip(
                pos_df.to_dict(orient="records"),
                neg_df.to_dict(orient="records"),
            )
        ):
            pos_fname: Path = data_root / pos_row_dict["path"]
            pos_image: bytes = readfile(pos_fname)
            pos_subject: str = str(
                int(str(pos_row_dict["path"]).split("/")[3]) - 1 + start_subject
            )
            pos_row_dict["subject"] = pos_subject

            del pos_row_dict["path"]
            _write_shard(sink, (idx * 2), pos_image, pos_row_dict)

            neg_fname: Path = data_root / neg_row_dict["path"]
            neg_image: bytes = readfile(neg_fname)
            neg_subject: str = str(
                int(str(neg_row_dict["path"]).split("/")[3]) - 1 + start_subject
            )
            neg_row_dict["subject"] = neg_subject

            del neg_row_dict["path"]
            _write_shard(sink, (idx * 2 + 1), neg_image, neg_row_dict)


def write_shards(
    label_df: pd.DataFrame,
    pattern: str,
    data_root: Path,
    maxcount: int,
    start_subject: int,
):
    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for idx, row_dict in enumerate(label_df.to_dict(orient="records")):
            fname: Path = data_root / row_dict["path"]
            image: bytes = readfile(fname)
            subject: str = str(
                int(str(row_dict["path"]).split("/")[3]) - 1 + start_subject
            )
            spoof_type: str = str(str(row_dict["path"]).split("/")[4])

            row_dict["subject"] = subject
            row_dict["spoof_type"] = str(spoof_type_to_int(spoof_type))
            del row_dict["path"]

            key = f"{idx:010d}"
            sample = {"__key__": key, "png": image, "pickle": row_dict}

            sink.write(sample)


def main():
    args = get_args()
    label_csv: Path = args.label_csv
    data_root: Path = args.data_root
    maxcount: int = args.maxcount
    resample: str = args.resample
    equal_shard: bool = args.equal_shard
    shuffle: bool = args.shuffle
    pattern: str = args.pattern
    flip_label: bool = args.flip_label
    dest: Path = args.dest
    start_subject = args.start_subject

    if not Path.exists(dest):
        Path.mkdir(dest)

    label_df = pd.read_csv(label_csv)

    if resample == "oversample":
        label_df = oversample(label_df)
    elif resample == "stratified":
        label_df = stratify(
            df_data=label_df,
            stratify_column_name="label",
            stratify_values=[0, 1],
            stratify_proportions=[0.5, 0.5],
        )

    if shuffle:
        label_df = label_df.sample(frac=1).reset_index(drop=True)

    if flip_label:
        label_df["label"] = 1 - label_df["label"]

    label_save_path = str(dest / "labels.csv")
    label_df.to_csv(label_save_path, index=False)

    dataset_detail_path = str(dest / f"{len(label_df)}")
    with open(dataset_detail_path, "w") as f:
        f.write("")

    pattern = str(dest / pattern)
    if equal_shard:
        write_equal_shards(label_df, pattern, data_root, maxcount, start_subject)
    else:
        write_shards(label_df, pattern, data_root, maxcount, start_subject)


if __name__ == "__main__":
    main()
