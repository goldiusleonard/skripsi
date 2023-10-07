import pandas as pd

def webdataset_label_transform(annodict: dict):
    lbl = annodict.get("label")
    if lbl is None:
        lbl = annodict.get("labels")

    if lbl is None:
        raise RuntimeError(
            f"Cannot get label key, available keys are {annodict.keys()}"
        )

    return int(lbl)

def webdataset_label_str_transform_with_dict(annodict: dict):
    lbl = annodict.get("label")
    if lbl is None:
        lbl = annodict.get("labels")

    if lbl is None:
        raise RuntimeError(
            f"Cannot get label key, available keys are {annodict.keys()}"
        )

    label_int_dict = pd.read_csv("/opt/ml/code/dict/dict_labels_verified_df.csv")
    temp_df = label_int_dict.loc[label_int_dict["labels_str"] == lbl]
    label_int = temp_df["labels_int"]
    return int(label_int)
