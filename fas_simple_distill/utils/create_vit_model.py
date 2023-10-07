from cvnets.models.classification import build_classification_model
from addict import Dict as Adict
import yaml
import collections
import argparse


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_model(yaml_path: str):
    with open(
        yaml_path,
        "r",
    ) as f:
        model_config = Adict(yaml.load(f, Loader=yaml.SafeLoader))
    model_config.freeze()
    model_config = flatten_dict(model_config, parent_key="", sep=".")
    model_config = argparse.Namespace(**model_config)

    model = build_classification_model(model_config)
    return model
