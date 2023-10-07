import sys
from typing import Any, Tuple
from collections.abc import Mapping
from argparse import Namespace, ArgumentParser

import yaml
from addict import Dict as Adict

from fas_simple_distill.utils.raytune.raytune import parse_tuning_config

from .general import flatten_dict, strtobool


def getconfig(conf, n, delim="."):
    n = n.split(delim)
    if n[0] == "":
        return conf
    if isinstance(conf, dict):
        return getconfig(conf.__getattr__(n[0]), delim.join(n[1:]), delim=delim)
    else:
        return conf


def setconfig(nv, v, conf, n, delim="."):
    n = n.split(delim)
    if len(n) == 1 and n[0] != "":
        conf[nv] = v
        return conf[nv]
    if isinstance(conf, dict):
        return getconfig(conf.__getattr__(n[0]), delim.join(n[1:]), delim=delim)
    else:
        return conf


def setconfig_dict(
    conf_dict: dict, name: str, new_val: Any, delim: str = ".", verbose=False
):
    if not isinstance(conf_dict, dict):
        return

    name = name.split(delim)
    first_key = name[0]
    last_key = name[-1]
    for k, v in conf_dict.items():
        if len(name) == 1 and last_key != "" and k == last_key:
            if verbose:
                old_val = conf_dict[last_key]
                if conf_dict[last_key] != new_val:
                    print(
                        f"Overriding value of {old_val} ({type(old_val)}) with {new_val} ({type(new_val)})"
                    )
            conf_dict[last_key] = new_val
        elif isinstance(v, dict):
            setconfig_dict(
                conf_dict[first_key], delim.join(name[1:]), new_val, delim, verbose
            )


def parse_trainer_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="path to training .yml config file",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="", help="Override checkpoint path."
    )
    parser.add_argument("--run_name", type=str, default="", help="Run name for MLFlow.")
    args, other_args_list = parser.parse_known_args()

    return args, other_args_list


def parse_tuner_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="path to training .yml config file",
    )
    parser.add_argument(
        "--tune_config_path",
        type=str,
        required=True,
        help="path to tuning .yml config file",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="", help="Override checkpoint path."
    )
    parser.add_argument(
        "--run_name", type=str, required=True, help="Run name for result folder."
    )
    args, other_args_list = parser.parse_known_args()

    return args, other_args_list


def create_config_cli_parser(config_dict: dict) -> ArgumentParser:
    parser = ArgumentParser(add_help=False, usage=" ")
    parser.add_argument("--list_config", action="store_true")

    def add_argument(k, v):
        arg_type = type(v)
        metavar = arg_type.__name__
        if arg_type == bool:
            arg_type = strtobool
        parser.add_argument(f"--{k}", type=arg_type, default=v, metavar=metavar)

    def config_to_argument(conf_dict):
        for k, v in flatten_dict(conf_dict, sep=".").items():
            if not isinstance(v, (list, tuple)):
                if isinstance(v, Mapping):
                    config_to_argument(conf_dict)
                else:
                    add_argument(k, v)

    config_to_argument(config_dict)

    return parser


def parse_args_with_config() -> Tuple[Namespace, Adict]:
    base_args, config_args_list = parse_trainer_args()
    with open(base_args.config_path) as f:
        config_dict = Adict(yaml.load(f, Loader=yaml.SafeLoader)).to_dict()

    config_parser = create_config_cli_parser(config_dict)
    config_args = config_parser.parse_args(config_args_list)
    if config_args.list_config:
        config_parser.print_help()
        sys.exit(0)

    config_args.__delattr__("list_config")

    for k, v in config_args.__dict__.items():
        setconfig_dict(config_dict, k, v, delim=".", verbose=True)

    return base_args, Adict(config_dict)


def parse_tuner_args_with_config() -> Tuple[Namespace, Adict, dict]:
    base_args, config_args_list = parse_tuner_args()

    with open(base_args.config_path) as f:
        config_dict = Adict(yaml.load(f, Loader=yaml.SafeLoader)).to_dict()

    config_parser = create_config_cli_parser(config_dict)
    config_args = config_parser.parse_args(config_args_list)
    if config_args.list_config:
        config_parser.print_help()
        sys.exit(0)

    config_args.__delattr__("list_config")

    for k, v in config_args.__dict__.items():
        setconfig_dict(config_dict, k, v, delim=".", verbose=True)

    tuning_config = parse_tuning_config(base_args.tune_config_path)

    return base_args, Adict(config_dict), tuning_config


def load_config(config_path):
    with open(config_path, "r") as configfile:
        config = Adict(yaml.load(configfile, Loader=yaml.SafeLoader))

    return config
