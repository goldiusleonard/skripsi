import subprocess
import random
import collections
from functools import wraps

import torch
import numpy as np
from torch.distributed import get_rank


def rank_zero(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if get_rank() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def is_homogenous_iterable(iterable_obj, type_=int):
    return all(isinstance(el, type_) for el in iterable_obj)


def get_head_commit():
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        )
    except subprocess.CalledProcessError:
        commit = ""

    return commit


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
