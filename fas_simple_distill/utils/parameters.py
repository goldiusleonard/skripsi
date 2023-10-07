import re
from collections import OrderedDict

import torch.nn as nn


def split_batchnorm_params(model: nn.Module):
    """Finds the set of BatchNorm parameters in the model.
    Recursively traverses all parameters in the given model and returns a tuple
    of lists: the first element is the set of batchnorm parameters, the second
    list contains all other parameters of the model.

    Adapted from ClassyVision
    https://github.com/facebookresearch/ClassyVision/blob/ca0067594a8ea4a53006f399db9222851ad79cd2/classy_vision/generic/util.py#L501
    """
    batchnorm_params = []
    other_params = []
    for module in model.modules():
        # If module has children (i.e. internal node of constructed DAG) then
        # only add direct parameters() to the list of params, else go over
        # children node to find if they are BatchNorm or have "bias".
        if list(module.children()) != []:
            for params in module.parameters(recurse=False):
                if params.requires_grad:
                    other_params.append(params)
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            for params in module.parameters():
                if params.requires_grad:
                    batchnorm_params.append(params)
        else:
            for params in module.parameters():
                if params.requires_grad:
                    other_params.append(params)
    return batchnorm_params, other_params


def remove_prefix_from_stdict(stdict: dict):
    repatt = re.compile(r"^module\.(.+)")

    new_stdict = OrderedDict()
    for k, v in stdict.items():
        match = repatt.search(k)
        if match is None:
            raise RuntimeError(f"Prefix not found on key {k} using regex {repatt}")

        new_k = match.group(1)
        new_stdict[new_k] = v

    return new_stdict


def check_module_has_params(module: nn.Module):
    return bool(len(list(module.parameters())))

