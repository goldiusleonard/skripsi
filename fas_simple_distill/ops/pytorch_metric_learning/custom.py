from typing import Any, Callable, Dict, Optional
from torch import Tensor as T
from iglovikov_helper_functions.config_parsing.utils import object_from_dict

import pytorch_metric_learning.losses as pml_losses


def get_custom_weight_softmax_loss(
    loss_name: str,
    *,
    ce_loss: Optional[Callable[[T, T], T]] = None,
    ce_loss_conf: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    try:
        loss = vars(pml_losses)[loss_name](**kwargs)
    except KeyError:
        conf = {"type": loss_name}
        loss = object_from_dict(conf, **kwargs)

    if not isinstance(loss, pml_losses.WeightRegularizerMixin):
        raise RuntimeError(f"{loss} is not a subset of WeightRegularizerMixin!")

    if not hasattr(loss, "cross_entropy"):
        raise RuntimeError(f"{loss} does not have cross_entropy attribute!")

    if ce_loss is None == ce_loss_conf is None:
        raise RuntimeError("One of ce_loss or ce_loss_conf should be passed!")

    if ce_loss_conf:
        ce_loss = object_from_dict(ce_loss_conf)

    loss.cross_entropy = ce_loss
    return loss
