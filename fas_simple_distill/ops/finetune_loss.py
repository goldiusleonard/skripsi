from mimetypes import init
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import magface,focal_loss
from iglovikov_helper_functions.config_parsing.utils import object_from_dict

def fas_loss(config, loss_type, device):
    if(loss_type == "MagFace"):
        maglinear = magface.MagLinear(
                                in_features= config.hparams.model.embedding_size,
                                out_features= config.hparams.MagLoss.num_classes,
                                lower_a = config.hparams.MagLoss.lower_a,
                                upper_a = config.hparams.MagLoss.upper_a,
                                lower_margin = config.hparams.MagLoss.lower_margin,
                                upper_margin = config.hparams.MagLoss.upper_margin,
                            ).to(device)
        metric_loss = object_from_dict(config.hparams.MagLoss).to(
            device
        )
        
        return metric_loss, maglinear
    
    elif(loss_type == "TripleLoss"):
        metric_loss = object_from_dict(config.hparams.SoftTripleLoss).to(
            device
        )
    elif(loss_type == "AdaFace"):
        metric_loss = object_from_dict(config.hparams.AdaFace).to(
            device
        )
    else:
        raise ValueError(
            f"metric_loss loss_type must be either `MagFace, `TripletLoss` or `AdaFace`. Got {loss_type}"
        )
    
    return metric_loss
            
            
