"""
Implementation of Single-Side Domain Generalization.
Based from https://github.com/taylover-pei/SSDG-CVPR2020/.
"""


from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet
from fas_simple_distill.model.iresnet import IResNet, get_model

from fas_simple_distill.ops.ssdg import ssdg_normalize, GRL


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class FRFeatureEmbedderiResNetTorch(nn.Module):
    def __init__(
        self,
        model_type: str = "r18",
        pretrained_path: Optional[str] = None,
        norm_flag: bool = True,
    ):
        super(FRFeatureEmbedderiResNetTorch, self).__init__()
        resnet_model: IResNet = get_model(model_type)

        if pretrained_path is not None:
            resnet_model.load_state_dict(torch.load(pretrained_path))
            print("loading model: ", pretrained_path)

        self.backbone = nn.Sequential()
        self.backbone.add_module("conv1", resnet_model.conv1)
        self.backbone.add_module("bn1", resnet_model.bn1)
        self.backbone.add_module("prelu", resnet_model.prelu)
        self.backbone.add_module("layer1", resnet_model.layer1)
        self.backbone.add_module("layer2", resnet_model.layer2)
        self.backbone.add_module("layer3", resnet_model.layer3)
        self.backbone.add_module("layer4", resnet_model.layer4)

        self.output_layer = nn.Sequential()
        self.output_layer.add_module("bn2", resnet_model.bn2)
        self.output_layer.add_module("flatten", nn.Flatten())
        self.output_layer.add_module("dropout", resnet_model.dropout)
        self.output_layer.add_module("fc", resnet_model.fc)
        self.output_layer.add_module("features", resnet_model.features)

        self.norm_flag = norm_flag

    def forward(self, x):
        x = self.backbone(x)
        x = self.output_layer(x)
        if self.norm_flag:
            x = ssdg_normalize(x)
        return x


class Classifier(nn.Module):
    def __init__(self, embedding_size=512, num_classes=2):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(embedding_size, num_classes)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, inp, norm_flag):
        if norm_flag:
            self.classifier_layer.weight.data = l2_norm(
                self.classifier_layer.weight, axis=0
            )
            classifier_out = self.classifier_layer(inp)
        else:
            classifier_out = self.classifier_layer(inp)
        return classifier_out


class DG_model(nn.Module):
    def __init__(
        self,
        model: str = "resnet18",
        pretrained_path: Optional[str] = None,
        embedding_size: int = 512,
        norm_flag: bool = True,
        num_subject: int = 140,
        **kwargs
    ):
        """Single-Side Domain Generalization Model

        Args:
            model (str, optional): Type of resnet model as backbone. Defaults to "resnet18".
            embedding_size (int, optional): Dimension of the features. Defaults to 512.
            drop_rate (float, optional): Probability used in dropout layer. Defaults to 0.5.
            pretrained_path (bool, optional): Whether to use pretrained_path resnet model. Defaults to True.
            norm_flag (bool, optional): Whether to normalize features. Defaults to True.
            detach_classifier (bool, optional): Whether to computation graph of classifier from features,
                which disables gradient backpropagation from the classifier to the backbone. Defaults to False.
        """
        super(DG_model, self).__init__()
        self.embedder = FRFeatureEmbedderiResNetTorch(
            model_type=model,
            norm_flag=norm_flag,
            pretrained_path=pretrained_path,
            **kwargs,
        )
        self.classifier = Classifier(embedding_size, num_subject)
        self.norm_flag = norm_flag

    def forward(self, x):
        features: torch.Tensor = self.embedder(x)
        cls_out = self.classifier(features, self.norm_flag)
        return cls_out, features
