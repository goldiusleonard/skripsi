"""
Implementation of Single-Side Domain Generalization.
Based from https://github.com/taylover-pei/SSDG-CVPR2020/.
"""


from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet
from fas_simple_distill.model.iresnet_ssdg.iresnet_cosface import get_model, IResNet

from fas_simple_distill.ops.ssdg import ssdg_normalize, GRL
from fas_simple_distill.ops.mixstyle import MixStyle

class FeatureGeneratorResNetTorch(nn.Module):
    def __init__(
        self,
        model_type: str = "resnet18",
        pretrained_path: Optional[str] = None,
    ):
        super(FeatureGeneratorResNetTorch, self).__init__()
        resnet_model: resnet.ResNet = getattr(resnet, model_type)(
            pretrained=False, progress=False
        )

        if pretrained_path is not None:
            resnet_model.load_state_dict(torch.load(pretrained_path))
            print("loading model: ", pretrained_path)

        self.backbone_generator = nn.Sequential()
        self.backbone_generator.add_module("fas_conv1", resnet_model.conv1)
        self.backbone_generator.add_module("fas_bn1", resnet_model.bn1)
        self.backbone_generator.add_module("fas_relu", resnet_model.relu)
        self.backbone_generator.add_module("fas_maxpool", resnet_model.maxpool)
        self.backbone_generator.add_module("fas_layer1", resnet_model.layer1)

    def forward(self, x):
        x = self.backbone_generator(x)
        return x

class FASFeatureEmbedderResNetTorch(nn.Module):
    def __init__(
        self,
        model_type: str = "resnet18",
        embedding_size: int = 512,
        drop_rate: float = 0.5,
        norm_flag: float = True,
        pretrained_path: Optional[str] = None,
    ):
        super(FASFeatureEmbedderResNetTorch, self).__init__()
        fas_resnet_model: resnet.ResNet = getattr(resnet, model_type)(
            pretrained=False, progress=False
        )

        fas_resnet_model_no_ptr: resnet.ResNet = getattr(resnet, model_type)(
            pretrained=False, progress=False
        )

        if pretrained_path is not None:
            fas_resnet_model.load_state_dict(torch.load(pretrained_path))
            print("loading model: ", pretrained_path)

        self.backbone = nn.Sequential()
        self.backbone.add_module("fas_layer2", fas_resnet_model.layer2)
        self.backbone.add_module("fas_layer3", fas_resnet_model.layer3)
        self.backbone.add_module("fas_layer4", fas_resnet_model_no_ptr.layer4)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool_feat = 512
        if model_type not in ["resnet18", "resnet34"]:
            self.avgpool_feat = 2048

        self.fc = nn.Linear(self.avgpool_feat, embedding_size)
        self.fc.weight.data.normal_(0, 0.005)
        self.fc.bias.data.fill_(0.1)
        self.embedder = nn.Sequential(self.fc, nn.ReLU(), nn.Dropout(drop_rate))

        self.norm_flag = norm_flag

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedder(x)
        if self.norm_flag:
            x = ssdg_normalize(x)
        return x

class DomFeatureEmbedderResNetTorch(nn.Module):
    def __init__(
        self,
        model_type: str = "resnet18",
        embedding_size: int = 512,
        drop_rate: float = 0.5,
        norm_flag: float = True,
        pretrained_path: Optional[str] = None,
    ):
        super(DomFeatureEmbedderResNetTorch, self).__init__()
        dom_resnet_model: resnet.ResNet = getattr(resnet, model_type)(
            pretrained=False, progress=False
        )

        dom_resnet_model_no_ptr: resnet.ResNet = getattr(resnet, model_type)(
            pretrained=False, progress=False
        )

        if pretrained_path is not None:
            dom_resnet_model.load_state_dict(torch.load(pretrained_path))
            print("loading model: ", pretrained_path)

        self.backbone = nn.Sequential()
        self.backbone.add_module("fas_layer2", dom_resnet_model.layer2)
        self.backbone.add_module("fas_layer3", dom_resnet_model.layer3)
        self.backbone.add_module("fas_layer4", dom_resnet_model_no_ptr.layer4)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool_feat = 512
        if model_type not in ["resnet18", "resnet34"]:
            self.avgpool_feat = 2048

        self.fc = nn.Linear(self.avgpool_feat, embedding_size)
        self.fc.weight.data.normal_(0, 0.005)
        self.fc.bias.data.fill_(0.1)
        self.embedder = nn.Sequential(self.fc, nn.ReLU(), nn.Dropout(drop_rate))

        self.norm_flag = norm_flag

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedder(x)
        if self.norm_flag:
            x = ssdg_normalize(x)
        return x

class Classifier(nn.Module):
    def __init__(self, embedding_size=512, num_classes=2):
        super(Classifier, self).__init__()
        self.classifier_layer_1 = nn.Linear(embedding_size, embedding_size)
        self.classifier_layer_1.weight.data.normal_(0, 0.01)
        self.classifier_layer_1.bias.data.fill_(0.0)
        self.classifier_layer_2 = nn.Linear(embedding_size, num_classes)
        self.classifier_layer_2.weight.data.normal_(0, 0.3)
        self.classifier_layer_2.bias.data.fill_(0.0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inp, norm_flag):
        if norm_flag:
            self.classifier_layer_1.weight.data = F.normalize(
                self.classifier_layer_1.weight, dim=0
            )
            self.classifier_layer_2.weight.data = F.normalize(
                self.classifier_layer_2.weight, dim=0
            )
            classifier_out = self.classifier_layer_1(inp)
            classifier_out = self.relu(classifier_out)
            classifier_out = self.dropout(classifier_out)
            classifier_out = self.classifier_layer_2(classifier_out)
        else:
            classifier_out = self.classifier_layer_1(inp)
            classifier_out = self.relu(classifier_out)
            classifier_out = self.dropout(classifier_out)
            classifier_out = self.classifier_layer_2(classifier_out)
        return classifier_out

class Discriminator(nn.Module):
    def __init__(self, num_classes: int, max_iter: int, in_features=512):
        """Discriminator model as implemented in SSDG.
        """
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(self.fc1, nn.ReLU(), nn.Dropout(0.5), self.fc2)
        self.grl_layer = GRL(max_iter=max_iter)

    def forward(self, feature, current_iter: int):
        adversarial_out = self.ad_net(self.grl_layer(feature, current_iter))
        return adversarial_out

class DG_model(nn.Module):
    def __init__(
        self,
        model: str = "resnet18",
        embedding_size: int = 512,
        drop_rate: float = 0.5,
        pretrained_path: Optional[str] = None,
        norm_flag: bool = True,
    ):
        super(DG_model, self).__init__()
        self.generator = FeatureGeneratorResNetTorch(
            model_type=model,
            pretrained_path=pretrained_path,
        )
        self.fas_embedder = FASFeatureEmbedderResNetTorch(
            model_type=model,
            embedding_size=embedding_size,
            drop_rate=drop_rate,
            norm_flag=norm_flag,
            pretrained_path=pretrained_path,
        )
        self.dom_embedder = DomFeatureEmbedderResNetTorch(
            model_type=model,
            embedding_size=embedding_size,
            drop_rate=drop_rate,
            norm_flag=norm_flag,
            pretrained_path=pretrained_path,
        )

    def forward(self, fas_input):
        embedding = self.generator(fas_input)
        fas_features: torch.Tensor = self.fas_embedder(embedding)
        dom_features: torch.Tensor = self.dom_embedder(embedding)
        return fas_features, dom_features