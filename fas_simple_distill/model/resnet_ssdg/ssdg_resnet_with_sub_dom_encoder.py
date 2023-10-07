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

class FASFeatureGeneratorResNetTorch(nn.Module):
    def __init__(
        self,
        model_type: str = "resnet18",
        pretrained_path: Optional[str] = None,
    ):
        super(FASFeatureGeneratorResNetTorch, self).__init__()
        fas_resnet_model: resnet.ResNet = getattr(resnet, model_type)(
            pretrained=False, progress=False
        )

        if pretrained_path is not None:
            fas_resnet_model.load_state_dict(torch.load(pretrained_path))
            print("loading model: ", pretrained_path)

        self.fas_backbone_generator = nn.Sequential()
        self.fas_backbone_generator.add_module("fas_conv1", fas_resnet_model.conv1)
        self.fas_backbone_generator.add_module("fas_bn1", fas_resnet_model.bn1)
        self.fas_backbone_generator.add_module("fas_relu", fas_resnet_model.relu)
        self.fas_backbone_generator.add_module("fas_maxpool", fas_resnet_model.maxpool)
        self.fas_backbone_generator.add_module("fas_layer1", fas_resnet_model.layer1)

    def forward(self, x):
        x = self.fas_backbone_generator(x)
        return x

class FRFeatureGeneratoriResNetTorch(nn.Module):
    def __init__(
        self,
        model_type: str = "r18",
        pretrained_path: Optional[str] = None,
    ):
        super(FRFeatureGeneratoriResNetTorch, self).__init__()
        fr_resnet_model: IResNet = get_model(model_type)

        if pretrained_path is not None:
            fr_resnet_model.load_state_dict(torch.load(pretrained_path))
            print("loading model: ", pretrained_path)

        self.fr_backbone_generator = nn.Sequential()
        self.fr_backbone_generator.add_module("fr_conv1", fr_resnet_model.conv1)
        self.fr_backbone_generator.add_module("fr_bn1", fr_resnet_model.bn1)
        self.fr_backbone_generator.add_module("fr_prelu", fr_resnet_model.prelu)
        self.fr_backbone_generator.add_module("fr_layer1", fr_resnet_model.layer1)

    def forward(self, x):
        x = self.fr_backbone_generator(x)
        return x
    
class DomFeatureGeneratorResNetTorch(nn.Module):
    def __init__(
        self,
        model_type: str = "resnet18",
        pretrained_path: Optional[str] = None,
    ):
        super(DomFeatureGeneratorResNetTorch, self).__init__()
        dom_resnet_model: resnet.ResNet = getattr(resnet, model_type)(
            pretrained=False, progress=False
        )

        if pretrained_path is not None:
            dom_resnet_model.load_state_dict(torch.load(pretrained_path))
            print("loading model: ", pretrained_path)

        self.dom_backbone_generator = nn.Sequential()
        self.dom_backbone_generator.add_module("fas_conv1", dom_resnet_model.conv1)
        self.dom_backbone_generator.add_module("fas_bn1", dom_resnet_model.bn1)
        self.dom_backbone_generator.add_module("fas_relu", dom_resnet_model.relu)
        self.dom_backbone_generator.add_module("fas_maxpool", dom_resnet_model.maxpool)
        self.dom_backbone_generator.add_module("fas_layer1", dom_resnet_model.layer1)

    def forward(self, x):
        x = self.dom_backbone_generator(x)
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

        self.fas_backbone = nn.Sequential()
        self.fas_backbone.add_module("fas_layer2", fas_resnet_model.layer2)
        self.fas_backbone.add_module("fas_layer3", fas_resnet_model.layer3)
        self.fas_backbone.add_module("fas_layer4", fas_resnet_model_no_ptr.layer4)

        self.fas_avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool_feat = 512
        if model_type not in ["resnet18", "resnet34"]:
            self.avgpool_feat = 2048

        self.fc = nn.Linear(self.avgpool_feat, embedding_size)
        self.fc.weight.data.normal_(0, 0.005)
        self.fc.bias.data.fill_(0.1)
        self.fas_embedder = nn.Sequential(self.fc, nn.ReLU(), nn.Dropout(drop_rate))

        self.norm_flag = norm_flag

    def forward(self, x):
        x = self.fas_backbone(x)
        x = self.fas_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fas_embedder(x)
        if self.norm_flag:
            x = ssdg_normalize(x)
        return x

class FRFeatureEmbedderiResNetTorch(nn.Module):
    def __init__(
        self,
        model_type: str = "r18",
        pretrained_path: Optional[str] = None,
    ):
        super(FRFeatureEmbedderiResNetTorch, self).__init__()
        fr_resnet_model: IResNet = get_model(model_type)

        if pretrained_path is not None:
            fr_resnet_model.load_state_dict(torch.load(pretrained_path))
            print("loading model: ", pretrained_path)

        self.fr_backbone_embedder = nn.Sequential()
        self.fr_backbone_embedder.add_module("fr_layer2", fr_resnet_model.layer2)
        self.fr_backbone_embedder.add_module("fr_layer3", fr_resnet_model.layer3)
        self.fr_backbone_embedder.add_module("fr_layer4", fr_resnet_model.layer4)

        self.fr_output_layer = nn.Sequential()
        self.fr_output_layer.add_module("fr_bn2", fr_resnet_model.bn2)
        self.fr_output_layer.add_module("fr_flatten", nn.Flatten())
        self.fr_output_layer.add_module("fr_dropout", fr_resnet_model.dropout)
        self.fr_output_layer.add_module("fr_fc", fr_resnet_model.fc)
        self.fr_output_layer.add_module("fr_features", fr_resnet_model.features)

    def forward(self, x):
        x = self.fr_backbone_embedder(x)
        x = self.fr_output_layer(x)
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

        self.dom_backbone = nn.Sequential()
        self.dom_backbone.add_module("fas_layer2", dom_resnet_model.layer2)
        self.dom_backbone.add_module("fas_layer3", dom_resnet_model.layer3)
        self.dom_backbone.add_module("fas_layer4", dom_resnet_model_no_ptr.layer4)

        self.dom_avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool_feat = 512
        if model_type not in ["resnet18", "resnet34"]:
            self.avgpool_feat = 2048

        self.fc = nn.Linear(self.avgpool_feat, embedding_size)
        self.fc.weight.data.normal_(0, 0.005)
        self.fc.bias.data.fill_(0.1)
        self.dom_embedder = nn.Sequential(self.fc, nn.ReLU(), nn.Dropout(drop_rate))

        self.norm_flag = norm_flag

    def forward(self, x):
        x = self.dom_backbone(x)
        x = self.dom_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dom_embedder(x)
        if self.norm_flag:
            x = ssdg_normalize(x)
        return x

class FAS_Classifier(nn.Module):
    def __init__(self, embedding_size=512, num_classes=2):
        super(FAS_Classifier, self).__init__()
        self.fas_classifier_layer_1 = nn.Linear(embedding_size, embedding_size)
        self.fas_classifier_layer_1.weight.data.normal_(0, 0.01)
        self.fas_classifier_layer_1.bias.data.fill_(0.0)
        self.fas_classifier_layer_2 = nn.Linear(embedding_size, num_classes)
        self.fas_classifier_layer_2.weight.data.normal_(0, 0.3)
        self.fas_classifier_layer_2.bias.data.fill_(0.0)
        self.fas_relu = nn.ReLU()
        self.fas_dropout = nn.Dropout(0.5)

    def forward(self, inp, norm_flag):
        if norm_flag:
            self.fas_classifier_layer_1.weight.data = F.normalize(
                self.fas_classifier_layer_1.weight, dim=0
            )
            self.fas_classifier_layer_2.weight.data = F.normalize(
                self.fas_classifier_layer_2.weight, dim=0
            )
            classifier_out = self.fas_classifier_layer_1(inp)
            classifier_out = self.fas_relu(classifier_out)
            classifier_out = self.fas_dropout(classifier_out)
            classifier_out = self.fas_classifier_layer_2(classifier_out)
        else:
            classifier_out = self.fas_classifier_layer_1(inp)
            classifier_out = self.fas_relu(classifier_out)
            classifier_out = self.fas_dropout(classifier_out)
            classifier_out = self.fas_classifier_layer_2(classifier_out)
        return classifier_out

class Dom_Classifier(nn.Module):
    def __init__(self, embedding_size=512, num_classes=2):
        super(Dom_Classifier, self).__init__()
        self.dom_classifier_layer_1 = nn.Linear(embedding_size, embedding_size)
        self.dom_classifier_layer_1.weight.data.normal_(0, 0.01)
        self.dom_classifier_layer_1.bias.data.fill_(0.0)
        self.dom_classifier_layer_2 = nn.Linear(embedding_size, num_classes)
        self.dom_classifier_layer_2.weight.data.normal_(0, 0.3)
        self.dom_classifier_layer_2.bias.data.fill_(0.0)
        self.dom_relu = nn.ReLU()
        self.dom_dropout = nn.Dropout(0.5)

    def forward(self, inp, norm_flag):
        if norm_flag:
            self.dom_classifier_layer_1.weight.data = F.normalize(
                self.dom_classifier_layer_1.weight, dim=0
            )
            self.dom_classifier_layer_2.weight.data = F.normalize(
                self.dom_classifier_layer_2.weight, dim=0
            )
            classifier_out = self.dom_classifier_layer_1(inp)
            classifier_out = self.dom_relu(classifier_out)
            classifier_out = self.dom_dropout(classifier_out)
            classifier_out = self.dom_classifier_layer_2(classifier_out)
        else:
            classifier_out = self.dom_classifier_layer_1(inp)
            classifier_out = self.dom_relu(classifier_out)
            classifier_out = self.dom_dropout(classifier_out)
            classifier_out = self.dom_classifier_layer_2(classifier_out)
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

class FAS_model(nn.Module):
    def __init__(
        self,
        model: str = "resnet18",
        embedding_size: int = 512,
        drop_rate: float = 0.5,
        pretrained_path: Optional[str] = None,
        norm_flag: bool = True,
    ):
        super(FAS_model, self).__init__()
        self.fas_generator = FASFeatureGeneratorResNetTorch(
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

    def forward(self, fas_input):
        fas_embedding = self.fas_generator(fas_input)
        fas_features: torch.Tensor = self.fas_embedder(fas_embedding)
        return fas_features

class FR_model(nn.Module):
    def __init__(
        self,
        model: str = "r18",
        pretrained_path: Optional[str] = None,
    ):
        super(FR_model, self).__init__()
        self.fr_generator = FRFeatureGeneratoriResNetTorch(
            model_type=model,
            pretrained_path=pretrained_path,
        )
        self.fr_embedder = FRFeatureEmbedderiResNetTorch(
            model_type=model,
            pretrained_path=pretrained_path,
        )

    def forward(self, fr_input):
        fr_embedding = self.fr_generator(fr_input)
        fr_features: torch.Tensor = self.fr_embedder(fr_embedding)
        return fr_features

class Dom_model(nn.Module):
    def __init__(
        self,
        model: str = "resnet18",
        embedding_size: int = 512,
        drop_rate: float = 0.5,
        pretrained_path: Optional[str] = None,
        norm_flag: bool = True,
    ):
        super(Dom_model, self).__init__()
        self.dom_generator = DomFeatureGeneratorResNetTorch(
            model_type=model,
            pretrained_path=pretrained_path,
        )
        self.dom_embedder = DomFeatureEmbedderResNetTorch(
            model_type=model,
            embedding_size=embedding_size,
            drop_rate=drop_rate,
            norm_flag=norm_flag,
            pretrained_path=pretrained_path,
        )

    def forward(self, dom_input):
        dom_embedding = self.dom_generator(dom_input)
        dom_features: torch.Tensor = self.dom_embedder(dom_embedding)
        return dom_features