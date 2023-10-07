"""
Implementation of Single-Side Domain Generalization.
Based from https://github.com/taylover-pei/SSDG-CVPR2020/.
"""


from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet

from fas_simple_distill.ops.ssdg import ssdg_normalize, GRL

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class FeatureEmbedderResNetTorch(nn.Module):
    def __init__(
        self,
        model_type: str = "resnet18",
        embedding_size: int = 512,
        drop_rate: float = 0.5,
        norm_flag: float = True,
        pretrained_path: Optional[str] = None,
        **kwargs
    ):
        super(FeatureEmbedderResNetTorch, self).__init__()
        resnet_model: resnet.ResNet = getattr(resnet, model_type)(
            pretrained=False, progress=False, **kwargs
        )
        resnet_model_no_ptr: resnet.ResNet = getattr(resnet, model_type)(
            pretrained=False, progress=False, **kwargs
        )

        if pretrained_path is not None:
            resnet_model.load_state_dict(torch.load(pretrained_path))
            print("loading model: ", pretrained_path)

        self.backbone = nn.Sequential()
        self.backbone.add_module("conv1", resnet_model.conv1)
        self.backbone.add_module("bn1", resnet_model.bn1)
        self.backbone.add_module("relu", resnet_model.relu)
        self.backbone.add_module("maxpool", resnet_model.maxpool)
        self.backbone.add_module("layer1", resnet_model.layer1)
        self.backbone.add_module("layer2", resnet_model.layer2)
        self.backbone.add_module("layer3", resnet_model.layer3)
        self.backbone.add_module("layer4", resnet_model_no_ptr.layer4)

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

class Discriminator(nn.Module):
    def __init__(self, num_classes: int, max_iter: int, in_features:int = 512):
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
        detach_classifier: bool = False,
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
        self.embedder = FeatureEmbedderResNetTorch(
            model_type=model,
            embedding_size=embedding_size,
            drop_rate=drop_rate,
            norm_flag=norm_flag,
            pretrained_path=pretrained_path,
            **kwargs,
        )
        self.classifier = Classifier(embedding_size=embedding_size)
        self.norm_flag = norm_flag
        self.detach_classifier = detach_classifier

    def forward(self, x):
        features: torch.Tensor = self.embedder(x)

        if self.detach_classifier:
            feat_for_classif = features.clone().detach()
        else:
            feat_for_classif = features
        classifier_out = self.classifier(feat_for_classif, self.norm_flag)

        return classifier_out, feat_for_classif
