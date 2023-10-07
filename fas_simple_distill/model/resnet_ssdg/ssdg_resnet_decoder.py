"""
Implementation of Single-Side Domain Generalization.
Based from https://github.com/taylover-pei/SSDG-CVPR2020/.
"""


from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet
from fas_simple_distill.model.iresnet_ssdg.iresnet_cosface import IResNet, get_model

from fas_simple_distill.ops.ssdg import ssdg_normalize, GRL
from fas_simple_distill.model.resnet_ssdg.resnet_decoder import ResNet18Dec


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
    def __init__(self, num_classes: int, max_iter: int, in_features: int = 512):
        """Discriminator model as implemented in SSDG."""
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

        self.fr_backbone = nn.Sequential()
        self.fr_backbone.add_module("fr_conv1", fr_resnet_model.conv1)
        self.fr_backbone.add_module("fr_bn1", fr_resnet_model.bn1)
        self.fr_backbone.add_module("fr_prelu", fr_resnet_model.prelu)
        self.fr_backbone.add_module("fr_layer1", fr_resnet_model.layer1)
        self.fr_backbone.add_module("fr_layer2", fr_resnet_model.layer2)
        self.fr_backbone.add_module("fr_layer3", fr_resnet_model.layer3)
        self.fr_backbone.add_module("fr_layer4", fr_resnet_model.layer4)

        self.fr_output_layer = nn.Sequential()
        self.fr_output_layer.add_module("fr_bn2", fr_resnet_model.bn2)
        self.fr_output_layer.add_module("fr_flatten", nn.Flatten())
        self.fr_output_layer.add_module("fr_dropout", fr_resnet_model.dropout)
        self.fr_output_layer.add_module("fr_fc", fr_resnet_model.fc)
        self.fr_output_layer.add_module("fr_features", fr_resnet_model.features)

    def forward(self, x):
        x = self.fr_backbone(x)
        x = self.fr_output_layer(x)
        return x


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


class Dec_model(nn.Module):
    def __init__(
        self,
        fas_model: str = "resnet18",
        fr_model: str = "r18",
        embedding_size: int = 512,
        drop_rate: float = 0.5,
        fas_pretrained_path: Optional[str] = None,
        fr_pretrained_path: Optional[str] = None,
        norm_flag: bool = True,
        **kwargs
    ):
        super(Dec_model, self).__init__()
        self.fas_embedder = DG_model(
            model=fas_model,
            embedding_size=embedding_size,
            drop_rate=drop_rate,
            norm_flag=norm_flag,
        )

        if fas_pretrained_path is not None:
            ckpt = torch.load(fas_pretrained_path)
            model_ckpt = ckpt["model"]
            self.fas_embedder.load_state_dict(model_ckpt)

        self.fr_embedder = FRFeatureEmbedderiResNetTorch(
            model_type=fr_model,
            pretrained_path=fr_pretrained_path,
        )
        self.decoder = ResNet18Dec(
            nc=3,
            z_dim=512,
        )

        self.fas_embedder.requires_grad_(False)
        self.fr_embedder.requires_grad_(False)
        self.fas_embedder.eval()
        self.fr_embedder.eval()

    def forward(self, x, resized_x):
        classifier_out, fas_features = self.fas_embedder(x)
        fr_features = self.fr_embedder(resized_x)
        # features = torch.cat([fas_features, fr_features], dim=1)
        recon_img = self.decoder(fas_features)

        return fas_features, fr_features, classifier_out, recon_img

    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self.fas_embedder.eval()
        self.fr_embedder.eval()
        self.decoder.train()
        return self
