from fas_simple_distill.utils.create_vit_model import create_model
from fas_simple_distill.ops.ssdg import ssdg_normalize
import torch
import torch.nn as nn


class DG_model(nn.Module):
    def __init__(
        self,
        yaml_path: str,
        embedding_size: int = 640,
        drop_rate: float = 0.5,
        norm_flag: bool = False,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        model = create_model(yaml_path)
        self.backbone = nn.Sequential(*(list(model.children())[:-1]))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(embedding_size, embedding_size)
        # self.fc.weight.data.normal_(0, 0.005)
        # self.fc.bias.data.fill_(0.1)
        # self.embedder = nn.Sequential(self.fc, nn.ReLU(), nn.Dropout(drop_rate))
        # self.classifier = nn.Linear(embedding_size, num_classes)
        self.classifier = nn.Sequential()
        self.classifier.add_module(
            "classifier_layer1", nn.Linear(embedding_size, num_classes)
        )
        # self.classifier.add_module("bn1", nn.BatchNorm1d(1000))
        # self.classifier.add_module("relu1", nn.ReLU())
        # self.classifier.add_module(
        #     "classifier_layer2", nn.Linear(1000, (embedding_size // 8))
        # )
        # self.classifier.add_module("bn2", nn.BatchNorm1d((embedding_size // 16)))
        # self.classifier.add_module("relu2", nn.ReLU())
        # self.classifier.add_module(
        #     "classifier_layer3", nn.Linear((embedding_size // 8), num_classes)
        # )
        # self.classifier.add_module("relu2", nn.ReLU())
        # self.classifier.add_module(
        #     "classifier_layer3", nn.Linear((embedding_size // 16), num_classes)
        # )

        self.norm_flag = norm_flag

    def forward(self, x):
        featmap = self.backbone(x)
        x = self.avgpool(featmap)
        feat = torch.flatten(x, 1)
        # feat = self.embedder(feat)
        if self.norm_flag:
            feat = ssdg_normalize(feat)
        cls_out = self.classifier(feat)

        return cls_out, feat, featmap
