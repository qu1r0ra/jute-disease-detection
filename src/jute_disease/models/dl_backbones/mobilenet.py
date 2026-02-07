import timm
import torch
from torch import nn


class MobileNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv2_100",
            pretrained=pretrained,
            num_classes=0,
        )
        self.out_features = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
