import timm
import torch
from torch import nn


class VGG(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "vgg19",
            pretrained=pretrained,
            num_classes=0,
        )
        # VGG-19 output is a 4096-d vector, but num_features reports 512
        self.out_features = 4096

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
