import timm
import torch
from torch import nn


class TimmBackbone(nn.Module):
    """Generic timm backbone wrapper."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        out_features: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        self.model_name = model_name
        self.out_features = out_features or self.backbone.num_features

    def extra_repr(self) -> str:
        return f"model_name={self.model_name!r}, out_features={self.out_features}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
