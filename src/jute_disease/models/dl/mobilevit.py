from pathlib import Path

import timm
import torch
from torch import nn

from jute_disease.utils.logger import get_logger

logger = get_logger(__name__)


class MobileViT(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "mobilevit_s", pretrained=pretrained, num_classes=0, **kwargs
        )
        self.out_features = self.backbone.num_features

        if checkpoint_path:
            self._load_custom_checkpoint(checkpoint_path)

    def _load_custom_checkpoint(self, path: str | Path) -> None:
        logger.info(f"Loading custom backbone weights from {path}...")
        checkpoint = torch.load(path, map_location="cpu")

        state_dict = checkpoint.get("state_dict", checkpoint)

        backbone_dict = {}
        for k, v in state_dict.items():
            if "feature_extractor.backbone." in k:
                name = k.replace("feature_extractor.backbone.", "")
                backbone_dict[name] = v
            elif "feature_extractor." in k:
                name = k.replace("feature_extractor.", "")
                backbone_dict[name] = v

        if not backbone_dict:
            logger.warning(
                f"No matching backbone keys found in {path}. "
                "Using default initialization."
            )
            return

        msg = self.backbone.load_state_dict(backbone_dict, strict=False)
        logger.info(f"Backbone load status: {msg}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
