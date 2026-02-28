import timm
import torch
from torch import nn

from jute_disease.utils import get_logger

logger = get_logger(__name__)


class TimmBackbone(nn.Module):
    """Generic timm backbone wrapper."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        out_features: int | None = None,
        checkpoint_path: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        self.model_name = model_name
        self.out_features = out_features or self.backbone.num_features

        if checkpoint_path is not None and checkpoint_path != "null":
            logger.info(f"Loading TimmBackbone custom weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)

            backbone_dict = {}
            for k, v in state_dict.items():
                if k.startswith("feature_extractor.backbone."):
                    name = k.replace("feature_extractor.backbone.", "")
                elif k.startswith("feature_extractor."):
                    name = k.replace("feature_extractor.", "")
                else:
                    name = k

                if name.startswith("_orig_mod."):
                    name = name.replace("_orig_mod.", "")

                backbone_dict[name] = v

            if backbone_dict:
                msg = self.backbone.load_state_dict(backbone_dict, strict=False)
                logger.info(f"Grid Search backbone load status: {msg}")
            else:
                logger.warning("No 'feature_extractor' keys found dynamically.")

    def extra_repr(self) -> str:
        return f"model_name={self.model_name!r}, out_features={self.out_features}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
