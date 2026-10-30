"""Image pipelines built per use, never shared.

Every call to :func:`create_pipeline` returns a new, independently seeded pipeline:
no module-level ``Compose`` carries random state between datasets, partitions or
runs. Evaluation pipelines are deterministic; training augmentation is seeded and
reseeds itself per DataLoader worker from the loader's generator.

:class:`PreprocessingSpec` is the identity of the deterministic input
preprocessing. Classical-feature and CNN inputs both start from it, and its digest
is a dependency of every cache.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL.Image import Image

from jute_disease.config import DEFAULT_SEED, IMAGE_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessingSpec:
    """Deterministic preprocessing applied to every partition.

    ``cache_shortest_side`` is the lossless cache's downsize target (images are
    never upscaled); ``size`` is the model input after the evaluation resize and
    centre crop.
    """

    size: int = IMAGE_SIZE
    cache_shortest_side: int = 512
    cache_format: str = "png"
    resize: str = "smallest-max-size+center-crop"
    interpolation: str = "bicubic-downsize-no-upscale"

    def digest(self) -> str:
        """Return the SHA-256 of the spec."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


class AlbumentationsAdapter:
    """Adapts Albumentations transforms for PyTorch datasets."""

    def __init__(
        self, transform: A.Compose, description: str = "", deterministic: bool = True
    ) -> None:
        self.transform = transform
        self.description = description
        self.deterministic = deterministic

    def __call__(self, img: Image | np.ndarray) -> torch.Tensor | np.ndarray:
        """Apply the transform to a PIL image or array and return tensor/array."""
        return self.transform(image=np.asarray(img))["image"]


def _train_ops(size: int) -> list:
    return [
        A.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0), p=1.0),
        A.SquareSymmetry(p=0.5),
        A.Affine(rotate=(-15, 15), shear=(-15, 15), p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.05, 0.1),
            hole_width_range=(0.05, 0.1),
            p=0.2,
        ),
        A.OneOf(
            [A.RandomBrightnessContrast(p=1.0), A.PlanckianJitter(p=1.0)],
            p=0.4,
        ),
        A.OneOf([A.GaussNoise(p=1.0), A.ISONoise(p=1.0)], p=0.2),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ],
            p=0.2,
        ),
    ]


def _eval_ops(size: int) -> list:
    return [
        A.SmallestMaxSize(max_size=size, p=1.0),
        A.CenterCrop(height=size, width=size, pad_if_needed=True, p=1.0),
    ]


def create_pipeline(
    size: int = IMAGE_SIZE,
    is_train: bool = False,
    is_dl: bool = True,
    seed: int = DEFAULT_SEED,
) -> AlbumentationsAdapter:
    """Build a fresh pipeline.

    Args:
        size: Output side length.
        is_train: Add seeded augmentation. Evaluation pipelines are deterministic.
        is_dl: Normalise and return a CHW tensor; otherwise return the HWC array
            unnormalised, as the classical feature extractors expect.
        seed: Seed of this pipeline's random state (training only matters).
    """
    ops = _train_ops(size) if is_train else _eval_ops(size)
    if is_dl:
        ops.extend(
            [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, p=1.0), ToTensorV2()]
        )
    return AlbumentationsAdapter(
        A.Compose(ops, seed=seed),
        description=f"size={size};train={is_train};dl={is_dl}",
        deterministic=not is_train,
    )


def classical_pipeline(size: int = IMAGE_SIZE) -> AlbumentationsAdapter:
    """Deterministic, unnormalised preprocessing used for every partition."""
    return create_pipeline(size, is_train=False, is_dl=False)
