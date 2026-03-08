import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL.Image import Image

from jute_disease.utils.constants import DEFAULT_SEED, IMAGE_SIZE


class AlbumentationsAdapter:
    """Adapts Albumentations transforms for PyTorch pipelines."""

    def __init__(self, transform: A.Compose) -> None:
        self.transform = transform

    def __call__(self, img: Image) -> torch.Tensor | np.ndarray:
        """Applies transformation to PIL Image and returns Tensor/Array."""
        img_np = np.array(img)
        img_augmented = self.transform(image=img_np)
        return img_augmented["image"]


def _get_base_train_ops(size: int) -> A.Compose:
    """Internal list of core training augmentations for a given size."""
    return A.Compose(
        [
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
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.PlanckianJitter(p=1.0),
                ],
                p=0.4,
            ),
            A.OneOf(
                [A.GaussNoise(p=1.0), A.ISONoise(p=1.0)],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.2,
            ),
        ],
        seed=DEFAULT_SEED,
    )


def _get_base_val_ops(size: int) -> A.Compose:
    """Internal core validation ops (resizing and cropping) for a given size."""
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=size, p=1.0),
            A.CenterCrop(height=size, width=size, pad_if_needed=True, p=1.0),
        ],
        seed=DEFAULT_SEED,
    )


# Global Normalization Pipeline
NORMALIZATION_OPS = A.Compose(
    [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ToTensorV2(),
    ],
    seed=DEFAULT_SEED,
)


def create_pipeline(size: int, is_train: bool, is_dl: bool) -> AlbumentationsAdapter:
    """Factory to create a full transformation pipeline."""
    base_group = _get_base_train_ops(size) if is_train else _get_base_val_ops(size)

    # Flatten the groups to ensure deterministic seed behavior across construction
    transforms = list(base_group.transforms)
    if is_dl:
        transforms.extend(list(NORMALIZATION_OPS.transforms))

    return AlbumentationsAdapter(A.Compose(transforms, seed=DEFAULT_SEED))


# Standard 256px baseline
ml_train_transforms = create_pipeline(IMAGE_SIZE, is_train=True, is_dl=False)
ml_val_transforms = create_pipeline(IMAGE_SIZE, is_train=False, is_dl=False)

dl_train_transforms = create_pipeline(IMAGE_SIZE, is_train=True, is_dl=True)
dl_val_transforms = create_pipeline(IMAGE_SIZE, is_train=False, is_dl=True)

# High-Resolution 512px variants
dl_train_transforms_512 = create_pipeline(512, is_train=True, is_dl=True)
dl_val_transforms_512 = create_pipeline(512, is_train=False, is_dl=True)
