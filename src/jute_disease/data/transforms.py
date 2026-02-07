import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL.Image import Image

from jute_disease.utils.constants import DEFAULT_SEED, IMAGE_SIZE


class AlbumentationsAdapter:
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, img: Image) -> torch.Tensor:
        img_np = np.array(img)
        img_augmented = self.transform(image=img_np)
        return img_augmented["image"]


# The transforms were constructed with the guidance of
# https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/
train_transforms = AlbumentationsAdapter(
    A.Compose(
        [
            # 1. Cropping
            A.RandomResizedCrop(
                size=(IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.8, 1.0),
                p=1.0,
            ),
            # 2. Geometric
            A.SquareSymmetry(p=0.5),
            A.Affine(
                rotate=(-15, 15),
                shear=(-15, 15),
                p=0.5,
            ),
            # 3. Dropout
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.05, 0.1),
                hole_width_range=(0.05, 0.1),
                p=0.2,
            ),
            # 6. Domain-Specific and Advanced
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.PlanckianJitter(p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.ISONoise(p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.2,
            ),
            # 7. Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0,
            ),
            ToTensorV2(),
        ],
        seed=DEFAULT_SEED,
    )
)

val_transforms = AlbumentationsAdapter(
    A.Compose(
        [
            A.SmallestMaxSize(max_size=IMAGE_SIZE, p=1.0),
            A.CenterCrop(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                pad_if_needed=True,
                p=1.0,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0,
            ),
            ToTensorV2(),
        ],
        seed=DEFAULT_SEED,
    )
)
