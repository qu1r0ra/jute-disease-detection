import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL.Image import Image

from jute_disease_pest.utils.constants import DEFAULT_SEED, IMAGE_SIZE


class AlbumentationsAdapter:
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, img: Image) -> torch.Tensor:
        img_np = np.array(img)
        img_augmented = self.transform(image=img_np)
        return img_augmented["image"]


# TODO: Finalize transforms
train_transform = AlbumentationsAdapter(
    A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
            ),
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        seed=DEFAULT_SEED,
    )
)

val_transform = AlbumentationsAdapter(
    A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        seed=DEFAULT_SEED,
    )
)
