from pathlib import Path

import numpy as np
from albumentations import (
    Compose,
    HorizontalFlip,
    HueSaturationValue,
    Normalize,
    RandomBrightnessContrast,
    Resize,
    Rotate,
    ShiftScaleRotate,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets

from jute_disease_pest.utils.constants import IMAGE_SIZE, ML_SPLIT_DIR, NUM_WORKERS


class JuteDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path = ML_SPLIT_DIR,
        batch_size: int = 32,
        image_size: int = IMAGE_SIZE,
        use_sampler: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_sampler = use_sampler
        self.sampler = None
        self.classes = None

        # TODO: Finalize augmentation plan
        self.train_transform = Compose(
            [
                Resize(height=image_size, width=image_size),
                ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                Rotate(limit=30, p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomBrightnessContrast(p=0.3),
                HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
                ),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        self.val_transform = Compose(
            [
                Resize(height=image_size, width=image_size),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.jute_train = datasets.ImageFolder(
                root=f"{self.data_dir}/train",
                transform=lambda x: self.train_transform(image=np.array(x))["image"],
            )
            self.jute_val = datasets.ImageFolder(
                root=f"{self.data_dir}/val",
                transform=lambda x: self.val_transform(image=np.array(x))["image"],
            )
            self.classes = self.jute_train.classes

            if self.use_sampler:
                targets = np.array(self.jute_train.targets)
                class_sample_count = np.array(
                    [len(np.where(targets == t)[0]) for t in np.unique(targets)]
                )
                weight = 1.0 / class_sample_count
                samples_weight = np.array([weight[t] for t in targets])
                self.sampler = WeightedRandomSampler(
                    samples_weight, len(samples_weight)
                )

        if stage == "test":
            self.jute_test = datasets.ImageFolder(
                root=f"{self.data_dir}/test",
                transform=lambda x: self.val_transform(image=np.array(x))["image"],
            )
            self.classes = self.jute_test.classes

        if stage == "predict":
            self.jute_predict = datasets.ImageFolder(
                root=f"{self.data_dir}/test",
                transform=lambda x: self.val_transform(image=np.array(x))["image"],
            )

    def train_dataloader(self):
        return DataLoader(
            self.jute_train,
            batch_size=self.batch_size,
            shuffle=True if self.sampler is None else False,
            sampler=self.sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.jute_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.jute_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.jute_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
