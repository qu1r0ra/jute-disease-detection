from pathlib import Path

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from jute_disease_pest.data.transforms import train_transforms, val_transforms
from jute_disease_pest.engines.split import split_dataset
from jute_disease_pest.utils.constants import (
    BATCH_SIZE,
    DEFAULT_SEED,
    ML_SPLIT_DIR,
    NUM_WORKERS,
)
from jute_disease_pest.utils.seed import seed_everything


class JuteDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path = ML_SPLIT_DIR,
        batch_size: int = BATCH_SIZE,
        use_weighted_sampler: bool = False,
        seed: int = DEFAULT_SEED,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_weighted_sampler = use_weighted_sampler
        self.seed = seed

        self.sampler = None
        self.classes = None
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        seed_everything(self.seed)

    def prepare_data(self):
        split_dataset()

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.jute_train = ImageFolder(
                root=f"{self.data_dir}/train",
                transform=self.train_transforms,
            )
            self.jute_val = ImageFolder(
                root=f"{self.data_dir}/val",
                transform=self.val_transforms,
            )
            self.classes = self.jute_train.classes

            if self.use_weighted_sampler:
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
            self.jute_test = ImageFolder(
                root=f"{self.data_dir}/test",
                transform=self.val_transforms,
            )
            self.classes = self.jute_test.classes

        if stage == "predict":
            self.jute_predict = ImageFolder(
                root=f"{self.data_dir}/test",
                transform=self.val_transforms,
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
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.jute_test,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.jute_predict,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
