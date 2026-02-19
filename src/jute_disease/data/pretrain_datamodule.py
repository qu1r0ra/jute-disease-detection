from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder

from jute_disease.data.transforms import dl_train_transforms, dl_val_transforms
from jute_disease.utils.constants import BATCH_SIZE, DEFAULT_SEED, NUM_WORKERS


class PretrainDataModule(LightningDataModule):
    """
    A generic DataModule for pre-training on external datasets (PlantVillage, PlantDoc).
    Structure expected: root/class_name/image.jpg
    """

    def __init__(
        self,
        data_dir: str,
        val_split: float = 0.2,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        pin_memory: bool = True,
        seed: int = DEFAULT_SEED,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.num_classes = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            train_dir = self.data_dir / "train"
            val_dir = self.data_dir / "val"

            if train_dir.exists() and val_dir.exists():
                self.train_dataset = ImageFolder(
                    root=train_dir, transform=dl_train_transforms
                )
                self.val_dataset = ImageFolder(
                    root=val_dir, transform=dl_val_transforms
                )
                self.num_classes = len(self.train_dataset.classes)
            else:
                full_train_ds = ImageFolder(
                    root=str(self.data_dir), transform=dl_train_transforms
                )
                full_val_ds = ImageFolder(
                    root=str(self.data_dir), transform=dl_val_transforms
                )
                self.num_classes = len(full_train_ds.classes)

                total_size = len(full_train_ds)
                val_size = int(total_size * self.val_split)
                train_size = total_size - val_size

                train_subset, val_subset = random_split(
                    full_train_ds,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.seed),
                )

                self.train_dataset = Subset(full_train_ds, train_subset.indices)
                self.val_dataset = Subset(full_val_ds, val_subset.indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
