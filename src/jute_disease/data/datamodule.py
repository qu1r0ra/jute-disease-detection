from pathlib import Path

import numpy as np
from lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from jute_disease.data.transforms import dl_train_transforms, dl_val_transforms
from jute_disease.utils.constants import (
    BATCH_SIZE,
    DEFAULT_SEED,
    ML_SPLIT_DIR,
    NUM_WORKERS,
)
from jute_disease.utils.data_utils import split_data


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path = ML_SPLIT_DIR,
        batch_size: int = BATCH_SIZE,
        use_weighted_sampler: bool = False,
        seed: int = DEFAULT_SEED,
        num_workers: int = NUM_WORKERS,
        pin_memory: bool = True,
        k_fold: int = 1,
        fold_index: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_weighted_sampler = use_weighted_sampler
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.k_fold = k_fold
        self.fold_index = fold_index

        self.sampler = None
        self.classes = None
        self.train_transforms = dl_train_transforms
        self.val_transforms = dl_val_transforms

    def prepare_data(self):
        split_data()

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            dataset_pool = ImageFolder(root=f"{self.data_dir}/train")
            self.classes = dataset_pool.classes
            labels = dataset_pool.targets

            if self.k_fold > 1:
                skf = StratifiedKFold(
                    n_splits=self.k_fold, shuffle=True, random_state=self.seed
                )
                splits = list(skf.split(np.zeros(len(labels)), labels))
                train_idx, val_idx = splits[self.fold_index]

                # Two ImageFolders pointing to same data with different transforms
                # so train/val subsets get the correct augmentation pipeline.
                train_root_ds = ImageFolder(
                    root=f"{self.data_dir}/train", transform=self.train_transforms
                )
                val_root_ds = ImageFolder(
                    root=f"{self.data_dir}/train", transform=self.val_transforms
                )

                self.jute_train = Subset(train_root_ds, train_idx)
                self.jute_val = Subset(val_root_ds, val_idx)

                if self.use_weighted_sampler:
                    train_labels = [labels[i] for i in train_idx]
                    class_sample_count = np.array(
                        [
                            len(np.where(np.array(train_labels) == t)[0])
                            for t in np.unique(train_labels)
                        ]
                    )
                    weight = 1.0 / class_sample_count
                    weight_map = {
                        t: weight[i] for i, t in enumerate(np.unique(train_labels))
                    }
                    samples_weight = np.array([weight_map[t] for t in train_labels])
                    self.sampler = WeightedRandomSampler(
                        samples_weight, len(samples_weight)
                    )
            else:
                self.jute_train = ImageFolder(
                    root=f"{self.data_dir}/train",
                    transform=self.train_transforms,
                )
                self.jute_val = ImageFolder(
                    root=f"{self.data_dir}/val",
                    transform=self.val_transforms,
                )

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
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.jute_val,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.jute_test,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.jute_predict,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
