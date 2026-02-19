from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Subset,
    WeightedRandomSampler,
    random_split,
)
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
    """
    A unified DataModule for Jute disease detection and external pre-training datasets.
    Optimized for multi-GPU training and fast K-Fold iteration.
    """

    def __init__(
        self,
        data_dir: Path | str = ML_SPLIT_DIR,
        val_split: float | None = None,
        batch_size: int = BATCH_SIZE,
        use_weighted_sampler: bool = False,
        seed: int = DEFAULT_SEED,
        num_workers: int = NUM_WORKERS,
        pin_memory: bool = True,
        k_fold: int = 1,
        fold_index: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)

        self.jute_train = None
        self.jute_val = None
        self.jute_test = None
        self.jute_predict = None
        self.sampler = None
        self._classes = None

        # K-Fold
        self._splits = None
        self._pool_labels = None
        self._train_pool = None
        self._val_pool = None

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.data_dir / "val"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return len(self._classes) if self._classes else 0

    def prepare_data(self):
        if self.data_dir == ML_SPLIT_DIR:
            split_data()

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            if self.train_dir.exists() and self.val_dir.exists():
                if self.hparams.k_fold > 1:
                    train_ds = ImageFolder(root=self.train_dir)
                    val_ds = ImageFolder(root=self.val_dir)

                    self._classes = train_ds.classes
                    self._pool_labels = train_ds.targets + val_ds.targets

                    skf = StratifiedKFold(
                        n_splits=self.hparams.k_fold,
                        shuffle=True,
                        random_state=self.hparams.seed,
                    )
                    self._splits = list(
                        skf.split(np.zeros(len(self._pool_labels)), self._pool_labels)
                    )

                    self._train_pool = ConcatDataset(
                        [
                            ImageFolder(
                                root=self.train_dir, transform=dl_train_transforms
                            ),
                            ImageFolder(
                                root=self.val_dir, transform=dl_train_transforms
                            ),
                        ]
                    )
                    self._val_pool = ConcatDataset(
                        [
                            ImageFolder(
                                root=self.train_dir, transform=dl_val_transforms
                            ),
                            ImageFolder(root=self.val_dir, transform=dl_val_transforms),
                        ]
                    )

                    self.set_fold(self.hparams.fold_index)
                else:
                    self.jute_train = ImageFolder(
                        root=self.train_dir, transform=dl_train_transforms
                    )
                    self.jute_val = ImageFolder(
                        root=self.val_dir, transform=dl_val_transforms
                    )
                    self._classes = self.jute_train.classes
                    if self.hparams.use_weighted_sampler:
                        self.sampler = self._create_weighted_sampler(
                            self.jute_train.targets
                        )

            elif self.hparams.val_split is not None:
                full_ds = ImageFolder(root=self.data_dir)
                self._classes = full_ds.classes

                total_size = len(full_ds)
                val_size = int(total_size * self.hparams.val_split)
                train_size = total_size - val_size

                train_subset, val_subset = random_split(
                    full_ds,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.hparams.seed),
                )

                self.jute_train = Subset(
                    ImageFolder(root=self.data_dir, transform=dl_train_transforms),
                    train_subset.indices,
                )
                self.jute_val = Subset(
                    ImageFolder(root=self.data_dir, transform=dl_val_transforms),
                    val_subset.indices,
                )

        if stage == "test" or stage is None:
            if self.test_dir.exists():
                self.jute_test = ImageFolder(
                    root=self.test_dir, transform=dl_val_transforms
                )
                self._classes = self.jute_test.classes

        if stage == "predict":
            predict_root = self.test_dir if self.test_dir.exists() else self.data_dir
            self.jute_predict = ImageFolder(
                root=predict_root, transform=dl_val_transforms
            )

    def set_fold(self, fold_index: int):
        """Swaps active subsets and updates sampler."""
        if self._splits is None:
            raise RuntimeError("Base splits not initialized. Call setup() first.")

        if not (0 <= fold_index < self.hparams.k_fold):
            raise ValueError(
                f"Fold index {fold_index} out of range (0-{self.hparams.k_fold - 1})"
            )

        self.hparams.fold_index = fold_index
        train_idx, val_idx = self._splits[fold_index]
        self.jute_train = Subset(self._train_pool, train_idx)
        self.jute_val = Subset(self._val_pool, val_idx)

        if self.hparams.use_weighted_sampler:
            train_labels = [self._pool_labels[i] for i in train_idx]
            self.sampler = self._create_weighted_sampler(train_labels)

    def _create_weighted_sampler(self, labels: list | np.ndarray):
        """Helper to create a WeightedRandomSampler to address class imbalance."""
        targets = np.array(labels)
        unique_targets, counts = np.unique(targets, return_counts=True)
        weight = 1.0 / counts
        weight_map = {t: weight[i] for i, t in enumerate(unique_targets)}
        samples_weight = np.array([weight_map[t] for t in targets])
        return WeightedRandomSampler(samples_weight, len(samples_weight))

    def train_dataloader(self):
        return DataLoader(
            self.jute_train,
            batch_size=self.hparams.batch_size,
            shuffle=True if self.sampler is None else False,
            sampler=self.sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.jute_val,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.jute_test,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.jute_predict,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
