import os

from dotenv import load_dotenv
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import DemoModel
from lightning.pytorch.loggers import WandbLogger

import wandb
from jute_disease.data.jute_datamodule import JuteDataModule
from jute_disease.models.jute_classifier import JuteClassifier
from jute_disease.utils.constants import (
    BATCH_SIZE,
    DATA_DIR,
    DEFAULT_SEED,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
)
from jute_disease.utils.seed import seed_everything


def train():
    seed_everything(DEFAULT_SEED)
    load_dotenv()

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()

    wandb_logger = WandbLogger(entity="grade-descent", project="jute-disease-detection")
    datamodule = JuteDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    feature_extractor = DemoModel()
    model = JuteClassifier(feature_extractor=feature_extractor, lr=LEARNING_RATE)

    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="manual-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
            ),
            EarlyStopping(monitor="val_loss", patience=PATIENCE),
        ],
        max_epochs=MAX_EPOCHS,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
