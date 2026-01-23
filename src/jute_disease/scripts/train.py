import os

import wandb
from dotenv import load_dotenv
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.demos.model import DemoModel
from lightning.pytorch.loggers import WandbLogger

from jute_disease.data.jute_datamodule import JuteDataModule
from jute_disease.utils.config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS


def train():
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")

    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()

    wandb_logger = WandbLogger(project="jute-disease-detection")
    wandb_logger.experiment.config.update(
        {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
        }
    )

    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=1,
        precision="16-mixed",
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="best-model",
                monitor="val_loss",
                mode="min",
            ),
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ],
        logger=wandb_logger,
        max_epochs=NUM_EPOCHS,
    )
    model = DemoModel()  # NOTE: Replace with actual model
    datamodule = JuteDataModule()

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
