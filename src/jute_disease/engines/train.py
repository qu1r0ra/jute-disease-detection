from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from jute_disease.data.jute_datamodule import JuteDataModule
from jute_disease.models.dl_backbones.mobilevit import MobileViT
from jute_disease.models.jute_classifier import JuteClassifier
from jute_disease.utils.constants import (
    BATCH_SIZE,
    DEFAULT_SEED,
    LEARNING_RATE,
    MAX_EPOCHS,
    ML_SPLIT_DIR,
    PATIENCE,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from jute_disease.utils.seed import seed_everything
from jute_disease.utils.wandb_utils import setup_wandb


def train():
    seed_everything(DEFAULT_SEED)
    setup_wandb()

    wandb_logger = WandbLogger(entity=WANDB_ENTITY, project=WANDB_PROJECT)
    datamodule = JuteDataModule(data_dir=ML_SPLIT_DIR, batch_size=BATCH_SIZE)
    feature_extractor = MobileViT()
    model = JuteClassifier(feature_extractor=feature_extractor, lr=LEARNING_RATE)

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
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
