import argparse
from pathlib import Path
from shutil import copyfile

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from jute_disease.data import DataModule
from jute_disease.models.dl import Classifier, MobileViT
from jute_disease.utils import get_logger, seed_everything

logger = get_logger(__name__)


def train_pretext_task(
    data_dir: Path | str,
    output_path: Path | str,
    pretrained: bool = True,
    base_backbone_weights: dict[str, torch.Tensor] | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
) -> None:
    """
    Trains MobileViT on a given dataset and saves the checkpoint.
    """
    seed_everything(seed)

    # 1. Setup Data
    logger.info(f"Initializing DataModule for {data_dir}...")
    dm = DataModule(data_dir=data_dir, val_split=0.2, batch_size=batch_size, seed=seed)
    dm.setup()

    num_classes = dm.num_classes
    logger.info(f"Found {num_classes} classes.")

    # 2. Setup Backbone
    logger.info("Initializing MobileViT...")
    backbone = MobileViT(pretrained=pretrained)

    # Override weights if provided (e.g. from PlantVillage checkpoint)
    if base_backbone_weights is not None:
        logger.info("Loading custom backbone weights...")
        # Note: base_backbone_weights should be the state_dict of the BACKBONE only
        msg = backbone.load_state_dict(base_backbone_weights, strict=True)
        logger.info(f"Backbone load status: {msg}")

    # 3. Setup Classifier
    model = Classifier(
        feature_extractor=backbone,
        num_classes=num_classes,
        lr=lr,
        freeze_backbone=False,  # Unfreeze for transfer learning
    )

    # 4. Setup Trainer
    task_name = Path(output_path).stem
    logger_wandb = WandbLogger(project="jute-pretraining", name=task_name)

    callbacks = [
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            filename=f"{task_name}-{{epoch:02d}}-{{val_acc:.4f}}",
            save_top_k=1,
            dirpath="artifacts/checkpoints/pretrained/",
        ),
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ]

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger=logger_wandb,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    # 5. Train
    logger.info("Starting Training...")
    trainer.fit(model, datamodule=dm)

    # 6. Save Best
    if isinstance(trainer.checkpoint_callback, ModelCheckpoint):
        best_path = trainer.checkpoint_callback.best_model_path
    else:
        best_path = None
    if best_path:
        logger.info(f"Best model saved at: {best_path}")
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(best_path, str(target_path))
        logger.info(f"Copied best model to target path: {output_path}")
    else:
        logger.error("Training failed or no checkpoint saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Where to save the final .ckpt"
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        help=(
            "Optional: Path to PREVIOUS stage checkpoint to resume from (e.g. "
            "PlantVillage ckpt)"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    backbone_weights: dict[str, torch.Tensor] | None = None

    if args.base_weights:
        logger.info(f"Extracting backbone weights from {args.base_weights}...")
        checkpoint = torch.load(args.base_weights, map_location="cpu")
        state_dict: dict[str, torch.Tensor] = checkpoint["state_dict"]

        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith("feature_extractor."):
                name = k.replace("feature_extractor.", "")
                backbone_dict[name] = v

        if backbone_dict:
            logger.info(f"Extracted {len(backbone_dict)} keys for backbone.")
            backbone_weights = backbone_dict
        else:
            logger.warning("No 'feature_extractor' keys found in base checkpoint.")

    train_pretext_task(
        data_dir=args.data_dir,
        output_path=args.output_path,
        pretrained=True,
        base_backbone_weights=backbone_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
