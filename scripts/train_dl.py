from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI

from jute_disease.data.datamodule import DataModule
from jute_disease.models.dl.classifier import Classifier


def main() -> None:
    """Entry point for the unified Deep Learning CLI."""
    load_dotenv()
    LightningCLI(
        model_class=Classifier,
        datamodule_class=DataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
