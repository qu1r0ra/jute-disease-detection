from lightning.pytorch.cli import LightningCLI

from jute_disease.data.jute_datamodule import JuteDataModule
from jute_disease.models.jute_classifier import JuteClassifier


def cli_main():
    _cli = LightningCLI(JuteClassifier, JuteDataModule)


if __name__ == "__main__":
    cli_main()
