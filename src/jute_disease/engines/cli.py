from lightning.pytorch.cli import LightningCLI

from jute_disease.utils.wandb_utils import setup_wandb


def main():
    setup_wandb()
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
