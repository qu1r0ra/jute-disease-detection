from lightning.pytorch.cli import LightningCLI
from dotenv import load_dotenv


def main():
    load_dotenv()
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
