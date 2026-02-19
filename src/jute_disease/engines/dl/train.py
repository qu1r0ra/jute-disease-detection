from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI


def main():
    load_dotenv()
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
