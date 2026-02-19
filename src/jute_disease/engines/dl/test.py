from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI


def main():
    load_dotenv()
    # Test subcommand
    LightningCLI(save_config_kwargs={"overwrite": False})


if __name__ == "__main__":
    main()
