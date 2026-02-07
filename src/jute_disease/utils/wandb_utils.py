import os

from dotenv import load_dotenv

import wandb


def setup_wandb():
    """Load environment variables and login to WandB."""
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()
