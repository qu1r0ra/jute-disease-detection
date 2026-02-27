# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploratory Data Analysis
#
# In this notebook, we will
# - set up the Colab runtime environment
# - mount our Google Drive to access jute data
# - download PlantVillage, PlantDoc data
# - *etc

# %% [markdown]
# ## 1. Environment Setup

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import sys
from pathlib import Path

project_root = Path("../../").resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# %%
import os

from google.colab import drive
from jute_disease.utils.data import download_plant_village
from torchvision.datasets import ImageFolder

from jute_disease.utils.constants import BY_CLASS_DIR, DATA_DIR, DEFAULT_SEED
from jute_disease.utils.logger import get_logger
from jute_disease.utils.seed import seed_everything

# %%
logger = get_logger(__name__)
seed_everything(DEFAULT_SEED)

# %%
# !git clone https://github.com/qu1r0ra/jute-disease-detection.git
# %cd jute-disease-detection

# %pip install uv

# Install dependencies into the Colab runtime, not into a virtual environment (.venv)
logger.info("Installing dependencies with uv...")
# !uv pip install --system -e .
# !uv sync

# Mount your Google Drive into the Colab runtime
drive.mount("/content/drive")

# %% [markdown]
# ## 2. Prepare Datasets
#
# Steps
# 1. Download `data.zip` from <https://drive.google.com/drive/folders/1WoQ-Xzy0Prl9lInHW5JpGX4tpE9YDUua?usp=sharing> and upload it to your Google Colab account's Google Drive. You can simply upload it to the root of _My Drive_ for simplicity.
# 2. Update `DATA_ZIP_PATH` below to the path where you stored the file. If you uploaded it to the root of _My Drive_, you can set it to **"/content/drive/MyDrive/data.zip"**.

# %%
# Update this to where you stored data.zip in your GDrive.
# For organization, we stored ours in
# "/content/drive/MyDrive/Colab/Jute Leaf Disease/data.zip"
DATA_ZIP_PATH = "/content/drive/MyDrive/data.zip"

if os.path.exists(DATA_ZIP_PATH):
    logger.info(f"Unzipping {DATA_ZIP_PATH}...")
    # !unzip -q -n "$DATA_ZIP_PATH" -d .
    logger.info("Data unpacked.")
else:
    logger.warning(
        f"Zip file not found at {DATA_ZIP_PATH}. "
        "Please check the path or upload your data."
    )

# %% [markdown]
# Let us also download the _Plant Village_ dataset from Kaggle. We already created a script to download it and consolidate it to `data/plant_village` for organization.

# %%
download_plant_village()

# %% [markdown]
# At this point, we have downloaded and prepared two datasets:
# 1. **PlantVillage**: a general dataset of diseased leaf images
# 2. our curated Jute leaf disease and pest dataset compiled from different sources
#
# We will be using PlantVillage for multistage transfer learning and our curated dataset for fine-tuning. Our end goal is to train a model that performs excellently on our curated Jute leaf disease and pest dataset.

# %%
jute_dataset = ImageFolder(root=BY_CLASS_DIR)
logger.info(f"Jute Dataset ({BY_CLASS_DIR}):")
logger.info(f"  - Total Images: {len(jute_dataset)}")
logger.info(f"  - Classes: {len(jute_dataset.classes)}")

plant_village_dir = DATA_DIR / "plant_village"
if plant_village_dir.exists():
    pv_dataset = ImageFolder(root=plant_village_dir)
    logger.info(f"\nPlantVillage Dataset ({plant_village_dir}):")
    logger.info(f"  - Total Images: {len(pv_dataset)}")
    logger.info(f"  - Classes: {len(pv_dataset.classes)}")
else:
    logger.warning("\nPlantVillage dataset not found! Check download step.")

# %% [markdown]
# To be continued
