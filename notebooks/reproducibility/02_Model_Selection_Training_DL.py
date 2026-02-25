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
# # Deep Learning Model Selection and Training
#
# `(update)`
# In this notebook, we will focus exclusively on Phase 1 testing and selection for Deep Learning:
# - Validate heavy Deep Learning Baselines (Inception V3, VGG16, DenseNet201).
# - Perform a grid search across Transfer Learning Initialization Strategies for MobileViT.
# - Execute runs using the unified Lightning `train_dl.py` scripts and `run_grid_search.py`.

# %% [markdown]
# ## Environment Setup

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
# ruff: noqa: F821
from jute_disease.utils.constants import DEFAULT_SEED
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

# Mount your Google Drive into the Colab runtime
drive.mount("/content/drive")

# %% [markdown]
# If you haven't yet,
#
# 1. Download `data.zip` from <https://drive.google.com/drive/folders/1WoQ-Xzy0Prl9lInHW5JpGX4tpE9YDUua?usp=sharing> and upload it to your Google Colab account's Google Drive. You can simply upload it to the root of _My Drive_ for simplicity.
# 2. Update `DATA_ZIP_PATH` below to the path where you stored the file. If you uploaded it to the root of _My Drive_, you can set it to **"/content/drive/MyDrive/data.zip"**.

# %%
# Update this to where you stored data.zip in your GDrive.
# For organization, we stored ours in
# "/content/drive/MyDrive/Colab/Jute Leaf Disease/data.zip"
DATA_ZIP_PATH = "/content/drive/MyDrive/Colab/Jute Leaf Disease/data.zip"

if Path(DATA_ZIP_PATH).exists():
    logger.info(f"Unzipping {DATA_ZIP_PATH}...")
    # !unzip -q -n "$DATA_ZIP_PATH" -d .
    logger.info("Data unpacked.")
else:
    logger.warning(
        f"Zip file not found at {DATA_ZIP_PATH}. "
        "Please check the path or upload your data."
    )

# %% [markdown]
# We saw from EDA that our dataset is pretty small (2382 images across 6 classes) for an image recognition task. To address our limitation in data, we decided to employ transfer learning as a key technique (among others, such as data augmentation) for our deep learning experiments.
#
# (levels of transfer learning)
# 1. **Level 1**: ImageNet -> our Jute dataset
# 2. **Level 2**: ImageNet -> PlantVillage -> our Jute dataset
# 3. **Level 3**: ImageNet -> PlantVillage -> PlantDoc -> our Jute dataset
#
# Levels 2 and 3 are known as **multistage transfer learning (MSTL)**, which as the name suggests, is transfer learning with multiple stages. An analogy of TL vs. MSTL can be made in the task of teaching a Tagalog-speaking Filipino the Spanish language. While we could just teach them Spanish directly (TL), it may be more effective to first teach them Chavacano (a Spanish-based language spoken in the Philippines) before teaching them Spanish (MSTL). Perhaps learning Chavacano first will make learning Spanish a lot easier, leading to a greater Spanish proficiency by the end.
#
# Thus, we hope that transfer learning will enable our deep learning models to adapt general patterns learned from ImageNet objects to the domain of leaf disease detection. We are also curious as to whether utilizing MSTL with similar but general datasets such as PlantVillage and PlantDoc can improve performance.
#
# Specifically, we will experiment with six (6) established deep learning architectures:
# - InceptionV3
# - VGG16
# - DenseNet201
# - ResNet
# - MobileNet
# - MobileViT
#
# `(i will polish later)`

# %%

# %% [markdown]
# ## === Everything above is final ===

# %% [markdown]
# Let us download the datasets we chose for transfer learning:
# - **PlantVillage**: (short description)
# - **PlantDoc**: 

# %%
from jute_disease.data.download import download_plant_village, download_plant_doc

download_plant_village()
download_plant_doc()

# %% [markdown]
# **Pre-training MobileViT on PlantVillage (Level 2 Checkpoint)**
# We invoke the pre-training engine on the newly downloaded PlantVillage subset.

# %%
# !uv run python src/jute_disease/engines/dl/pretrain.py \
#   --data_dir data/external/plantvillage \
#   --output_path artifacts/checkpoints/pretrained/mobilevit_plantvillage.ckpt \
#   --epochs 5

# %% [markdown]
# **Pre-training MobileViT on PlantDoc (Level 3 Checkpoint)**
# *Note: We initialize this pre-training from the PlantVillage checkpoint we just created (hence ImageNet -> PlantVillage -> PlantDoc).*

# %%
# !uv run python src/jute_disease/engines/dl/pretrain.py \
#   --data_dir data/external/plantdoc \
#   --checkpoint_path artifacts/checkpoints/pretrained/mobilevit_plantvillage.ckpt \
#   --output_path artifacts/checkpoints/pretrained/mobilevit_plantvillage_plantdoc.ckpt \
#   --epochs 5

# %% [markdown]
# ## 1. Heavy DL Baselines
# Training InceptionV3, VGG16, and DenseNet201 to establish standard benchmarks using pure ImageNet pretraining.
#
# *Note: We will invoke `scripts/train_dl.py` via Python subprocess or bash commands (`!uv run python scripts/train_dl.py`)*

# %%

# %% [markdown]
# ## 2. MobileViT Transfer Learning Grid
# Evaluating MobileViT initialization strategies to test the impact of domain-adaptive pretraining:
# 1. **Level 1**: ImageNet -> Jute
# 2. **Level 2**: ImageNet -> PlantVillage -> Jute
# 3. **Level 3**: ImageNet -> PlantVillage -> PlantDoc -> Jute
#
# *Note: We will use `scripts/run_grid_search.py` leveraging `configs/grid/mobilevit_grid.yaml`*

# %%

# %% [markdown]
# ## 3. WandB Analysis
# Reviewing the logs generated by Weights & Biases to identify the "Champion" baseline and the most effective transfer learning initialization strategy.

# %%
