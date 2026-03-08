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
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/qu1r0ra/jute-disease-detection/blob/main/notebooks/reproducibility/02_Model_Selection_Training_DL.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="7fb27b941602401d91542211134fc71a"
# # Deep Learning Model Selection and Training
#
# `(update)`
# In this notebook, we will focus exclusively on Phase 1 testing and selection for Deep Learning:
# - Validate heavy Deep Learning Baselines (Inception v3, EfficientNet-B5, EfficientNet-B7, ResNet-50, MobileNetV2).
# - Perform a grid search across Transfer Learning Initialization Strategies for MobileNetV2.
# - Execute runs using the unified Lightning `train_dl.py` scripts and `run_grid_search.py`.
#
# **Note:**
#
# This notebook must be executed in **Google Colab**, just as we did. Specifically, we used Colab's L4 GPU.

# %% [markdown] id="b04a890a"
# ## Environment Setup

# %% id="6930feae"
# !git clone https://github.com/qu1r0ra/jute-disease-detection.git
# %cd jute-disease-detection

# %pip install uv
# !uv pip install --system -e .
# !uv sync

# %% [markdown] id="26255b47"
# If you encounter `ModuleNotFoundError`, you can simply restart the session and rerun the cell below.

# %% id="cd6910a8"
# ruff: noqa: T201
from jute_disease.utils.constants import DEFAULT_SEED
from jute_disease.utils.seed import seed_everything

seed_everything(DEFAULT_SEED)

# %% [markdown] id="20a1a666"
# Mount your Google Drive to the Colab runtime.

# %% id="61b55c67"
from google.colab import drive

drive.mount("/content/drive")

# %% [markdown] id="69363899"
# If you haven't yet,
#
# 1. Download `data.zip` from <https://drive.google.com/drive/folders/1WoQ-Xzy0Prl9lInHW5JpGX4tpE9YDUua?usp=sharing> and upload it to your Google Colab account's Google Drive. You can simply upload it to the root of _My Drive_ for simplicity.
# 2. Update `DATA_ZIP_PATH` below to the path where you stored the file. If you uploaded it to the root of _My Drive_, you can set it to **"/content/drive/MyDrive/data.zip"**.

# %% id="oLY0LIHFw3nV"
# %cd jute-disease-detection

# %% id="7caa248a"
from pathlib import Path

# Update DATA_ZIP_PATH to where data.zip is stored relative to the Colab VM filesystem.
# For organization, we stored ours in!uv pip install --system -e .
# "/content/drive/MyDrive/Colab Notebooks/Jute Leaf Disease/data.zip"
DATA_ZIP_PATH = "/content/drive/MyDrive/Colab Notebooks/Jute Leaf Disease/data.zip"
DEST_PATH = Path("data/by_class")

if Path(DATA_ZIP_PATH).exists():
    DEST_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {DATA_ZIP_PATH} to {DEST_PATH}...")
    # !unzip -q -n "$DATA_ZIP_PATH" -d "$DEST_PATH"
    print("Data unpacked.")
else:
    print(
        f"Zip file not found at {DATA_ZIP_PATH}. "
        "Please check the path or upload your data."
    )

# %% [markdown] id="ccaff569"
# Let's cleanly construct the `train`, `val`, and `test` sub-folders inside `data/ml_split/` from your unzipped files.

# %% id="8dcf08eb"
# !uv run python src/jute_disease/data/utils.py split

# %% [markdown] id="849b7c47"
# To persist our training artifacts beyond the Colab VM, we can _symlink_ the `artifacts` folder directly to our Google Drive.

# %% id="c644e78d"
GDRIVE_PATH = Path(DATA_ZIP_PATH).parent
GDRIVE_ARTIFACTS = GDRIVE_PATH / "artifacts"
GDRIVE_ARTIFACTS.mkdir(parents=True, exist_ok=True)

LOCAL_ARTIFACTS = Path("artifacts")

if not LOCAL_ARTIFACTS.exists() and not LOCAL_ARTIFACTS.is_symlink():
    LOCAL_ARTIFACTS.symlink_to(GDRIVE_ARTIFACTS)
    print(f"Symlinked {LOCAL_ARTIFACTS.absolute()} to {GDRIVE_ARTIFACTS}")
else:
    print(f"{LOCAL_ARTIFACTS} already exists or is linked.")

# %% [markdown] id="875aabf0"
# Let us perform a quick sanity test to ensure all generated files show up inside your Google Drive folder containing your `data.zip`. If you see a generated `test.txt` file then you are all set to proceed.

# %% id="7965401e"
test_file = LOCAL_ARTIFACTS / "test.txt"
test_file.write_text("Hacking into the mainframe.")

if (GDRIVE_ARTIFACTS / "test.txt").exists():
    print("Symlink worked.")
else:
    print("Symlink failed :<")

# %% [markdown] id="9d77d540"
# ## Transfer Learning Setup
#
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
# - ResNet-50
# - Inception v3
# - EfficientNet-B5
# - EfficientNet-B7
# - MobileNetV2
#
# `(i will polish later)`
# %% [markdown] id="3ba92b2d"
# ## Deep Learning Baselines (Level 1: ImageNet Only)
#
# We systematically train and evaluate our six chosen architectures on the `DATA_ZIP_PATH` jute splits. Every model's feature extractor initiates from ImageNet generic representations. We will freeze their backbones and only train the final custom dense classifiers.

# %% [markdown] id="65968e63"
# **Fast Dev Run Validation**
#
# First, we dispatch a rapid sanity check using PyTorch Lightning's `fast_dev_run` capability. This performs exactly 1 training and validation batch traversing through all 6 architectures. It mathematically verifies gradients flow properly without silently crashing an hour later!

# %% id="c99d0d78"
# !uv run python scripts/train_all_dl_check.py

# %% [markdown] id="a5624dad"
# **Execute 6 Deep Learning Baselines**
#
# Running the master sequential launcher. This autonomously `fits` and subsequently `tests` each `.yaml` model config entirely using your GPU.

# %% id="d32c265a"
# !uv run python scripts/train_all_dl.py

# %% [markdown] id="zezgVidiyti0"
# Let us fine-tune an EfficientNet-B7.

# %% id="IiNA4hRqyxnN"
# !make train-dl-single MODEL=efficientnet_b7

# %% [markdown] id="e3157de6"
# ## === Everything above is final ===


# %% [markdown] id="b53753d7"
# ## 2. MSTL Domain Initializations (Pre-training)
# We now download the massive `PlantVillage` and specialized `PlantDoc` datasets via KaggleHub to execute the Multi-Stage Transfer Learning on our Top 2 performing models.

# %% id="4ccb68e2"
from jute_disease.data.download import download_plant_doc, download_plant_village

download_plant_village()
download_plant_doc()

# %% [markdown] id="b44de4eb"
# **Pre-Train Top Model A on PlantVillage (Produces Level 2 Checkpoint)**
# We use our custom PyTorch Lightning pre-training script on PlantVillage. Early-stopping is implemented intrinsically (Defaults to 50 epochs, halts upon val_loss convergence).

# %% id="bc2b9c26"
# !uv run python src/jute_disease/engines/dl/pretrain.py \
#   --data_dir data/external/plantvillage \
#   --output_path artifacts/checkpoints/pretrained/mobilenet_v2-plantvillage.ckpt

# %% [markdown] id="8edb47106e1a46a883d545849b8ab81b"
# **Pre-Train Top Model A on PlantDoc (Produces Level 3 Checkpoint)**
# Note the `--base_weights` parameter: We resume *exactly* from the Level 2 checkpoint! This synthesizes the entire ImageNet -> PlantVillage -> PlantDoc hierarchy!

# %% id="69798b96"
# !uv run python src/jute_disease/engines/dl/pretrain.py \
#   --data_dir data/external/plantdoc \
#   --base_weights artifacts/checkpoints/pretrained/mobilenet_v2-plantvillage.ckpt \
#   --output_path artifacts/checkpoints/pretrained/mobilenet_v2-plantvillage-plantdoc.ckpt

# %% [markdown] id="a1cf8c02"
# **Grid search on different levels of transfer learning on MobileNet V2**
#
# Hoping this works.

# %% id="32809cc5"
# !uv run python scripts/run_grid_search.py configs/grid/mobilenet_v2_grid.yaml

# %% [markdown] id="72d060bf"
# ## 4. WandB Analysis
# From Weights & Biases, we can now deduce our Champion architectural baseline, as well as definitively prove whether or not Level 3 MSTL was superior computationally versus Level 1 generic pretraining.

# %% [markdown]
# ## Extras

# %%
# !make train-dl-512
