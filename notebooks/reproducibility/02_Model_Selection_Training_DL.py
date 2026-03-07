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
# You may have noticed that the Deep Learning (DL) and the Classical Machine Learning (ML) workflows were separated into different notebooks. We deliberately separated them to allow for independent work. Specifically, I, CJ ([@qu1r0ra](https://github.com/qu1r0ra)) focused on DL while my friend Imman ([@Immern](https://github.com/Immern)) focused on Classical ML.
#
# You may have also noticed that each notebook has a corresponding `.py` script. This is to allow for better version control for `.ipynb` files, which are notoriously difficult to track with Git.
#  
# That said, in this notebook, we will focus on training DL models experimentally. Specifically, we will do the ff.:
# - Conduct transfer learning on the ff. chosen Deep Learning architectures pretrained on _ImageNet-1K_ as baseline DL models:
#   - **EfficientNet-B5**
#   - **EfficientNet-B7** (not included initially but later added out of curiosity to compare with B5)
#   - **Inception v3**
#   - **MobileNet V2**
#   - **MobileViT (small)**
#   - **ResNet-50**
# - Decide on a baseline model to move forward with
# - Get checkpoints for different levels of fine-tuning on the chosen model:
#   - _Level 1_: ImageNet (pre-trained)
#   - _Level 2_: ImageNet (pre-trained) -> PlantVillage (fine-tuning)
#   - _Level 3_: ImageNet (pre-trained) -> PlantVillage (fine-tuning) -> PlantDoc (fine-tuning)
# - Conduct grid search on the chosen model with the ff. hyperparameters/settings:
#   - _Dropout rate_: 0.0, 0.1, 0.2
#   - _Checkpoints_: Levels 1, 2, 3
#
# **Note:**
#
# This notebook is expected to be executed in **Google Colab**, just as I did. You can click the 'Open in Colab' button above this cell. Specifically, I used Colab's L4 GPU.

# %% [markdown] id="b04a890a"
# ## Environment Setup

# %% [markdown]
# Let's first clone the GitHub repository containing the code then install its dependencies.

# %% id="6930feae"
# !git clone https://github.com/qu1r0ra/jute-disease-detection.git
# %cd jute-disease-detection

# %pip install uv
# !uv pip install --system -e .
# !uv sync

# %% [markdown] id="26255b47"
# Let's seed the environment for reproducibility.
#
# > If you encounter `ModuleNotFoundError`, you can simply restart the session and rerun the cell below.

# %% id="cd6910a8"
# ruff: noqa: T201
from jute_disease.utils.constants import DEFAULT_SEED
from jute_disease.utils.seed import seed_everything

seed_everything(DEFAULT_SEED)

# %% [markdown] id="69363899"
# Before proceeding,
#
# 1. Download `data.zip` from <https://drive.google.com/drive/folders/1WoQ-Xzy0Prl9lInHW5JpGX4tpE9YDUua?usp=sharing> and upload it to your Google Colab account's Google Drive. You can simply upload it to the root of _My Drive_ for simplicity, but we recommend creating a separate folder for organization.
# 2. Update `DATA_ZIP_PATH` below to the path where you stored the file. If you uploaded it to the root of _My Drive_, you can set it to **"/content/drive/MyDrive/data.zip"**.

# %% id="7caa248a"
from pathlib import Path

# Update DATA_ZIP_PATH to where data.zip is stored relative to the Colab VM filesystem.
# For organization, we stored ours in
# "/content/drive/MyDrive/Colab Notebooks/Jute Leaf Disease/data.zip"
DATA_ZIP_PATH = "/content/drive/MyDrive/Colab Notebooks/Jute Leaf Disease/data.zip"
DEST_PATH = Path("data/by_class")

if Path(DATA_ZIP_PATH).exists():
    DEST_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {DATA_ZIP_PATH} to {DEST_PATH}...")
    # !unzip -q -n "$DATA_ZIP_PATH" -d "$DEST_PATH"
    print("Data unzipped.")
else:
    print(
        f"Zip file not found at {DATA_ZIP_PATH}. "
        "Please check the path or upload your data."
    )

# %% [markdown] id="20a1a666"
# After following the instructions above, let us mount our Google Drive to the Colab runtime. This is necessary to access the Jute data (`data.zip`) and to persist training artifacts such as model checkpoints and logs beyond the Colab VM's runtime.
#
# > You may be prompted to permit access; please do so.

# %% id="61b55c67"
from google.colab import drive

drive.mount("/content/drive")

# %% id="oLY0LIHFw3nV"
# %cd jute-disease-detection

# %% [markdown] id="ccaff569"
# Let's construct the `train`, `val`, and `test` sub-folders inside `data/ml_split/` from the unzipped data.
#
# > Throughout the notebooks, you will see scripts like this being executed. We greatly modularized our code so that notebooks merely serve as a presentation layer with the specifics abstracted away by the codebase. If you want to find out what's happening under the hood, feel free to inspect the the codebase.

# %% id="8dcf08eb"
# !uv run python src/jute_disease/data/utils.py split

# %% [markdown] id="849b7c47"
# To persist our training artifacts beyond the Colab VM, we can _symlink_ the project's `artifacts` folder to our Google Drive.

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
# Let's perform a quick sanity test to ensure all generated files show up inside your Google Drive folder containing your `data.zip`. If you see a generated `test.txt` file then you are all set to proceed.

# %% id="7965401e"
test_file = LOCAL_ARTIFACTS / "test.txt"
test_file.write_text("Hacking into the mainframe.")

if (GDRIVE_ARTIFACTS / "test.txt").exists():
    print("Symlink worked.")
else:
    print("Symlink failed :<")

# %% [markdown] id="9d77d540"
# ## Fine-Tuning and Transfer Learning Setup
#
# We saw from EDA that our dataset is pretty small (2382 images across 6 classes) for an image recognition task. To address our limitation in data, we decided to employ fine tuning and transfer learning as key techniques for our DL experiments.
#
# First, what is the difference between fine-tuning and transfer learning? Both involve a pre-trained model, but fine-tuning trains part or all of the pre-trained model's layers whereas transfer learning only trains the final layer, freezing the rest (GeeksforGeeks, 2025).
#
# As a part of this experiment, we will employ the ff. 'levels' of fine-tuning and transfer learning:
# 1. **Level 1**: ImageNet (pre-trained) -> our Jute dataset (transfer learning)
# 2. **Level 2**: ImageNet (pre-trained) -> PlantVillage (fine-tuning) -> our Jute dataset (transfer learning)
# 3. **Level 3**: ImageNet (pre-trained) -> PlantVillage (fine-tuning) -> PlantDoc (fine-tuning) -> our Jute dataset (transfer learning)
#
# An intuition as to why this may work can be formed from the task of teaching a Tagalog-speaking Filipino the Spanish language. While we could just teach them Spanish directly, it may be more effective to first teach them Chavacano (a Spanish-based language spoken in the Philippines) before teaching them Spanish. Perhaps learning Chavacano first will make learning Spanish more effective, leading to a greater Spanish proficiency at the end.
#
# Thus, we hope that transfer learning will enable our deep learning models to adapt general patterns learned from ImageNet objects to the domain of leaf disease detection. We are also curious as to whether fine-tuning on related datasets such as PlantVillage and PlantDoc first before conducting transfer learning on our dataset will improve performance.
#
# Enough yapping, let's experiment!
# %% [markdown] id="3ba92b2d"
# ## Model Training
#
# Before we proceed with actual model training, it's always a good idea to perform a 'trial run' for each of our models. In our case, a trial run will consist of running a single train, validation, and test epoch just to make sure our models don't crash unexpectedly in the middle of training. It has happened to me several times in the past and I do not want to experience it again.
#
# Fortunately, PyTorch Lightning (a PyTorch wrapper) supports this with their `fast_dev_run` capability. This is what the script below will do.

# %% id="c99d0d78"
# !uv run python scripts/train_all_dl_check.py

# %% [markdown] id="a5624dad"
# Having verified that the fast dev run works, we can now conduct transfer learning on our chosen DL architectures pretrained on ImageNet-1K.
#
# Before running the script below, make sure you have a [Weights and Biases](https://wandb.ai/site) account and an API key for it so you can track our experiments. You will be prompted to enter your API key.

# %% id="d32c265a"
# !uv run python scripts/train_all_dl.py

# %% [markdown]
# At this point, we have finished training the DL baselines. If training went well, you should have obtained similar results with us, which can be viewed in our [Weights and Biases project](https://wandb.ai/grade-descent/jute-disease-detection).
#
# You may notice there are multiple runs under the project. For now, you can focus on the ff. runs (which should also be present in a similar-named project):
# - efficientnet_b5
# - efficientnet_b7
# - inception_v3
# - mobilenet_v2
# - mobilevit_s
# - resnet_50

# %% [markdown]
# > explain why we chose MobileNet V2 over the rest, add a visualization comparing them

# %% [markdown] id="b53753d7"
# That said, we can now proceed with obtaining our level 2 and level 3 checkpoints for **MobileNet V2**, which will be used for the grid search. Let's download the _PlantVillage_ and _PlantDoc_ datasets from Kaggle.

# %% id="4ccb68e2"
from jute_disease.data.download import download_plant_doc, download_plant_village

download_plant_village()
download_plant_doc()

# %% [markdown] id="b44de4eb"
# ### Level 2 Checkpoint
#
# `ImageNet (pre-trained) -> PlantVillage (fine-tuning)`

# %% id="bc2b9c26"
# !uv run python src/jute_disease/engines/dl/pretrain.py \
#   --data_dir data/external/plantvillage \
#   --output_path artifacts/checkpoints/pretrained/mobilenet_v2-plantvillage.ckpt

# %% [markdown] id="8edb47106e1a46a883d545849b8ab81b"
# ### Level 3 Checkpoint
#
# Note the `--base_weights` argument. We are effectively resuming from the Level 2 checkpoint.
#
# `ImageNet (pre-trained) -> PlantVillage (fine-tuning) -> PlantDoc (fine-tuning)`

# %% id="69798b96"
# !uv run python src/jute_disease/engines/dl/pretrain.py \
#   --data_dir data/external/plantdoc \
#   --base_weights artifacts/checkpoints/pretrained/mobilenet_v2-plantvillage.ckpt \
#   --output_path artifacts/checkpoints/pretrained/mobilenet_v2-plantvillage-plantdoc.ckpt

# %% [markdown] id="a1cf8c02"
# > continue here
#
# **Grid search on different levels of transfer learning on MobileNet V2**

# %% id="32809cc5"
# !uv run python scripts/run_grid_search.py configs/grid/mobilenet_v2_grid.yaml

# %% [markdown]
# ## Extras

# %%
# !make train-dl-512

# %% [markdown]
# ## References
#
# [1] GeeksforGeeks. (2025, December 17). _Difference between fine-tuning and transfer learning_. <https://www.geeksforgeeks.org/machine-learning/what-is-the-difference-between-fine-tuning-and-transfer-learning/>
