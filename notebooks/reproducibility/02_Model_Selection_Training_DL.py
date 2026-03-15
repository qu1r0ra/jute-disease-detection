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
# # Deep Learning - Model Selection and Training
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
#   - `Level 1`: ImageNet (pre-trained)
#   - `Level 2`: ImageNet (pre-trained) -> PlantVillage (fine-tuning)
#   - `Level 3`: ImageNet (pre-trained) -> PlantVillage (fine-tuning) -> PlantDoc (fine-tuning)
# - Conduct grid search on the chosen model with the ff. hyperparameters/settings:
#   - Dropout rate: `0.0`, `0.1`, `0.2`
#   - Checkpoint: `Levels 1, 2, 3`
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

# %% [markdown] id="20a1a666"
# After following the instructions above, let us mount our Google Drive to the Colab runtime. This is necessary to access the Jute data (`data.zip`) and to persist training artifacts such as model checkpoints and logs beyond the Colab VM's runtime.
#
# > You may be prompted to permit access; please do so.

# %% id="61b55c67"
from google.colab import drive

drive.mount("/content/drive")

# %% [markdown]
# Let's unzip `data.zip` which contains our merged Jute leaf disease data.

# %% id="oLY0LIHFw3nV"
# %cd jute-disease-detection

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

# %% [markdown] id="ccaff569"
# Let's construct the `train`, `val`, and `test` sub-folders inside `data/ml_split/` from the unzipped data.
#
# > Throughout the notebooks, you will see scripts like this being executed. We greatly modularized our code so that notebooks merely serve as a presentation layer with the specifics abstracted away by the codebase. If you want to find out what's happening under the hood, feel free to inspect the the codebase.

# %% id="8dcf08eb"
# !uv run python src/jute_disease/data/utils.py split

# %% [markdown] id="849b7c47"
# To persist our training artifacts beyond the Colab VM, we can _symlink_ the project's `artifacts` folder to our Google Drive.

# %% id="c644e78d"
import shutil

GDRIVE_PATH = Path(DATA_ZIP_PATH).parent
GDRIVE_ARTIFACTS = GDRIVE_PATH / "artifacts"
GDRIVE_ARTIFACTS.mkdir(parents=True, exist_ok=True)

LOCAL_ARTIFACTS = Path("artifacts")

if not LOCAL_ARTIFACTS.is_symlink():
    if LOCAL_ARTIFACTS.exists():
        shutil.copytree(LOCAL_ARTIFACTS, GDRIVE_ARTIFACTS, dirs_exist_ok=True)
        shutil.rmtree(LOCAL_ARTIFACTS)
    LOCAL_ARTIFACTS.symlink_to(GDRIVE_ARTIFACTS)
    print(f"Symlinked {LOCAL_ARTIFACTS.absolute()} to {GDRIVE_ARTIFACTS}")
else:
    print(f"{LOCAL_ARTIFACTS} is already linked.")

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
# Before running the script below, we highly recommend you to create a [Weights and Biases](https://wandb.ai/site) (WandB) account to track the experiments. If you choose to log in to a WandB account, you will be prompted to enter an API key (see <https://docs.wandb.ai/models/quickstart>).

# %% id="d32c265a"
# !uv run python scripts/train_all_dl.py

# %% [markdown]
# At this point, we have finished training the DL baselines. If training went well, you should have obtained results similar to ours, which can be viewed [here](https://wandb.ai/grade-descent/jute-disease-detection/groups/Baseline%20DL%20Models/workspace).
#
# Looking at the validation F1 graph:
#
# ![Validation F1 Score Comparison: Baseline DL Models](../../assets/figures/dl/val_f1_baseline.png)
#
# For context, we used the ff. pre-trained models:
#
# | Architecture | Hugging Face Model Name | Parameters |
# | :--- | :--- | :--- |
# | EfficientNet-B7 | [tf_efficientnet_b7.ns_jft_in1k](https://huggingface.co/timm/tf_efficientnet_b7.ns_jft_in1k) | ~66.35M |
# | EfficientNet-B5 | [efficientnet_b5.sw_in12k_ft_in1k](https://huggingface.co/timm/efficientnet_b5.sw_in12k_ft_in1k) | ~30.39M |
# | ResNet-50 | [resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k) | ~25.56M |
# | Inception v3 | [inception_v3.tv_in1k](https://huggingface.co/timm/inception_v3.tv_in1k) | ~23.83M |
# | MobileViT (small) | [mobilevit_s.cvnets_in1k](https://huggingface.co/timm/mobilevit_s.cvnets_in1k) | ~5.58M |
# | MobileNet V2 | [mobilenetv2_100.ra_in1k](https://huggingface.co/timm/mobilenetv2_100.ra_in1k) | ~3.50M |
#
# Note that the Hugging Face models above may have been pre-trained in different environments and with different techniques. Still, we made sure all our models were eventually pre-trained on ImageNet, hence the `_in1k` suffixes.

# %% [markdown]
# Some insights:
# - **EfficientNet-B5** achieved the greatest top-1 validation F1. It has the second most parameters, so it is somewhat expected.
# - It is followed by **MobileNet V2**. Interestingly, MobileNetV2 achieved a performance comparable to EfficientNet-B5 despite having the least parameters. Hence, we decided to push through with MobileNet V2 for grid search. It is also more economical for us.
# - **MobileViT (small)** achieved the third-greatest top-1 validation F1. It has the second-least parameters.
# - Interestingly, **EfficientNet-B7** achieved the worst top-1 validation F1 despite having a similar architecture to EfficientNet-B5, but with more than double its size (and thus, the most parameters).

# %% [markdown] id="b53753d7"
# ## Checkpoints for Grid Search
#
# That said, we can now proceed with obtaining our level 2 and level 3 checkpoints for our chosen DL architecture, **MobileNet V2**. These checkpoints will be used for the grid search. Let's first download the [PlantVillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) and [PlantDoc](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset) datasets from Kaggle.

# %% id="4ccb68e2"
from jute_disease.data.download import download_plant_doc, download_plant_village

download_plant_village()
download_plant_doc()

# %% [markdown] id="b44de4eb"
# ### Level 2 Checkpoint
#
# `ImageNet (pre-trained) -> PlantVillage (fine-tuning)`
#
# For the fine-tuning stages, we will not freeze any parameters. Moreover, we will discard the classifier head and replace it with a new one with the number of neurons equal to the number of classes for the PlantVillage dataset (38).

# %% id="bc2b9c26"
# !uv run python src/jute_disease/engines/dl/pretrain.py \
#   --data_dir data/external/plantvillage \
#   --output_path artifacts/checkpoints/pretrained/mobilenet_v2-plantvillage.ckpt

# %% [markdown] id="8edb47106e1a46a883d545849b8ab81b"
# ### Level 3 Checkpoint
#
# `ImageNet (pre-trained) -> PlantVillage (fine-tuning) -> PlantDoc (fine-tuning)`
#
# Note the `--base_weights` argument. We are effectively resuming from the level 2 checkpoint.
#
# Like level 2, we will not freeze any parameters and we will discard the classifier head, replacing it with a new one with the number of neurons equal to the number of classes for the PlantDoc dataset (27).

# %% id="69798b96"
# !uv run python src/jute_disease/engines/dl/pretrain.py \
#   --data_dir data/external/plantdoc \
#   --base_weights artifacts/checkpoints/pretrained/mobilenet_v2-plantvillage.ckpt \
#   --output_path artifacts/checkpoints/pretrained/mobilenet_v2-plantvillage-plantdoc.ckpt

# %% [markdown] id="a1cf8c02"
# Now that we have level 2 and level 3 checkpoints, we can proceed with the grid search.

# %% id="32809cc5"
# !uv run python scripts/run_grid_search.py configs/grid/mobilenet_v2_grid.yaml

# %% [markdown]
# Our results for the MobileNet V2 grid search can be viewed [here](https://wandb.ai/grade-descent/jute-disease-detection/groups/MobileNet%20V2%20Grid/workspace).
#
# Looking at the graph of validation accuracy over global steps:
#
# ![MobileNet V2 Grid Validation Accuracy Graph](../../assets/figures/dl/val_acc_mobilenet_v2_grid.png)
#

# %% [markdown]
# Oof, that's pretty annoying to analyze. Maybe looking at the test accuracy graph and a list of models sorted by test accuracy will help:
#
# > Such is the pain of relying solely on WandB visualizations. After our first full run and several other mistakes, we realized the importance of saving experiment logs and metrics into a local CSV or parquet file to give us flexibility to visualize data in our own way.
#
# ![MobileNet V2 Grid Test Accuracy Graph](../../assets/figures/dl/test_acc_mobilenet_v2_grid.png)
#
# ![MobileNet V2 Grid Models Sorted by Test Accuracy](../../assets/figures/dl/grid_models_sorted_test_acc.png)

# %% [markdown]
# For context, the test accuracy of each model was computed on the best checkpoint obtained during training. By default, PyTorch Lightning determines the best checkpoint as the one that achieved the lowest validation loss, though you could set the criterion yourself. We went with the default.
#
# Looking at the graphs, we got the ff. insights:
# - Level 1 MobileNet V2 checkpoints (MobileNet V2 pre-trained on ImageNet-1K with no fine-tuning whatsoever) led to the best test accuracies, followed by level 3 checkpoints (fine-tuned on PlantVillage then PlantDoc), and lastly, level 2 checkpoints (fine-tuned on PlantVillage). This went against our hypothesis that fine-tuning on related datasets would improve model performance.
# - Within the same checkpoint groups:
#   - Dropout rates of 0.0 and 0.1 led to greater test accuracies than their 0.2 counterparts. This may suggest that dropout rates of 0.2 or higher may hurt model performance, though this has yet to be statistically tested.
#   - A dropout rate of 0.1 led to the greatest test accuracies, followed by 0.0 and 0.2.
#
# Thus, we concluded that the MobileNet V2 pre-trained on ImageNet-1K with a dropout rate of 0.1 was the best DL model to push through with (and potentially continue fine-tuning on) for our task of jute leaf disease classification.
#
# At this point, we decided to conduct error analysis and see what can be improved.

# %% [markdown]
# ## Extras

# %% [markdown]
# If you inspect the codebase, you will notice that the image preprocessing pipeline includes cropping the images to 256x256 pixels for standardization. Our intuition told us that training with higher-resolution images may improve MobileNet V2 performance. We got curious, so we trained a level 1 MobileNet V2 with dropout rates 0.0 and 0.1 but with images cropped to 512x512 pixels. We wanted to compare their performance to their 256x256 pixel counterparts.

# %%
# !uv run python scripts/train_dl_512.py

# %% [markdown]
# Our results can be viewed [here](https://wandb.ai/grade-descent/jute-disease-detection/groups/MobileNet%20V2%20512px/workspace). They will be compared against their 256x256 pixel counterparts in the [next DL notebook](./03_Model_Analysis_Tuning_DL.py).

# %% [markdown]
# ## References
#
# [1] GeeksforGeeks. (2025, December 17). _Difference between fine-tuning and transfer learning_. <https://www.geeksforgeeks.org/machine-learning/what-is-the-difference-between-fine-tuning-and-transfer-learning/>
