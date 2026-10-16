# %% [markdown]
# <a href="https://colab.research.google.com/github/qu1r0ra/jute-disease-detection/blob/Immern-patch-1/notebooks/reproducibility/02_Model_Selection_Training_ML.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Classical ML Model Selection and Training (Local)
#
# In this notebook, we will focus exclusively on Phase 1 testing and selection for Classical Machine Learning:
# - Validate heavy ML Baselines (Random Forest, SVM, Logistic Regression, KNN, Gaussian Naive Bayes).
# - Compare models on Raw Pixels vs Handcrafted Features (Color Histograms, LBP, GLCM, HOG).
# - Execute runs using the `train_ml.py`.

# %% [markdown]
# # Environment Setup

# %%
# !git clone https://github.com/qu1r0ra/jute-disease-detection.git
# %cd jute-disease-detection

# %pip install uv
# !uv pip install --system -e .
# !uv sync

# %% [markdown]
# If you encounter `ModuleNotFoundError`, you can simply restart the session and rerun the cell below.

# %%
# ruff: noqa: T201
from jute_disease.utils.constants import DEFAULT_SEED
from jute_disease.utils.seed import seed_everything

seed_everything(DEFAULT_SEED)

# %% [markdown]
# Mount your Google Drive to the Colab runtime.

# %%
from google.colab import drive

drive.mount("/content/drive")

# %% [markdown]
# If you haven't yet,
#
# 1. Download `data.zip` from <https://drive.google.com/drive/folders/1WoQ-Xzy0Prl9lInHW5JpGX4tpE9YDUua?usp=sharing> and upload it to your Google Colab account's Google Drive. You can simply upload it to the root of _My Drive_ for simplicity.
# 2. Update `DATA_ZIP_PATH` below to the path where you stored the file. If you uploaded it to the root of _My Drive_, you can set it to **"/content/drive/MyDrive/data.zip"**.

# %%
# %cd jute-disease-detection

# %%
from pathlib import Path

# Update DATA_ZIP_PATH to where data.zip is stored relative to the Colab VM filesystem.
# For organization, we stored ours in!uv pip install --system -e .
# "/content/drive/MyDrive/Colab Notebooks/Jute Leaf Disease/data.zip"
DATA_ZIP_PATH = "/content/drive/MyDrive/data.zip"
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

# %% [markdown]
# Let's cleanly construct the `train`, `val`, and `test` sub-folders inside `data/ml_split/` from your unzipped files.

# %%
# !uv run python src/jute_disease/data/utils.py split

# %% [markdown]
# To persist our training artifacts beyond the Colab VM, we can _symlink_ the `artifacts` folder directly to our Google Drive.

# %%
GDRIVE_PATH = Path(DATA_ZIP_PATH).parent
GDRIVE_ARTIFACTS = GDRIVE_PATH / "artifacts"
GDRIVE_ARTIFACTS.mkdir(parents=True, exist_ok=True)

LOCAL_ARTIFACTS = Path("artifacts")

if not LOCAL_ARTIFACTS.exists() and not LOCAL_ARTIFACTS.is_symlink():
    LOCAL_ARTIFACTS.symlink_to(GDRIVE_ARTIFACTS)
    print(f"Symlinked {LOCAL_ARTIFACTS.absolute()} to {GDRIVE_ARTIFACTS}")
else:
    print(f"{LOCAL_ARTIFACTS} already exists or is linked.")

# %% [markdown]
# Let us perform a quick sanity test to ensure all generated files show up inside your Google Drive folder containing your `data.zip`. If you see a generated `test.txt` file then you are all set to proceed.

# %%
test_file = LOCAL_ARTIFACTS / "test.txt"
test_file.write_text("ML model selection test.")

if (GDRIVE_ARTIFACTS / "test.txt").exists():
    print("Symlink worked.")
else:
    print("Symlink failed :<")

# %% [markdown]
# # Model Training Setup
#
# We will now proceed to training the baseline models for this project. We will be using a variety of techniques in order to preprocess and supplement our `by_class` data. This preprocessing step consists of (1) data cleaning and (2) data augmentations. Thankfully, these two processes are already handled in the `src/jute_disease/data/transforms.py`, wherein the images are transformed to be set to 256x256 pixels for standardization. Additionally, it augments our image data using the **Albumentation** library in order to perform the following set of transformations:
# - Cropping
# - Geometry (Rotation and Shear)
# - Dropout (randomly drops rectangular regions from the image)
# - Color and Lighting (Brightness and Contrast)
# - Noise and Blur (Grain and Blur)
#
# The `transforms.py` routine is called when training a model, hence we may jump straight into the training process.
#
#

# %% [markdown]
#
# To do this, we will be conducting experiments in classifying the images from the `by_class` folder using common image classification models. Specifically, the Classical ML models that we will be using are as follows:
# - K-Nearest Neighbors (KNN)
# - Logistic Regression (LR)
# - Gaussian Naive Bayes (GNB)
# - Random Forests (RF)
# - Support Vector Machines (SVM)
#
# The training process will consist of three key phases:
# 1. Feature Extraction
# 2. Classical Model Baselines
# 3. Results and Model Selection

# %% [markdown]
# ## 1. Feature Extraction Pipeline
# The feature extractions process consists of two types of extractions:
# - **Raw**: flattened pixels
# - **Crafted**: HOG, LBP, HSV, and GLCM
#
# We expand on the crafted features as follows:
# - **Histogram of Oriented Gradients (HOG)** - Captures the structure or the shape of an object by analyzing the distribution of gradient orientations in localized portions of an image (GeeksforGeeks, 2025).
# - **Local Binary Pattern (LBP)** - Encode the local texture information into binary patterns (Local Binary Pattern (LBP) Optimization for Feature Extraction, 2020)
# - **Hue, Saturation, Value (HSV)**
#   - Hue - represents the color as a value that ranges from 0 to 360 degrees
#   - Saturation - represents the quantity of the respective color added
#   - Value - represents the brightness concerning the saturation of the color
# - **Color Moments**
#   - Provides global representation of color distribution through the mean and standard deviation of the HSV channels.  
# - **Gray Level Co-occurrence Matrix (GLCM)** - A statistical method of examining texture that considers the spatial relationship of pixels.
#
# These features were selected in accordance to a review, specifically, Javidan et al., (2024). Based on our EDA, we observed that the symptoms of the diseases are characterized by color, texture, and shape which reinforced our decision in the use of our crafted features.
#
# We rely on `jute_disease.models.ml.features` to extract and cache these datasets.
#
# %% [markdown]
# ## 2. Classical Model Baselines
#
# To assess the model baselines, we will be running our `train all_ml` script that handles the entire training process of the classical Ml models as follows:
# Transform and Standardize Data -> Extract Features -> Train the size models -> Save Results to `WANDB`

# %%
import os

# CHANGE THIS
os.environ["WANDB_API_KEY"] = ""

# !uv run python scripts/train_all_ml.py

# %% [markdown]
# ## 3. Results and Model Selection
# We can see our results from the training in [here](https://wandb.ai/grade-descent/jute-disease-detection/table). Run the code below to see the validation results from our original run of the training process.

# %%
import pandas as pd
import wandb

# predefined run ids
runs_to_track = {
    "Gaussian Naive Bayes - Crafted": "ekubs2of",
    "K-Nearest Neighbors - Crafted": "s7o2w865",
    "Logistic Regression- Crafted": "g0defq70",
    "Random Forest - Crafted": "a1vp3q52",
    "Support Vector Machines - Crafted": "5g1zzwvm",
    "Gaussian Naive Bayes - Raw": "07vcu671",
    "K-Nearest Neighbors - Raw": "woay9dn5",
    "Logistic Regression- Raw": "wqbi5m5n",
    "Random Forest - Raw": "j8yayf3h",
    "Support Vector Machines - Raw": "y31voxm0",
}

ENTITY = "grade-descent"
PROJECT = "jute-disease-detection"


def get_summary_metrics(run_dict):
    api = wandb.Api()
    summary_list = []

    for model_name, run_id in run_dict.items():
        try:
            run_path = f"{ENTITY}/{PROJECT}/{run_id}"
            run = api.run(run_path)

            metrics = {
                "Model": model_name,
                "Val Acc": run.summary.get("val_acc", "N/A"),
                "Val F1": run.summary.get("val_f1", "N/A"),
            }
            summary_list.append(metrics)

        except Exception as e:
            print(f"Error fetching run {run_id} ({model_name}): {e}")

    return pd.DataFrame(summary_list)


df_results = get_summary_metrics(runs_to_track)

styled_results = df_results.style.format(
    {
        "Val Acc": "{:.4f}"
        if isinstance(df_results["Val Acc"].iloc[0], float)
        else "{}",
        "Val F1": "{:.4f}" if isinstance(df_results["Val F1"].iloc[0], float) else "{}",
    }
)

display(styled_results)

# %% [markdown]
# Using the metrics described from Weights & Biases, we can see that the best performing baseline models were SVM and RF in terms of accuracy and F1-Macro by far as compared to the other models. These two models will then proceed to model analaysis and tuning stage in the next notebook.

# %% [markdown]
# ## References:
# - Everton Gomede, PhD. (2024, February 19). Gray-Level Co-Occurrence Matrix (GLCM): A Comprehensive Analysis. Medium; Ai monks.io. https://medium.com/aimonks/gray-level-co-occurrence-matrix-glcm-a-comprehensive-analysis-a58bbc49c0f6
# - GeeksforGeeks. (2025, June 13). Histogram of Oriented Gradients. GeeksforGeeks. https://www.geeksforgeeks.org/computer-vision/histogram-of-oriented-gradients/
# - GeeksforGeeks. (2022, August 20). HSV Color Model in Computer Graphics. GeeksforGeeks. https://www.geeksforgeeks.org/computer-graphics/hsv-color-model-in-computer-graphics/
# - Javidan, S. M., Banakar, A., Rahnama, K., Vakilian, K. A., & Ampatzidis, Y. (2024). Feature engineering to identify plant diseases using image processing and artificial intelligence: A comprehensive review. Smart Agricultural Technology, 8, 100480. https://doi.org/10.1016/j.atech.2024.100480
# - Local Binary Pattern (LBP) Optimization for Feature Extraction. (2020). Arxiv.org. https://arxiv.org/html/2407.18665v1
#
