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

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/qu1r0ra/jute-disease-detection/blob/main/notebooks/reproducibility/03_Model_Analysis_Tuning_DL.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Deep Learning - Model Analysis and Fine-Tuning
#
# In the [previous notebook](./02_Model_Selection_Training_DL.ipynb), we trained baseline DL models and chose a configuration to move forward with. We concluded with a MobileNet V2 pre-trained on ImageNet-1K with a dropout rate of 0.1 as our best model moving forward.
#
# In this notebook, we will analyze the model's performance and fine-tune chosen hyperparameters (not to be confused with fine-tuning for model training) in the hopes of improving its performance.
#
# Specifically, we will:
# - Compare the performance of our 256x256 pixel and 512x512 pixel MobileNet V2 models.
# - Visualize the training history to inspect for possible overfitting or underfitting.
# - Conduct error analysis by exploring latent space embeddings (t-SNE and UMAP) and inspecting the top confident errors.
# - Use Grad-CAM to visualize where the model focuses on in various images.
#
# **Notes:**
#
# - Like the [previous notebook](./02_Model_Selection_Training_DL.ipynb), this is expected to be executed in **Google Colab**. What Colab GPU to use is technically up to you, but we recommend sticking to whatever you used in the previous notebook for fair model comparison. In our case, we used an **L4**.
# - This time, we chose not to abstract the visualization logic into separate scripts to allow for quick inline revisions.
#
# ## Outline
#
# - [Environment Setup](#environment-setup)
# - [Part 1: Baseline Analysis](#part-1-baseline-analysis)
#   - [1A. Model Performance](#1a-model-performance)
#   - [1B. Error Analysis](#1b-error-analysis)
#   - [1C. Latent Space Analysis](#1c-latent-space-analysis)
#   - [1D. Interpretability](#1d-interpretability)
# - [Part 2: Fine-Tuning](#part-2-fine-tuning)
#   - [2A. Model Performance](#2a-model-performance)
#   - [2B. Error Analysis](#2b-error-analysis)
#   - [2C. Interpretability](#2c-interpretability)
# - [Conclusion](#conclusion)
# - [References](#references)

# %% [markdown]
# ## Environment Setup
#
# Let's run the environment setup again. Refer to the [previous notebook](./02_Model_Selection_Training_DL.ipynb) for detailed instructions and remarks.

# %%
# !git clone https://github.com/qu1r0ra/jute-disease-detection.git
# %cd jute-disease-detection

# %pip install uv
# !uv pip install --system -e .
# !uv sync

# %% [markdown]
# > If you encounter `ModuleNotFoundError`, you can simply restart the session and rerun the cell below.

# %%
# ruff: noqa: T201
from jute_disease.utils.constants import DEFAULT_SEED
from jute_disease.utils.seed import seed_everything

seed_everything(DEFAULT_SEED)

# %%
from google.colab import drive

drive.mount("/content/drive")

# %%
# %cd jute-disease-detection

# %%
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

# %%
# !uv run python src/jute_disease/data/utils.py split

# %%
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

# %%
test_file = LOCAL_ARTIFACTS / "test.txt"
test_file.write_text("Hacking into the mainframe, part 2.")

if (GDRIVE_ARTIFACTS / "test.txt").exists():
    print("Symlink worked.")
else:
    print("Symlink failed :<")

# %% [markdown]
# ## Part 1: Baseline Analysis
#
# ### 1A. Model Performance
#
# #### Impact of a Higher Image Resolution on Model Performance
#
# Let's begin by analyzing how training on a higher image resolution impacts our model's performance. Recall that our models were originally trained on 256x256 pixel images and that we trained 512x512 pixel counterparts for the pre-trained MobileNet V2 with dropout rates 0.0 and 0.1 for comparison.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from jute_disease.utils import get_logger
from jute_disease.utils.constants import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    DEFAULT_SEED,
    DPI,
    FIGURES_DL_DIR,
    LOGS_DIR,
    ML_SPLIT_DIR,
    NUM_WORKERS,
)

logger = get_logger("AnalysisNoteBook")

metrics_path = LOGS_DIR / "phase1_transfer_grid" / "aggregated_grid_metrics.csv"
res_512_00 = (
    LOGS_DIR / "resolution_exps" / "mobilenet_v2-512px-dr_0.0" / "summary_metrics.csv"
)
res_512_01 = (
    LOGS_DIR / "resolution_exps" / "mobilenet_v2-512px-dr_0.1" / "summary_metrics.csv"
)

if not metrics_path.exists():
    raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

df_phase1 = pd.read_csv(metrics_path)
extra_results = []
if res_512_00.exists():
    df_512_00 = pd.read_csv(res_512_00)
    extra_results.append(df_512_00)
if res_512_01.exists():
    df_512_01 = pd.read_csv(res_512_01)
    extra_results.append(df_512_01)

if extra_results:
    df = pd.concat([df_phase1] + extra_results, ignore_index=True)
else:
    df = df_phase1

plt.figure(figsize=(12, 6))
comparison_names = [
    "mobilenet_v2-l1_imagenet-dr_0.1",
    "mobilenet_v2-512px-dr_0.1",
    "mobilenet_v2-l1_imagenet-dr_0.0",
    "mobilenet_v2-512px-dr_0.0",
]
comp_df = df[df["Experiment"].isin(comparison_names)].copy()
comp_df["Resolution"] = comp_df["Experiment"].apply(
    lambda x: "512px" if "512px" in x else "256px"
)
comp_df["Dropout Rate"] = comp_df["Experiment"].apply(
    lambda x: 0.1 if "dr_0.1" in x else 0.0
)
comp_df = comp_df.sort_values("Dropout Rate")

ax_bar = sns.barplot(
    data=comp_df,
    x="Dropout Rate",
    y="test_acc",
    hue="Resolution",
    palette="viridis",
)
plt.ylim(0.8, 0.95)
plt.title("Impact of Image Resolution on Test Accuracy (MobileNetV2 with DR 0.1)")
plt.xlabel("Dropout Rate")
plt.ylabel("Test Accuracy")
plt.grid(axis="y", linestyle="--", alpha=0.7)

for p in ax_bar.patches:
    height = p.get_height()
    if height > 0:
        ax_bar.annotate(
            f"{height:.1%}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 4),
            textcoords="offset points",
        )

FIGURES_DL_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DL_DIR / "resolution_impact.png", bbox_inches="tight", dpi=DPI)
plt.show()

display(
    comp_df[["Experiment", "test_acc", "test_f1", "test_loss"]].reset_index(drop=True)
)

# %% [markdown]
# Some insights:
# - Training on 512x512 pixel images appears to lead to worse performance compared to training on 256x256 pixel images.
# - A dropout rate of 0.1 appears to lead to a higher test F1 compared to their 0.0 counterparts, though likely statistically insignificant given our sample. This may suggest that slightly increased regularization may improve our model's performance on unseen data.
# - Our current best test accuracy is **88.3%**, achieved by a MobileNet V2 pre-trained on ImageNet-1K with a dropout rate of **0.1**.
#
# Hence, our initial hypothesis of training on higher-resolution images is disproven, though not in a formal statistical manner.

# %% [markdown]
# #### Loss and Accuracy Curves
#
# Let's analyze the training and validation loss and accuracy curves of the baseline MobileNetV2. We'll compare DR 0.0, 0.1, and 0.2 to also see its regularization effect.

# %%
dr_rates = ["0.0", "0.1", "0.2"]
dr_colors = {"0.0": "tab:blue", "0.1": "tab:green", "0.2": "tab:orange"}

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

for dr in dr_rates:
    history_dir = (
        LOGS_DIR / "phase1_transfer_grid" / f"mobilenet_v2-l1_imagenet-dr_{dr}"
    )
    history_files = list(history_dir.glob("*-metrics.csv"))

    if not history_files:
        raise FileNotFoundError(f"No metrics found for DR {dr} at {history_dir}")

    dfs = [pd.read_csv(f) for f in history_files]
    hist = pd.concat(dfs, ignore_index=True)
    agg_dict = {}
    for col in hist.columns:
        if "loss" in col:
            agg_dict[col] = "mean"
        elif "acc" in col or "f1" in col:
            agg_dict[col] = "max"

    epoch_data = (
        hist.groupby("epoch").agg(agg_dict).dropna(subset=["train_loss", "val_loss"])
    )

    ax[0].plot(
        epoch_data.index,
        epoch_data["train_loss"],
        label=f"Train DR {dr}",
        color=dr_colors[dr],
        linestyle="--",
        alpha=0.6,
    )
    ax[0].plot(
        epoch_data.index,
        epoch_data["val_loss"],
        label=f"Val DR {dr}",
        color=dr_colors[dr],
    )

    ax[1].plot(
        epoch_data.index,
        epoch_data["train_acc"],
        label=f"Train DR {dr}",
        color=dr_colors[dr],
        linestyle="--",
        alpha=0.6,
    )
    ax[1].plot(
        epoch_data.index,
        epoch_data["val_acc"],
        label=f"Val DR {dr}",
        color=dr_colors[dr],
    )

ax[0].set_title(
    "Training and Validation Loss across Dropout Rates (MobileNet V2 with DR 0.1)"
)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()
ax[0].grid(True, alpha=0.3)

ax[1].set_title(
    "Training and Validation Accuracy across Dropout Rates\n(MobileNet V2 with DR 0.1)"
)
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    FIGURES_DL_DIR / "baseline_loss_accuracy_curves.png", bbox_inches="tight", dpi=DPI
)
plt.show()

# %% [markdown]
# Some insights:
# - Train loss appears to be consistently higher than validation loss.
#   - This is possibly explained by how our data is heavily augmented during training but not during validation, making it more difficult for the model to get correct predictions on the training set per epoch.
#   - Furthermore, due to dropout, some neurons are deactivated during training, making the task more difficult.
# - Train accuracy appears to be consistently lower than validation accuracy. This is possibly explained by the same reasons above. Fortunately, the gap between the two appears to decrease over time, indicating that the model was able to generalize better over time.
# - Train loss appears to be more erratic compared to validation loss. Moreover, higher dropout rates appear to lead to a slightly higher train loss and lower train accuracy during training (but nothing suggestive of test performance). This is possibly explained by the same reasons in the first point.
# - The loss and accuracy curves of different models appear to follow the same pattern.
#   - This is likely because we seeded the data splitting and augmentations, making them reproducible and thus, resulting in similar curves.
#
# It is worth inspecting how extending the training time will affect the training and validation metrics. We may have cut it too short by setting the early stopping patience low (originally 5). During fine-tuning, we will increase it to 20 to see whether the metrics will converge and improve.

# %% [markdown]
# ### 1B. Error Analysis
#
# #### Confusion Matrix
#
# Let's analyze where the model struggles by visualizing the confusion matrix logged by Weights and Biases. Since it is in a `.json` format, we'll first need to convert it into a pandas DataFrame.

# %%
import json

cmat_path = (
    LOGS_DIR
    / "phase1_transfer_grid"
    / "mobilenet_v2-l1_imagenet-dr_0.1"
    / "conf_mat.json"
)


def get_cm_metrics(cm_df):
    classes = cm_df.index
    total_samples = cm_df.values.sum()
    metrics = []
    for cls in classes:
        tp = cm_df.loc[cls, cls] if cls in cm_df.columns else 0
        fn = cm_df.loc[cls, :].sum() - tp
        fp = cm_df.loc[:, cls].sum() - tp if cls in cm_df.columns else 0
        tn = total_samples - (tp + fp + fn)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        acc = (tp + tn) / total_samples if total_samples > 0 else 0

        metrics.append(
            {
                "Class": cls,
                "Accuracy": acc,
                "Precision": p,
                "Recall": r,
                "F1-Score": f1,
                "Support": int(tp + fn),
            }
        )
    return pd.DataFrame(metrics).set_index("Class")


if not cmat_path.exists():
    raise FileNotFoundError(f"No confusion matrix found at {cmat_path}")

with open(cmat_path) as f:
    cmat_data = json.load(f)

df_cm = pd.DataFrame(cmat_data["data"], columns=cmat_data["columns"])
cm_pivot = df_cm.pivot(
    index="Actual", columns="Predicted", values="nPredictions"
).fillna(0)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_pivot, annot=True, fmt="g", cmap="Blues", cbar=False)
plt.title("Confusion Matrix (MobileNet V2 with DR 0.1)")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

FIGURES_DL_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(
    FIGURES_DL_DIR / "baseline_confusion_matrix.png",
    bbox_inches="tight",
    dpi=DPI,
)
plt.show()

print("Classification Metrics by Class")
df_metrics = get_cm_metrics(cm_pivot)
display(df_metrics.round(4))

# %% [markdown]
# Some insights:
# - The model performed pretty well classifying jute leaves that are _Healthy_ and those with _Dieback_, _General Damage_, and _Stem Rot_ diseases, achieving accuracies of at least 95%.
# - The model appears to struggle classifying _Mosaic_ and _Cercospora Leaf Spot_ diseases, both having two of the lowest accuracies and F1-Scores. We can also see from the confusion matrix that the model appears to struggle distinguishing between the two, as 9 _Mosaic_ images were misclassified as _Cercorposa Leaf Spot_ while 6 _Cercorposa Leaf Spot_ images were misclassified as _Mosaic_. This is significantly greater than every other disease pair.
# - The model misclassified 8 _Healthy_ jute leaves as _General Damage_, relatively higher compared to other classes it misclassified _Healthy_ leaves as (1 for Cercospora Leaf Spot, 2 for Mosaic). This is another possible source of confusion for the model.
#
# We can delve deeper into these phenomenon in the next visualizations.

# %% [markdown]
# #### Model Inference Setup
#
# The following error analysis, latent space analysis, and interpretability sections require the model's test set predictions and generated features, so let's first create them with our chosen MobileNet V2 configuration.
#
# > Funnily, this is the only time we'll have to manually write an ML loop as PyTorch Lightning normally handles most of this under the hood. However, we now need more control over what we feed into and get out from the model.
# %%
import time

import torch
import torch.nn.functional as F
import umap
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder

from jute_disease.data.datamodule import DataModule
from jute_disease.models.dl.backbone import TimmBackbone
from jute_disease.models.dl.classifier import Classifier

dm = DataModule(data_dir=str(ML_SPLIT_DIR))
dm.setup("test")
dm.setup("fit")

clean_train = ImageFolder(root=dm.train_dir, transform=dm.val_transform)
clean_val = ImageFolder(root=dm.val_dir, transform=dm.val_transform)
clean_test = ImageFolder(root=dm.test_dir, transform=dm.val_transform)

clean_train_loader = DataLoader(
    clean_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
val_loader = dm.val_dataloader()
test_loader = dm.test_dataloader()

pooled_dataset = ConcatDataset([clean_train, clean_val, clean_test])

champion_dir = ARTIFACTS_DIR / "checkpoints" / "mobilenet_v2-l1_imagenet-dr_0.1"
ckpt_paths = list(champion_dir.glob("*.ckpt"))

if not ckpt_paths:
    raise FileNotFoundError(f"Checkpoint not found for champion: {champion_dir}")

ckpt_path = ckpt_paths[0]
backbone = TimmBackbone(model_name="mobilenetv2_100")
model = Classifier.load_from_checkpoint(ckpt_path, feature_extractor=backbone)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_features = []
all_preds = []
all_targets = []
all_probs = []
all_splits = []
start_time = time.time()

loaders = [
    ("Train", clean_train_loader),
    ("Val", val_loader),
    ("Test", test_loader),
]

with torch.no_grad():
    for split_name, loader in loaders:
        for x, y in loader:
            x = x.to(device)
            feat = model.feature_extractor(x)
            logits = model.classifier(feat)
            probs = F.softmax(logits, dim=1)

            all_features.append(feat.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(logits.argmax(dim=1).cpu())
            all_targets.append(y)
            all_splits.extend([split_name] * x.size(0))

end_time = time.time()
total_imgs = len(pooled_dataset)
inf_time_ms = (end_time - start_time) / total_imgs * 1000
logger.info(f"Inference time per image: {inf_time_ms:.2f} ms")

features = torch.cat(all_features).numpy()
preds = torch.cat(all_preds).numpy()
targets = torch.cat(all_targets).numpy()
probs = torch.cat(all_probs).numpy()
splits = np.array(all_splits)

# %% [markdown]
# #### Top Confident Errors
#
# Let's visualize the model's top confident errors to analyze the images and classes it struggles with.

# %%
TOP_K = 20
is_wrong = preds != targets
wrong_indices = np.where(is_wrong)[0]

if len(wrong_indices) > 0:
    wrong_probs = [probs[i, preds[i]] for i in wrong_indices]
    n_display = min(TOP_K, len(wrong_indices))
    sorted_wrong = np.argsort(wrong_probs)[::-1][:n_display]
    top_wrong_idx = wrong_indices[sorted_wrong]

    ncols = 5
    nrows = int(np.ceil(n_display / ncols))
    plt.figure(figsize=(20, 5 * nrows))
    for i, idx in enumerate(top_wrong_idx):
        img, label = pooled_dataset[idx]
        img_disp = img.permute(1, 2, 0).numpy()
        img_disp = (
            img_disp * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        ).clip(0, 1)

        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img_disp)
        ax_sub = plt.gca()
        plt.title("")
        ax_sub.text(
            0.5,
            1.12,
            f"Pred: {dm.classes[preds[idx]]} ({probs[idx, preds[idx]]:.2f})",
            color="red",
            fontsize=10,
            ha="center",
            transform=ax_sub.transAxes,
        )
        ax_sub.text(
            0.5,
            1.02,
            f"Actual: {dm.classes[targets[idx]]}",
            color="black",
            fontsize=10,
            ha="center",
            transform=ax_sub.transAxes,
        )
        plt.axis("off")

    plt.suptitle(
        f"Top {n_display} Most Confident Incorrect Predictions\n"
        "(MobileNet V2 with DR 0.1)",
        fontsize=16,
    )
    plt.figtext(
        0.5,
        0.92,
        "(Note: The number in parenthesis is the prediction confidence)",
        ha="center",
        fontsize=12,
        color="gray",
    )
    plt.savefig(
        FIGURES_DL_DIR / f"top_{n_display}_errors.png", bbox_inches="tight", dpi=DPI
    )
    plt.show()
else:
    logger.info("No errors found in test set!")

# %% [markdown]
# Some insights:
# - 11 of the top 20 most confident incorrect predictions were _Mosaic_ misclassified as _Cercospora Leaf Spot_, while 2 were _Cercospora Leaf Spot_ misclassified as _Mosaic_. 
#   - This is likely caused by the visual similarity of spots present in both classes, which is possibly why the model was greatly confused between them.
#   - From a human standpoint, it is also pretty difficult to distinguish between the two diseases given low-resolution images of them.
#     - Admittedly, we are not experts in jute leaf diseases, so we can't tell for sure whether jute leaf spots can be attributed solely to either of the two. If anything, the _Mosaic_ disease causes the leaf to turn yellow or gold, which can be a more reliable indicator of it. It may be suggestive of the model focusing too much on the spots rather than the yellowing of the leaves.
#   - Moreover, it's possible for a jute leaf to have multiple diseases and for an image to contain multiple jute leaves. Thus, it may be better approaching the dataset as a multi-label or an object detection problem, rather than multi-class.
#     - If we make it multi-label, then we will need expert validation to ensure that an image truly exhibits multiple diseases.
#     - If we make it an object detection problem, then we will also need bounding boxes and more expert validation.
#     - Unfortunately, these approaches are beyond the scope of our project, so we will be sticking with the multi-class approach.
# - The 6th most confident incorrect prediction appears to focus on what appears to be a case of _Stem Rot_, but the image is somehow labeled as _Dieback_.
#   - This is caused by either incorrect labeling (where it should actually be Stem Rot) or center cropping from the image augmentations, which cropped away the leaf intended to be focused on.
# - There are 3 confusions between _General Damage_ and _Healthy_.
#   - Notably, these images have multiple visible leaves in the background. Ideally, we should have included blurring out irrelevant parts an image's background in its preprocessing so the model doesn't have to deal with noise. That may be what caused the model to misclassify some images.

# %% [markdown]
# ### 1C. Latent Space Analysis
#
# Some interesting stuff! Here, we will visualize our high-dimensional Jute leaf data in two dimensions with **t-SNE** and **UMAP**. Put simply, t-SNE is better at preserving local structures, while UMAP is generally faster and better at preserving global structure.

# %% [markdown]
# ### T-distributed Stochastic Neighbor Embedding (t-SNE)
#
# For t-SNE, we went with `perplexity=30` (recommended default).

# %%
tsne = TSNE(n_components=2, perplexity=30, random_state=DEFAULT_SEED)
feat_2d = tsne.fit_transform(features)

plt.figure(figsize=(14, 10))
colors = sns.color_palette("tab10", len(dm.classes))

for i, cls in enumerate(dm.classes):
    mask_train = (targets == i) & (splits == "Train")
    plt.scatter(
        feat_2d[mask_train, 0],
        feat_2d[mask_train, 1],
        color=colors[i],
        marker="x",
        s=25,
        alpha=0.4,
        label=None,
    )

    mask_eval = (targets == i) & (splits != "Train")
    plt.scatter(
        feat_2d[mask_eval, 0],
        feat_2d[mask_eval, 1],
        color=colors[i],
        marker="o",
        s=70,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
        label=cls,
    )

split_legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="gray",
        lw=0,
        markersize=8,
        label="Eval Set (Val/Test)",
    ),
    Line2D([0], [0], marker="x", color="gray", lw=0, markersize=8, label="Train Set"),
]
leg1 = plt.legend(handles=split_legend, loc="lower left", title="Splits")
plt.gca().add_artist(leg1)
plt.legend(loc="upper right", title="Classes", ncol=2)

plt.title("t-SNE Visualization of Jute Leaf Data (MobileNet V2 with DR 0.1)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig(FIGURES_DL_DIR / "tsne.png", bbox_inches="tight", dpi=DPI)
plt.show()

# %% [markdown]
# Some insights:
# - There are some well-clustered classes such as _General Damage_, _Stem Rot_, _Dieback_. _Healthy_ is also pretty well-clustered, but has three distinct clusters. These are the same classes that the model was able to distinguish significantly well with accuracies above 95%, thus explaining its performance.
# - There is a lone cluster of healthy images at the top right, isolated from other points. These may possibly be the images of healthy jute leaves taken with a clean background.
#   - As we saw from EDA, most of our jute leaf datasets had images collected from farms, but we also included a dataset which had images of healthy jute leaves taken on top of a table.
# - There's an area where _Cercospora Leaf Spot_ and _Mosaic_ latent features are mixed. This is possibly a consequence of the model focusing too much on leaf spots.
#   - Moreover, there are two clear _Mosaic_ clusters below it, which may possibly be _Mosaic_ examples without spots or other ambiguities. We hypothesize that the _Mosaic_ examples mixed with the _Cercospora Leaf Spot_ examples are those that have been confused as the latter due to the extracted features.

# %% [markdown]
# ### Uniform Manifold Approximation and Projection (UMAP)
#
# For UMAP, we went with `n_neighbors=15` and `min_dist=0.1` (recommended defaults by `umap-learn`). `n_neighbors` controls how UMAP balances local versus global structure (similar to perplexity), while `min_dist` determines how tightly UMAP is allowed to pack points together.

# %%
reducer = umap.UMAP(
    n_neighbors=15, min_dist=0.1, n_components=2, random_state=DEFAULT_SEED
)
feat_umap = reducer.fit_transform(features)

plt.figure(figsize=(14, 10))

for i, cls in enumerate(dm.classes):
    mask_train = (targets == i) & (splits == "Train")
    plt.scatter(
        feat_umap[mask_train, 0],
        feat_umap[mask_train, 1],
        color=colors[i],
        marker="x",
        s=25,
        alpha=0.4,
        label=None,
    )

    mask_eval = (targets == i) & (splits != "Train")
    plt.scatter(
        feat_umap[mask_eval, 0],
        feat_umap[mask_eval, 1],
        color=colors[i],
        marker="o",
        s=70,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
        label=cls,
    )

leg1 = plt.legend(handles=split_legend, loc="lower left", title="Splits")
plt.gca().add_artist(leg1)
plt.legend(loc="upper right", title="Classes", ncol=2)

plt.title("UMAP Visualization of Jute Leaf Data (MobileNet V2 with DR 0.1)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(FIGURES_DL_DIR / "umap.png", bbox_inches="tight", dpi=DPI)
plt.show()

# %% [markdown]
# Some insights:
# - There are three clearly isolated latent space 'islands.'
#   - The top-left consists of all classes except _Cercospora Leaf Spot_.
#   - The top-right consists solely of _Healthy_ samples.
#   - The bottom-left consists of _Healthy_, _Cercospora Leaf Spot_, and _Mosaic_ samples.
# - Similar to t-SNE, we can see a somewhat distinct clustering for _Dieback_, _General Damage_, _Stem Rot_, and _Healthy_.
# - Yet again, we see 3 different clusters for _Healthy_, with an isolated one at the top-right. We also see the mixture of _Cercospora Leaf Spot_ and _Mosaic_ latent space features at the bottom-left island.

# %% [markdown]
# ### 1D. Interpretability
#
# We now use Grad-CAM to visualize where on the jute leaf images the model focuses on when making predictions.

# %%
from captum.attr import LayerGradCam
from scipy.ndimage import zoom

target_layer = model.feature_extractor.backbone.conv_head
lgc = LayerGradCam(model, target_layer)

num_samples = 10
num_classes = len(dm.classes)
plt.figure(figsize=(20, 4 * num_classes))
np.random.seed(DEFAULT_SEED)

plot_idx = 1
for class_idx in range(num_classes):
    class_name = dm.classes[class_idx]
    all_class_indices = np.where(targets == class_idx)[0]

    n = min(len(all_class_indices), num_samples)
    selected_indices = np.random.choice(all_class_indices, n, replace=False)

    for idx in selected_indices:
        img, label = pooled_dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)

        attribution = lgc.attribute(input_tensor, target=label)
        heatmap = attribution.squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        img_disp = img.permute(1, 2, 0).numpy()
        img_disp = (
            img_disp * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        ).clip(0, 1)

        h, w = img_disp.shape[:2]
        heatmap_upsampled = zoom(heatmap, (h / heatmap.shape[0], w / heatmap.shape[1]))

        plt.subplot(num_classes, num_samples, plot_idx)
        plt.imshow(img_disp)
        plt.imshow(heatmap_upsampled, cmap="jet", alpha=0.4)
        if (plot_idx - 1) % num_samples == 0:
            plt.ylabel(class_name, fontsize=14, fontweight="bold")

        plt.title(f"Conf: {probs[idx, label]:.2f}")
        plt.xticks([])
        plt.yticks([])
        plot_idx += 1

    plot_idx += num_samples - n

plt.suptitle(
    "Grad-CAM Heatmaps on Sample Jute Leaf Disease Images\n(MobileNet V2 with DR 0.1)",
    fontsize=20,
    y=1.02,
)
plt.tight_layout()
plt.savefig(FIGURES_DL_DIR / "grad_cam.png", bbox_inches="tight", dpi=DPI)
plt.show()

# %% [markdown]
# Some insights:
# - The model appears to have lower confidence for _Cercospora Leaf Spot_ and _Mosaic_ images, likely due to its possible confusion between the two.
# - For _Cercospora Leaf Spot_ and _Mosaic_, the model is generally able to focus on leaf spots.
#   - However, in some images taken from the wild, the model focused on the background instead of the jute leaf disease in question. The 8th and 10th _Cercospora Leaf Spot_ images illustrate this.
# - The model is generally able to focus on curled leaves for _Dieback_, which is intended.
# - The model is generally able to focus on holes and tears for _General Damage_, which is intended.
#   - However, in some images taken from the wild, it appears to focus on features beyond the jute leaf in question. The 5th and 7th images illustrate this.
#   - This supports our general suggestion of preprocessing the image background in some way.
# - The model appears to classify _Healthy_ images pretty well, but some for the wrong reasons.
#   - Concerningly, for the 5 images taken on top of a table, the model appeared to focus on the background. This is indicative of the model learning the background instead of the features of a _Healthy_ jute leaf.
#   - This further supports our general suggestion of preprocessing the image background in some way.
# - Finally, the model is generally able to focus on relevant features for _Mosaic_ and _Stem Rot_.
#   - Three Mosaic images show the model focusing on the background instead of the jute leaf in question.

# %% [markdown]
# ## Part 2: Fine-Tuning
#
# Our analysis above led to the hypothesis that the problem is primarily in the preprocessing of the data rather than the architecture (though we can still modify the architecture and expect improvements). We want to determine whether our model is indeed bottle-necked by the data preprocessing or simply lacked training, as our early stopping was originally set to 5.
#
# Thus, we will execute a second grid search with the ff. configurations:
# - **Learning Rate**: `0.01`, `0.005`, `0.001`, `0.0005`, `0.0001`
# - **Extended Patience**: `early_stopping_patience` raised to 20.
#
# Our results can be found [here](https://wandb.ai/grade-descent/jute-disease-detection/groups/MobileNet%20V2%20Finetune%20Grid/workspace).

# %%
# !uv run python scripts/run_grid_search.py \
#     configs/grid/mobilenet_v2_finetune_grid.yaml \
#     --base-config configs/baselines/mobilenet_v2.yaml

# %% [markdown]
# ### 2A. Model Performance

# %% [markdown]
# #### Test Accuracy across Learning Rates
#
# Let's visualize test accuracy across all the learning rates we tested to see which learning rate may be better suited for our task.

# %%
ft_metrics_path = LOGS_DIR / "phase2_finetune_grid" / "aggregated_grid_metrics.csv"

if not ft_metrics_path.exists():
    raise FileNotFoundError(f"Metrics file not found at {ft_metrics_path}")

df_ft = pd.read_csv(ft_metrics_path)


def extract_lr(exp_name):
    import re

    match = re.search(r"lr_([0-9.]+)", exp_name)
    return float(match.group(1)) if match else None


df_ft["Learning Rate"] = df_ft["Experiment"].apply(extract_lr)

df_ft = df_ft.sort_values("Learning Rate", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df_ft,
    x="Learning Rate",
    y="test_acc",
    hue="Learning Rate",
    palette="Oranges_r",
    legend=False,
)
plt.ylim(0.85, 0.95)
plt.title("Test Accuracy across Finetuning Learning Rates (MobileNet V2 with DR 0.1)")
plt.xlabel("Learning Rate")
plt.ylabel("Test Accuracy")
plt.grid(axis="y", linestyle="--", alpha=0.7)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f"{height:.1%}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 4),
            textcoords="offset points",
        )

FIGURES_DL_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(
    FIGURES_DL_DIR / "finetuned_lr_impact.png",
    bbox_inches="tight",
    dpi=DPI,
)
plt.show()

disp_cols = ["Learning Rate", "epoch", "test_acc", "test_f1", "test_loss"]
display(df_ft[disp_cols].reset_index(drop=True))

# %% [markdown]
# Some insights:
# - A learning rate of `0.01` appears to be the best for our task, leading to the highest test accuracy of around **91.4%**. This is a slight jump from the previous best accuracy of **88.3%**.
# - Decreasing the learning rate appears to hurt model performance for our task. It also took the longest, taking up to 49 epochs.

# %% [markdown]
# #### Loss and Accuracy Curves
#
# Let's inspect the training curve of our best fine-tuned configuration (`LR=0.01`).

# %%
finetuned_history_dir = (
    LOGS_DIR / "phase2_finetune_grid" / "mobilenet_v2-l1_imagenet-lr_0.01-wd_0.05"
)
ft_history_files = list(finetuned_history_dir.glob("*-metrics.csv"))

if not ft_history_files:
    raise FileNotFoundError(f"History files not found in {finetuned_history_dir}")

dfs = [pd.read_csv(f) for f in ft_history_files]
history = pd.concat(dfs, ignore_index=True)
agg_dict = {}
for col in history.columns:
    if "loss" in col:
        agg_dict[col] = "mean"
    elif "acc" in col or "f1" in col:
        agg_dict[col] = "max"

epoch_data = (
    history.groupby("epoch")
    .agg(agg_dict)
    .dropna(
        subset=[
            "train_loss",
            "val_loss" if "val_loss" in history.columns else "train_loss",
        ]
    )
)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

loss_cols = [c for c in ["train_loss", "val_loss"] if c in epoch_data.columns]
epoch_data[loss_cols].plot(ax=ax[0])
ax[0].set_title("Training and Validation Loss (MobileNet V2 with DR 0.1, LR 0.01)")
ax[0].set_xlabel("Epoch")
ax[0].grid(True, alpha=0.3)

avail_acc = [c for c in ["train_acc", "val_acc"] if c in epoch_data.columns]
epoch_data[avail_acc].plot(ax=ax[1])
ax[1].set_title("Training and Validation Accuracy (MobileNet V2 with DR 0.1, LR 0.01)")
ax[1].set_xlabel("Epoch")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    FIGURES_DL_DIR / "finetuned_training_history.png", bbox_inches="tight", dpi=DPI
)
plt.show()

# %% [markdown]
# Our train and validation loss and accuracy across epochs are now much more erratic compared to the baselines, likely due to the higher LR of 0.01 compared to the initial LR of 0.001.

# %% [markdown]
# ### 2B. Error Analysis
#
# #### Confusion Matrix Comparison
#
# Let's compare our finely tuned model's confusion matrix with that of the baseline.

# %%
import json

ft_cmat_path = (
    LOGS_DIR
    / "phase2_finetune_grid"
    / "mobilenet_v2-l1_imagenet-lr_0.01-wd_0.05"
    / "conf_mat.json"
)

if not ft_cmat_path.exists():
    raise FileNotFoundError(f"Finetuned confusion matrix not found at {ft_cmat_path}")
if not cmat_path.exists():
    raise FileNotFoundError(f"Baseline confusion matrix not found at {cmat_path}")

with open(ft_cmat_path) as f:
    ft_cmat_data = json.load(f)

df_ft_cm = pd.DataFrame(ft_cmat_data["data"], columns=ft_cmat_data["columns"])
ft_cm_pivot = df_ft_cm.pivot(
    index="Actual", columns="Predicted", values="nPredictions"
).fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.heatmap(cm_pivot, annot=True, fmt="g", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_title("Confusion Matrix (MobileNet V2 with DR 0.1)")
axes[0].set_ylabel("Actual Class")
axes[0].set_xlabel("Predicted Class")
axes[0].tick_params(axis="x", rotation=45)

sns.heatmap(ft_cm_pivot, annot=True, fmt="g", cmap="Oranges", cbar=False, ax=axes[1])
axes[1].set_title("Confusion Matrix (MobileNet V2 with DR 0.1, LR 0.01)")
axes[1].set_ylabel("Actual Class")
axes[1].set_xlabel("Predicted Class")
axes[1].tick_params(axis="x", rotation=45)

FIGURES_DL_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(
    FIGURES_DL_DIR / "part2_confusion_matrix_comparison.png",
    bbox_inches="tight",
    dpi=DPI,
)
plt.show()

print("Part 2: Finetuned Classification Metrics by Class")
df_ft_metrics = get_cm_metrics(ft_cm_pivot)
display(df_ft_metrics.round(4))

# %% [markdown]
# Though the model's overall accuracy and per-class accuracy somewhat increased, the confusion matrix comparison shows that the model with the fine-tuned LR of 0.01 wasn't able to address its core issue of failing to distinguish between _Cercospora Leaf Spot_ and _Mosaic_, thus holding it back from achieving a much higher accuracy.

# %% [markdown]
# #### Finetuned Model Inference
# We now load the best checkpoint from our finetuning run and re-generate predictions for Error Analysis and Grad-CAM visualization.

# %%
finetuned_dir = (
    ARTIFACTS_DIR / "checkpoints" / "mobilenet_v2-l1_imagenet-lr_0.01-wd_0.05"
)
ft_ckpt_paths = list(finetuned_dir.glob("*.ckpt"))

if not ft_ckpt_paths:
    raise FileNotFoundError(
        f"Checkpoint not found for finetuned champion: {finetuned_dir}"
    )

ft_ckpt_path = ft_ckpt_paths[0]
backbone = TimmBackbone(model_name="mobilenetv2_100")
ft_model = Classifier.load_from_checkpoint(ft_ckpt_path, feature_extractor=backbone)
ft_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ft_model.to(device)

all_features = []
all_preds = []
all_targets = []
all_probs = []
all_splits = []
start_time = time.time()

loaders = [
    ("Train", clean_train_loader),
    ("Val", val_loader),
    ("Test", test_loader),
]

with torch.no_grad():
    for split_name, loader in loaders:
        for x, y in loader:
            x = x.to(device)
            feat = ft_model.feature_extractor(x)
            logits = ft_model.classifier(feat)
            probs = F.softmax(logits, dim=1)

            all_features.append(feat.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(logits.argmax(dim=1).cpu())
            all_targets.append(y)
            all_splits.extend([split_name] * x.size(0))

end_time = time.time()
total_imgs = len(pooled_dataset)
inf_time_ms = (end_time - start_time) / total_imgs * 1000
logger.info(f"[Finetuned] Inference time per image: {inf_time_ms:.2f} ms")

ft_features = torch.cat(all_features).numpy()
ft_preds = torch.cat(all_preds).numpy()
ft_targets = torch.cat(all_targets).numpy()
ft_probs = torch.cat(all_probs).numpy()
ft_splits = np.array(all_splits)

# %% [markdown]
# #### Finetuned Top Confident Errors

# %%
TOP_K = 20
is_wrong = ft_preds != ft_targets
wrong_indices = np.where(is_wrong)[0]

if len(wrong_indices) > 0:
    wrong_probs = [ft_probs[i, ft_preds[i]] for i in wrong_indices]
    n_display = min(TOP_K, len(wrong_indices))
    sorted_wrong = np.argsort(wrong_probs)[::-1][:n_display]
    top_wrong_idx = wrong_indices[sorted_wrong]

    ncols = 5
    nrows = int(np.ceil(n_display / ncols))
    plt.figure(figsize=(20, 5 * nrows))
    for i, idx in enumerate(top_wrong_idx):
        img, label = pooled_dataset[idx]
        img_disp = img.permute(1, 2, 0).numpy()
        img_disp = (
            img_disp * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        ).clip(0, 1)

        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img_disp)
        ax_sub = plt.gca()
        plt.title("")
        ax_sub.text(
            0.5,
            1.12,
            f"Pred: {dm.classes[ft_preds[idx]]}\n({ft_probs[idx, ft_preds[idx]]:.2f})",
            color="red",
            fontsize=10,
            ha="center",
            transform=ax_sub.transAxes,
        )
        ax_sub.text(
            0.5,
            1.02,
            f"Actual: {dm.classes[ft_targets[idx]]}",
            color="black",
            fontsize=10,
            ha="center",
            transform=ax_sub.transAxes,
        )
        plt.axis("off")

    plt.suptitle(
        f"Top {n_display} Most Confident Incorrect Predictions\n"
        "(MobileNet V2 with DR 0.1, LR 0.01)",
        fontsize=16,
    )
    plt.figtext(
        0.5,
        0.92,
        "(Note: The number in parenthesis is the prediction confidence)",
        ha="center",
        fontsize=12,
        color="gray",
    )
    plt.savefig(
        FIGURES_DL_DIR / f"finetuned_top_{n_display}_errors.png",
        bbox_inches="tight",
        dpi=DPI,
    )
    plt.show()
else:
    logger.info("[Finetuned] No errors found in test set!")

# %% [markdown]
# Unfortunately, the fine-tuned model still exhibits the same problem of failing to distinguish between _Cercospora Leaf Spot_ and _Mosaic_, but this time with higher confidences in the incorrect label.

# %% [markdown]
# ### 2C. Interpretability
#
# #### Finetuned Model Interpretability (Grad-CAM)

# %%
target_layer = ft_model.feature_extractor.backbone.conv_head
lgc = LayerGradCam(ft_model, target_layer)

num_samples = 10
num_classes = len(dm.classes)
plt.figure(figsize=(20, 4 * num_classes))
np.random.seed(DEFAULT_SEED)

plot_idx = 1
for class_idx in range(num_classes):
    class_name = dm.classes[class_idx]
    all_class_indices = np.where(ft_targets == class_idx)[0]

    n = min(len(all_class_indices), num_samples)
    selected_indices = np.random.choice(all_class_indices, n, replace=False)

    for idx in selected_indices:
        img, label = pooled_dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)

        attribution = lgc.attribute(input_tensor, target=label)
        heatmap = attribution.squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        img_disp = img.permute(1, 2, 0).numpy()
        img_disp = (
            img_disp * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        ).clip(0, 1)

        h, w = img_disp.shape[:2]
        heatmap_upsampled = zoom(heatmap, (h / heatmap.shape[0], w / heatmap.shape[1]))

        plt.subplot(num_classes, num_samples, plot_idx)
        plt.imshow(img_disp)
        plt.imshow(heatmap_upsampled, cmap="jet", alpha=0.4)
        if (plot_idx - 1) % num_samples == 0:
            plt.ylabel(class_name, fontsize=14, fontweight="bold")

        plt.title(f"Conf: {ft_probs[idx, label]:.2f}")
        plt.xticks([])
        plt.yticks([])
        plot_idx += 1

    plot_idx += num_samples - n

plt.suptitle(
    "Grad-CAM Heatmaps on Sample Jute Leaf Disease Images\n"
    "(MobileNet V2 with DR 0.1, LR 0.01)",
    fontsize=20,
    y=1.02,
)
plt.tight_layout()
plt.savefig(FIGURES_DL_DIR / "finetuned_grad_cam.png", bbox_inches="tight", dpi=DPI)
plt.show()

# %% [markdown]
# Nothing much changed and the model wasn't able to address previous problems.

# %% [markdown]
# ## Conclusion
#
# Through our analyses, we have shown that while fine-tuning the learning rate to $0.01$ can slightly improve accuracy, the model reaches a performance plateau due to major problems such as focusing on irrelevant details (e.g., background, other visual features) and failing to distinguish between _Cercospora Leaf Spot_ and _Mosaic_ diseases due to visual similarities. Future work can consider improving the data preprocessing pipeline or exploring multi-label classification or object detection to consider cases where a jute leaf may exhibit multiple diseases or multiple jute leaves are present in a single image.

# %% [markdown]
# ## References
#
# [1] Coenen, A., & Pearce, A. (2019, December 5). _Understanding UMAP_. <https://pair-code.github.io/understanding-umap/>
#
# [2] Wattenberg, M., Viégas, F., & Johnson, I. (2016). How to use t-SNE effectively. *Distill*, 1(10). <https://doi.org/10.23915/distill.00002>
