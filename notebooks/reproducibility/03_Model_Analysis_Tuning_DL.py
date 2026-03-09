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
# Continuing from our previous notebook, we will now conduct a detailed evaluation of our chosen MobileNet V2 champion model.
#
# Specifically, we will:
# - Compare the performance of our 256x256 pixel and 512x512 pixel MobileNet V2 models.
# - Visualize the training history to inspect for overfitting or underfitting.
# - Conduct Error Analysis by exploring latent space embeddings (t-SNE and UMAP) and inspecting the top confident errors.
# - Apply Model Interpretability techniques using Grad-CAM to see where the model focuses its attention.
#
# **Notes:**
#
# - Like the previous notebook, this is expected to be executed in **Google Colab**.
# - In this notebook, we specifically chose not to abstract the visualization logic (t-SNE, UMAP, Grad-CAM routines) into separate visualization scripts, prioritizing our ability to perform quick revisions and iterate interactively.

# %% [markdown]
# ## Environment Setup
#
# Let's run the environment setup again. Refer to the previous notebook for detailed instructions and remarks.

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
# ## Quantitative Performance
#
# Let's start by summarizing the performance metrics and comparing the effects of image resolution on our Level 1 MobileNet V2 model.

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
res_512_01 = (
    LOGS_DIR / "resolution_exps" / "mobilenet_v2-512px-dr_0.1" / "summary_metrics.csv"
)
res_512_00 = (
    LOGS_DIR / "resolution_exps" / "mobilenet_v2-512px-dr_0.0" / "summary_metrics.csv"
)

if metrics_path.exists():
    df_phase1 = pd.read_csv(metrics_path)

    extra_results = []
    if res_512_01.exists():
        df_512_01 = pd.read_csv(res_512_01)
        df_512_01["Experiment"] = "mobilenet_v2-512px-dr_0.1"
        extra_results.append(df_512_01)
    if res_512_00.exists():
        df_512_00 = pd.read_csv(res_512_00)
        df_512_00["Experiment"] = "mobilenet_v2-512px-dr_0.0"
        extra_results.append(df_512_00)

    if extra_results:
        df = pd.concat([df_phase1] + extra_results, ignore_index=True)
    else:
        df = df_phase1

    champion_name = "mobilenet_v2-l1_imagenet-dr_0.1"
    champion_row = df[df["Experiment"] == champion_name]

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
        lambda x: "0.1" if "dr_0.1" in x else "0.0"
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
    plt.title("Resolution Impact on Test Accuracy (MobileNetV2)")
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

    display(comp_df[["Experiment", "test_acc", "test_f1", "test_loss"]])

# %% [markdown]
# > continue here
#
# Some insights:
# - **The "Resolution Ceiling"**: The results show a consistent ~2.5% decrease in test accuracy when moving to 512x512.
# - **Feature Dilation**: MobileNetV2 was pre-trained on 224x224 ImageNet. At 512x512, the spatial features become "dilated" relative to the convolutional kernels, causing the model to struggle with matching its pre-trained weights to the "zoomed out" symptoms.
# - **Limited Parameters**: With only 2.2M parameters, MobileNetV2 lacks the capacity to effectively "process" the 4x larger input space without overfitting to the extra structural noise.

# %% [markdown]
# Now, let's look at the training dynamics of our champion model (256x256, 0.1 Dropout).

# %%
# 2. Load Training History for Curves
history_dir = LOGS_DIR / "phase1_transfer_grid" / "mobilenet_v2-l1_imagenet-dr_0.1"
history_files = list(history_dir.glob("*-metrics.csv"))

if history_files:
    dfs = [pd.read_csv(f) for f in history_files]
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
    ax[0].set_title("Training and Validation Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].grid(True, alpha=0.3)

    avail_acc = [c for c in ["train_acc", "val_acc"] if c in epoch_data.columns]
    epoch_data[avail_acc].plot(ax=ax[1])
    ax[1].set_title("Training and Validation Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DL_DIR / "training_history.png", bbox_inches="tight", dpi=DPI)
    plt.show()

# %% [markdown]
# > continue here
#
# Some insights:
# - ...

# %% [markdown]
# ## Error Analysis
#
# Let's perform a t-SNE and UMAP embedding analysis to visualize model separability, and then inspect the top confident errors.

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

dm = DataModule(data_dir=str(ML_SPLIT_DIR), batch_size=BATCH_SIZE)
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

if ckpt_paths:
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
# ### Latent Space Embeddings (t-SNE and UMAP)

# %%
if ckpt_paths:
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
        Line2D(
            [0], [0], marker="x", color="gray", lw=0, markersize=8, label="Train Set"
        ),
    ]
    leg1 = plt.legend(handles=split_legend, loc="lower left", title="Splits")
    plt.gca().add_artist(leg1)
    plt.legend(loc="upper right", title="Classes", ncol=2)

    plt.title("t-SNE Visualization of Jute Leaf Data")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(FIGURES_DL_DIR / "tsne.png", bbox_inches="tight", dpi=DPI)
    plt.show()

# %%
if ckpt_paths:
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

    plt.title("UMAP Visualization of Jute Leaf Data")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(FIGURES_DL_DIR / "umap.png", bbox_inches="tight", dpi=DPI)
    plt.show()

# %% [markdown]
# > continue here
#
# Some insights:
# - ...

# %% [markdown]
# ### Top Confident Errors

# %%
if ckpt_paths:
    is_wrong = preds != targets
    wrong_indices = np.where(is_wrong)[0]

    if len(wrong_indices) > 0:
        wrong_probs = [probs[i, preds[i]] for i in wrong_indices]
        sorted_wrong = np.argsort(wrong_probs)[::-1][:10]
        top_wrong_idx = wrong_indices[sorted_wrong]

        plt.figure(figsize=(20, 10))
        for i, idx in enumerate(top_wrong_idx):
            img, label = pooled_dataset[idx]
            img_disp = img.permute(1, 2, 0).numpy()
            img_disp = (
                img_disp * np.array([0.229, 0.224, 0.225])
                + np.array([0.485, 0.456, 0.406])
            ).clip(0, 1)

            plt.subplot(2, 5, i + 1)
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

        plt.suptitle("Top 10 Most Confident Incorrect Predictions", fontsize=16)
        plt.figtext(
            0.5,
            0.92,
            "(Note: The number in parenthesis is the prediction confidence)",
            ha="center",
            fontsize=12,
            color="gray",
        )
        plt.savefig(FIGURES_DL_DIR / "top_10_errors.png", bbox_inches="tight", dpi=DPI)
        plt.show()
    else:
        logger.info("No errors found in test set!")

# %% [markdown]
# > continue here
#
# Some insights:
# - **The "Data-Level Ceiling"**: Label Ambiguity. Many Jute leaves in our dataset exhibit symptoms of **multiple classes simultaneously** (e.g., both Mosaic and Cercospora). In our current **Single-Label Multiclass** setup, the model is forced to choose one, and is penalized for recognizing features of the other.
# - ...

# %% [markdown]
# ## Model Interpretability
#
# We use Grad-CAM to visualize where the model focuses its attention when making predictions.

# %%
from captum.attr import LayerGradCam
from scipy.ndimage import zoom

if ckpt_paths:
    target_layer = model.feature_extractor.backbone.conv_head
    lgc = LayerGradCam(model, target_layer)

    num_samples = 5
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
                img_disp * np.array([0.229, 0.224, 0.225])
                + np.array([0.485, 0.456, 0.406])
            ).clip(0, 1)

            h, w = img_disp.shape[:2]
            heatmap_upsampled = zoom(
                heatmap, (h / heatmap.shape[0], w / heatmap.shape[1])
            )

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
        "Grad-CAM Heatmaps on Sample Jute Leaf Disease Images", fontsize=20, y=1.02
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DL_DIR / "grad_cam.png", bbox_inches="tight", dpi=DPI)
    plt.show()

# %% [markdown]
# > continue here
#
# Some insights:
# - ...

# %% [markdown]
# ## Phase 2: Optimizer Fine-Tuning (Empirical Verification)
#
# Our analysis above (specifically regarding the multi-label ambiguity in our dataset) led us to form a strong hypothesis: the performance ceiling we are experiencing (~90% test accuracy) is a "Data-Level Ceiling" caused by overlapping symptoms, not an architectural capacity issue.
#
# However, to be scientifically rigorous, we must verify this. Is the model genuinely bottlenecked by the data, or was it simply under-trained due to a lack of epochs or killed prematurely by strict early stopping configurations?
#
# To answer this, we execute a final "Phase 2" grid search dedicated exclusively to fine-tuning the **Learning Rate** with significantly extended training bounds:
# - **Iterating LRs**: `0.01`, `0.005`, `0.001`, `0.0005`, `0.0001`
# - **Extended Patience**: `early_stopping_patience` raised to 20.
#
# If the model still caps at similar performance levels despite exhaustive optimizer iterations and extended time arrays, our data-ceiling hypothesis stands.

# %%
# !uv run python scripts/run_grid_search.py \
#     configs/grid/mobilenet_v2_finetune_grid.yaml \
#     --base-config configs/baselines/mobilenet_v2.yaml

# %% [markdown]
# ## Conclusion
#
# > continue here
#
# - Final words...
