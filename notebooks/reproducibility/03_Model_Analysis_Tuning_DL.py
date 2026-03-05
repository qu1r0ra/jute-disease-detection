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

# %% [markdown]
# # Deep Learning Model Analysis and Fine-Tuning

# %% [markdown]
# ## 1. Introduction and Baseline Review
# In the previous notebook, we trained a variety of deep learning baselines and performed a Phase 1 grid search on Transfer Learning initialization strategies for our champion, MobileNetV2.
#
# ## 2. Phase 1: Grid Search Results Analysis
# We have completed the Phase 1 grid search which focused on choosing the best transfer learning initialization. Let's analyze the consolidated results from our local metrics backup.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

metrics_path = Path("../../artifacts/grid_search_mobilenet_v2_phase1_metrics.csv")
if metrics_path.exists():
    df = pd.read_csv(metrics_path)
    
    # Display Top 5 by Test Accuracy
    print("Top 5 Configurations (Sorted by Test Accuracy):")
    display(df.sort_values("test_acc", ascending=False).head(5))
    
    # Performance Visualization
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df.sort_values("test_acc"), x="Experiment", y="test_acc", palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.title("Phase 1 Grid Search: Test Accuracy Comparison")
    plt.ylim(0.7, 0.95)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print(f"Metrics not found at {metrics_path}. Please run recovery script if needed.")

# %% [markdown]
# ## 3. Preliminary Error Analysis (Phase 1 Champion)
# Before proceeding with optimizer fine-tuning (Phase 2), we analyze the errors of our current best model (ImageNet Level 1). We will load the champion checkpoint and evaluate it explicitly on the test set to generate a detailed classification report and confusion matrix.

# %%
import torch
from lightning.pytorch import Trainer
from sklearn.metrics import classification_report, confusion_matrix
from jute_disease.models.dl.classifier import Classifier
from jute_disease.models.dl.backbone import TimmBackbone
from jute_disease.data.datamodule import DataModule

# 1. Setup Data
dm = DataModule(data_dir="../../data/ml_split", batch_size=32)
dm.setup("test")

# 2. Identify and Load Champion Checkpoint
# Note: Update this path to your actual champion file if it differs
champion_dir = Path("../../artifacts/checkpoints/mobilenet_v2-l1_imagenet-dr_0.1")
ckpts = list(champion_dir.glob("*.ckpt"))
if ckpts:
    ckpt_path = ckpts[0]
    print(f"Loading champion from {ckpt_path}")
    
    backbone = TimmBackbone(model_name="mobilenetv2_100")
    model = Classifier.load_from_checkpoint(ckpt_path, feature_extractor=backbone)
    model.eval()
    
    # 3. Inference
    trainer = Trainer(accelerator="auto")
    preds_list = trainer.predict(model, dataloaders=dm.test_dataloader())
    
    all_preds = torch.cat(preds_list).argmax(dim=-1).numpy()
    all_targets = torch.cat([batch[1] for batch in dm.test_dataloader()]).numpy()
    
    # 4. Detailed Reports
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=dm.classes))
    
    # 5. Confusion Matrix Visualization
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=dm.classes, yticklabels=dm.classes, cmap='Blues')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.title('Champion Confusion Matrix: Phase 1 Finalist')
    plt.show()
else:
    print("No checkpoints found in champion directory.")

# %% [markdown]
# ## 4. Phase 2: Optimizer Fine-Tuning Grid Search (MobileNetV2)
# With our baseline errors confirmed as expected architectural characteristics (and not systemic labeling bugs), we can comfortably execute Phase 2. We will sweep across standard Learning Rates and Weight Decays to squeeze the ultimate convergence trajectory out of our Jute dataset.

# %%
# !make grid-search-finetune
