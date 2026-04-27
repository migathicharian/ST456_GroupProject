"""
Evaluation, visualisation, and interpretability helpers for the PCam
histopathology SSL project.

Includes:
    - evaluate_model: AUC, F1, Accuracy on a DataLoader.
    - summarise_results, format_mean_std: turn raw runs into mean ± std tables.
    - plot_roc_pr_curves, plot_confusion, plot_tsne: figures for the report.
    - extract_embeddings: pull encoder features for visualisation.
    - GradCAM: class-activation maps for ResNet-based classifiers.

Usage from notebook:
    from src.evaluation import (
        evaluate_model, summarise_results, format_mean_std,
        plot_roc_pr_curves, plot_confusion, extract_embeddings,
        plot_tsne, GradCAM,
    )
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


# Device 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Metric computation

def evaluate_model(model: nn.Module, loader: DataLoader) -> Dict[str, np.ndarray]:
    """Run inference over loader and compute loss and AUC, F1, Accuracy.

    Returns a dict that also contains the raw y_true, y_prob, y_pred arrays
    so downstream plots (ROC, confusion matrix) can re-use them without a
    second forward pass.
    """
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs),
        "y_true": all_labels,
        "y_prob": all_probs,
        "y_pred": all_preds,
    }


# Result tables (mean +/- std)
 
def summarise_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw per-seed results into mean and std per configuration."""
    return (
        df.groupby(["method", "strategy", "label_fraction"])[
            ["test_accuracy", "test_f1", "test_auc"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )


def format_mean_std(df: pd.DataFrame, metric: str) -> pd.Series:
    """Return a Series of 'mean ± std' strings for a given metric column."""
    mean_col = (metric, "mean")
    std_col = (metric, "std")
    return (
        df[mean_col].map(lambda x: f"{x:.4f}")
        + " ± "
        + df[std_col].fillna(0).map(lambda x: f"{x:.4f}")
    )


def build_report_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Turn a summarise_results output into a report-ready table."""
    table = summary[["method", "strategy", "label_fraction"]].copy()
    table["AUC"] = format_mean_std(summary, "test_auc")
    table["F1"] = format_mean_std(summary, "test_f1")
    table["Accuracy"] = format_mean_std(summary, "test_accuracy")
    return table


# Plots: ROC, PR, Confusion matrix

def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title_prefix: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title(f"{title_prefix} ROC")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    axes[1].plot(recall, precision)
    axes[1].set_title(f"{title_prefix} Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion matrix",
    save_path: Optional[str] = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# Embeddings and t-SNE

def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pull encoder-level features from a classifier on the given loader.

    method argument determines how to call the encoder:
        - 'supervised_from_scratch' or 'simclr': model.encoder(x) returns 512-d
        - 'mae' or 'mae_improved': model.encoder.forward_encoder(x) returns 192-d
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if method in {"supervised_from_scratch", "simclr"}:
                embeddings = model.encoder(images)
            elif method in {"mae", "mae_improved"}:
                embeddings = model.encoder.forward_encoder(images)
            else:
                raise ValueError(method)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE embeddings",
    seed: int = 42,
    save_path: Optional[str] = None,
) -> None:
    reduced = TSNE(
        n_components=2,
        random_state=seed,
        init="random",
        learning_rate="auto",
    ).fit_transform(embeddings)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels,
                    palette="Set1", s=18)
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# Grad-CAM (ResNet)

class GradCAM:
    """Grad-CAM implementation suitable for ResNet-based classifiers.
    """

    def __init__(self, model: nn.Module, target_module: nn.Module):
        self.model = model
        self.target_module = target_module
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_module.register_forward_hook(forward_hook)
        self.target_module.register_full_backward_hook(backward_hook)

    def __call__(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None):
        self.model.eval()
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
