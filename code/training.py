"""
Training loops, loss functions, and experiment orchestration for the
PCam histopathology SSL project.

Includes:
    - set_seed, device utilities.
    - nt_xent_loss, patchify helpers.
    - EMA, get_cosine_schedule_with_warmup.
    - train_simclr, train_mae, train_mae_improved.
    - TrainConfig, train_classifier.
    - run_single_experiment, run_experiment_grid.

Usage from notebook:
    from src.training import (
        set_seed, device, nt_xent_loss,
        train_simclr, train_mae, train_mae_improved,
        TrainConfig, train_classifier,
        run_single_experiment, run_experiment_grid,
    )
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Global config 
@dataclass
class TrainConfig:
    epochs: int = 15
    encoder_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3

DEFAULT_EXPERIMENT_CONFIG = TrainConfig()
MAIN_METHODS = ["supervised_from_scratch", "simclr", "mae"]
FINETUNE_STRATEGIES = ["frozen", "partial", "full", "peft"]

from data import (
    DataBundle,
    LABEL_FRACTIONS,
    SSLImageDataset,
    SSLPairDataset,
    SimCLRTransform,
    make_mae_transform,
)
from evaluation import evaluate_model
from models import (
    MAEViT,
    MAEViTImproved,
    SimCLRModel,
    build_mae_classifier,
    build_mae_improved_classifier,
    build_simclr_classifier,
    build_supervised_scratch_classifier,
    configure_trainable_parameters,
    LoRAResNetClassifier,           
    VPTMAEClassifier,               
)


# Global utilities

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# SimCLR loss

def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent (normalised temperature-scaled cross entropy) loss from SimCLR.

    For each anchor z_i[k], the positive is z_j[k] and negatives are all
    other embeddings in the concatenated batch.
    """
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size),
    ]).to(z.device)

    return F.cross_entropy(sim, labels)


# MAE helpers

def patchify(imgs: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """Convert a batch of images into a sequence of flattened patches.

    The patch_size must match the MAE variant: 8 for MAEViT, 4 for MAEViTImproved.
    """
    n, c, h, w = imgs.shape
    x = imgs.reshape(n, c, h // patch_size, patch_size, w // patch_size, patch_size)
    x = torch.einsum("nchpwq->nhwcpq", x)
    x = x.reshape(n, (h // patch_size) * (w // patch_size), patch_size ** 2 * c)
    return x


class EMA:
    """Exponential moving average of model parameters used during MAE-Improved.

    apply_shadow() swaps in the EMA weights; restore() restores the current
    training weights. In this codebase, we keep the shadow internal and the
    checkpoint saves the training model's state_dict directly.
    """

    def __init__(self, model: nn.Module, decay: float = 0.996):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine annealing schedule with a linear warmup phase."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Phase 1: SSL pre-training

def train_simclr(
    data: DataBundle,
    checkpoint_dir: Path,
    epochs: int = 20,                    # 10 to 20
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    warmup_epochs: int = 2,              
    weight_decay: float = 0.05,          
    tailored: bool = True,
    checkpoint_name: str = "simclr_encoder_tailored.pth",
    num_workers: int = 2,
    seed: int = SEED,
):
    """Train a SimCLR encoder on the pretrain subset and save its weights.

    Returns (model, history, checkpoint_path).
    """
    set_seed(seed)
    transform = SimCLRTransform(
        channel_mean=data.channel_mean,
        channel_std=data.channel_std,
        tailored=tailored,
    )
    dataset = SSLPairDataset(data.x_pretrain, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=num_workers)
    
    model = SimCLRModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    
    total_steps = epochs * len(loader) 
    warmup_steps = warmup_epochs * len(loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps) 

    history = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xi, xj in loader:
            xi = xi.to(device)
            xj = xj.to(device)
            _, zi = model(xi)
            _, zj = model(xj)
            loss = nt_xent_loss(zi, zj)

            optimizer.zero_grad()
            loss.backward()       
            optimizer.step()
            scheduler.step() 
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch + 1, "loss": avg_loss,
                        "lr": current_lr})  
        print(f"SimCLR epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} "
              f"- lr: {current_lr:.6f}") 

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(model.encoder.state_dict(), checkpoint_path)
    return model, history, checkpoint_path


def train_mae(
    data: DataBundle,
    checkpoint_dir: Path,
    epochs: int = 20,                    # 10 to 20
    batch_size: int = 128,
    learning_rate: float = 1.5e-4,       # 1e-3 to 1.5e-4
    warmup_epochs: int = 2,              
    weight_decay: float = 0.05,          
    checkpoint_name: str = "mae_encoder.pth",
    num_workers: int = 2,
    seed: int = SEED,
):
    """Train a base MAEViT and save the full model state_dict."""
    set_seed(seed)
    transform = make_mae_transform(data.channel_mean, data.channel_std)
    dataset = SSLImageDataset(data.x_pretrain, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = MAEViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)           # for weight_decay

    total_steps = epochs * len(loader)                                  
    warmup_steps = warmup_epochs * len(loader)                          
    scheduler = get_cosine_schedule_with_warmup(optimizer,              
                                                warmup_steps, total_steps)

    history = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs in loader:
            imgs = imgs.to(device)
            pred, mask = model(imgs)
            target = patchify(imgs, patch_size=model.patch_size)

            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()                                           # ← 新加
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        current_lr = optimizer.param_groups[0]["lr"]                   # ← 新加
        history.append({"epoch": epoch + 1, "loss": avg_loss,
                        "lr": current_lr})                             # ← 带 lr
        print(f"MAE epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} "
              f"- lr: {current_lr:.6f}")  

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(model.state_dict(), checkpoint_path)
    return model, history, checkpoint_path


def train_mae_improved(
    data: DataBundle,
    checkpoint_dir: Path,
    epochs: int = 15,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    warmup_epochs: int = 2,
    weight_decay: float = 0.05,
    mask_ratio: float = 0.65,
    ema_decay: float = 0.996,
    checkpoint_name: str = "mae_improved_encoder.pth",
    num_workers: int = 2,
    seed: int = SEED,
):
    """Train the improved MAE variant: deeper, warmup + cosine LR, EMA, grad clip."""
    set_seed(seed)
    transform = make_mae_transform(data.channel_mean, data.channel_std)
    dataset = SSLImageDataset(data.x_pretrain, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = MAEViTImproved(mask_ratio=mask_ratio).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ema = EMA(model, decay=ema_decay)
    history = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs in loader:
            imgs = imgs.to(device)
            pred, mask = model(imgs)
            target = patchify(imgs, patch_size=model.patch_size)

            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch + 1, "loss": avg_loss, "lr": current_lr})
        print(
            f"MAE-Improved epoch {epoch + 1}/{epochs} - "
            f"loss: {avg_loss:.4f} - lr: {current_lr:.6f}"
        )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(model.state_dict(), checkpoint_path)
    return model, history, checkpoint_path



# Phase 2: downstream classifier training

def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    strategy: str,
    config: TrainConfig,
):
    """Fine-tune a classifier on a labelled subset with val-based early stopping.

    Returns (best_model, history, best_epoch, best_val_auc).
    """
    model = model.to(device)
    configure_trainable_parameters(model, strategy)

    encoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "fc" not in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "fc" in n]

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": config.encoder_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": config.head_lr})

    optimizer = torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_auc = -np.inf
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_f1": val_metrics["f1"],
            "val_acc": val_metrics["accuracy"],
        })

        print(
            f"epoch {epoch + 1}/{config.epochs} | train_loss={train_loss:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch, best_val_auc


def run_single_experiment(
    method: str,
    strategy: str,
    label_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    simclr_checkpoint: Optional[Path] = None,
    mae_checkpoint: Optional[Path] = None,
    mae_improved_checkpoint: Optional[Path] = None,
    peft_rank: int = 8,                
    peft_num_prompts: int = 10,         
    seed: int = SEED,
    config: Optional[TrainConfig] = None,
) -> Dict:
    """Run one (method, strategy, label_name) combination end-to-end."""
    if config is None:
        config = DEFAULT_EXPERIMENT_CONFIG
    set_seed(seed)

    if method == "supervised_from_scratch":
        model = build_supervised_scratch_classifier()
    elif method == "simclr":
        if simclr_checkpoint is None:
            raise ValueError("simclr_checkpoint must be provided for SimCLR experiments.")
        model = build_simclr_classifier(simclr_checkpoint)
    elif method == "mae":
        if mae_checkpoint is None:
            raise ValueError("mae_checkpoint must be provided for MAE experiments.")
        model = build_mae_classifier(mae_checkpoint)
    elif method == "mae_improved":
        if mae_improved_checkpoint is None:
            raise ValueError("mae_improved_checkpoint must be provided for MAE Improved experiments.")
        model = build_mae_improved_classifier(mae_improved_checkpoint)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if strategy == "peft":
        if method == "simclr":
            model = LoRAResNetClassifier(
                model.encoder, feature_dim=512, rank=peft_rank,
            )
        elif method in {"mae", "mae_improved"}:
            model = VPTMAEClassifier(
                model.encoder, num_prompts=peft_num_prompts, embed_dim=192,
            )
        else:
            raise ValueError(
                f"PEFT strategy not supported for method '{method}'. "
                "Use 'simclr', 'mae', or 'mae_improved'."
            )

    model, history, best_epoch, best_val_auc = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        strategy=strategy,
        config=config,
    )

    test_metrics = evaluate_model(model, test_loader)
    return {
        "method": method,
        "strategy": strategy,
        "label_fraction": label_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
        "history": history,
        "y_true": test_metrics["y_true"],
        "y_prob": test_metrics["y_prob"],
        "y_pred": test_metrics["y_pred"],
    }


def run_experiment_grid(
    methods: List[str],
    strategies: List[str],
    label_names: List[str],
    seeds: List[int],
    train_loaders: Dict[str, DataLoader],
    val_loader: DataLoader,
    test_loader: DataLoader,
    simclr_checkpoint: Optional[Path] = None,
    mae_checkpoint: Optional[Path] = None,
    mae_improved_checkpoint: Optional[Path] = None,
    config: Optional[TrainConfig] = None,
) -> pd.DataFrame:
    """Iterate over a (method, strategy, label, seed) grid and collect metrics.

    Returns a DataFrame where each row is one experiment.
    """
    if config is None:
        config = DEFAULT_EXPERIMENT_CONFIG
    records = []
    for seed in seeds:
        for method in methods:
            for strategy in strategies:
                if strategy == "peft" and method == "supervised_from_scratch":
                    continue
                for label_name in label_names:
                    print(f"Running {method} | {strategy} | {label_name} | seed={seed}")
                    result = run_single_experiment(
                        method=method,
                        strategy=strategy,
                        label_name=label_name,
                        train_loader=train_loaders[label_name],
                        val_loader=val_loader,
                        test_loader=test_loader,
                        simclr_checkpoint=simclr_checkpoint,
                        mae_checkpoint=mae_checkpoint,
                        mae_improved_checkpoint=mae_improved_checkpoint,
                        seed=seed,
                        config=config,
                    )
                    records.append({
                        "method": result["method"],
                        "strategy": result["strategy"],
                        "label_fraction": result["label_fraction"],
                        "seed": result["seed"],
                        "best_epoch": result["best_epoch"],
                        "best_val_auc": result["best_val_auc"],
                        "test_accuracy": result["test_accuracy"],
                        "test_f1": result["test_f1"],
                        "test_auc": result["test_auc"],
                    })
    return pd.DataFrame(records)
