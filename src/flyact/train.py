"""Training and grouped cross-validation.

There is no early stopping and no checkpoint selection: every model is trained for a fixed
schedule and the weights after the last step are the ones evaluated. Test folds are scored
during training only to draw learning curves, never to choose anything.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from . import CLASSES
from .config import ExperimentConfig, load_config
from .data import WindowDataset, balanced_sampler
from .index import read_manifest
from .metrics import predictions_frame, summarize
from .model import FlyNet


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    # Weight decay on normalisation parameters and biases only hurts.
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    """Linear warm-up followed by cosine decay to zero."""
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


@torch.no_grad()
def predict_clips(
    model: torch.nn.Module,
    clips: pd.DataFrame,
    cache_dir: Path,
    cfg: ExperimentConfig,
    device: torch.device,
    workers: int,
) -> np.ndarray:
    """Class probabilities per clip: the mean over its evaluation windows."""
    dataset = WindowDataset(clips, cache_dir, cfg.sampling, train=False)
    loader = DataLoader(dataset, batch_size=2 * cfg.train.batch_size, num_workers=workers)
    total = torch.zeros(len(clips), len(CLASSES))
    count = torch.zeros(len(clips))
    model.eval()
    for x, _, clip in loader:
        probs = F.softmax(model(x.to(device)), dim=1).cpu()
        total.index_add_(0, clip, probs)
        count.index_add_(0, clip, torch.ones(len(clip)))
    return (total / count[:, None]).numpy()


def train_model(
    train_clips: pd.DataFrame,
    cache_dir: Path,
    cfg: ExperimentConfig,
    device: torch.device,
    workers: int,
    monitor: pd.DataFrame | None = None,
    log: list | None = None,
) -> torch.nn.Module:
    """Train one model on ``train_clips``; optionally track accuracy on ``monitor``."""
    tc = cfg.train
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    dataset = WindowDataset(train_clips, cache_dir, cfg.sampling, train=True, augment=tc.augment)
    loader = DataLoader(
        dataset,
        batch_size=tc.batch_size,
        sampler=balanced_sampler(dataset.labels, len(dataset), tc.seed),
        num_workers=workers,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    model = FlyNet(cfg.model, cfg.sampling.num_snippets, len(CLASSES)).to(device)
    optimizer = make_optimizer(model, tc.lr, tc.weight_decay)
    total_steps = tc.epochs * len(loader)
    warmup_steps = int(tc.warmup_epochs * len(loader))
    step = 0
    for epoch in range(1, tc.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = correct = seen = 0.0
        for x, y, _ in loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, warmup_steps, tc.lr)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=tc.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        entry = {
            "epoch": epoch,
            "train_loss": round(loss_sum / seen, 4),
            "train_acc": round(correct / seen, 4),
            "seconds": round(time.time() - t0, 1),
        }
        last = epoch == tc.epochs
        if monitor is not None and not last and tc.eval_every and epoch % tc.eval_every == 0:
            probs = predict_clips(model, monitor, cache_dir, cfg, device, workers)
            y_true = monitor["label"].cat.codes.to_numpy()
            entry["monitor_acc"] = round(float((probs.argmax(1) == y_true).mean()), 4)
        print("   " + "  ".join(f"{k}={v}" for k, v in entry.items()), flush=True)
        if log is not None:
            log.append(entry)
    return model


def make_folds(df: pd.DataFrame, cfg: ExperimentConfig) -> np.ndarray:
    if cfg.split == "group":
        return df["fold"].to_numpy(dtype=int)
    if cfg.split == "random":
        # Same number of folds, same class balance, but recordings are ignored.
        fold = np.empty(len(df), dtype=int)
        splitter = StratifiedKFold(int(df["fold"].max()) + 1, shuffle=True, random_state=0)
        for k, (_, test) in enumerate(splitter.split(df, df["label"].cat.codes)):
            fold[test] = k
        return fold
    raise ValueError(f"unknown split '{cfg.split}'")


def run_cross_validation(
    df: pd.DataFrame,
    cache_dir: Path,
    cfg: ExperimentConfig,
    out_dir: Path,
    workers: int,
    only_folds: list[int] | None = None,
) -> pd.DataFrame:
    device = pick_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    folds = make_folds(df, cfg)
    all_preds = []
    for k in sorted(set(folds.tolist())):
        if only_folds and k not in only_folds:
            continue
        fold_file = out_dir / f"fold{k}.csv"
        if fold_file.exists():
            all_preds.append(pd.read_csv(fold_file, dtype={"clip_id": str}))
            continue
        train_clips, test_clips = df[folds != k], df[folds == k]
        print(f"[{cfg.name}] fold {k}: train {len(train_clips)} clips, test {len(test_clips)}")
        log: list = []
        t0 = time.time()
        model = train_model(train_clips, cache_dir, cfg, device, workers, test_clips, log)
        probs = predict_clips(model, test_clips, cache_dir, cfg, device, workers)
        preds = predictions_frame(test_clips, probs)
        preds["fold"] = k
        acc = float((preds["pred"] == preds["label"]).mean())
        print(f"   test accuracy {acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        (out_dir / f"fold{k}_log.json").write_text(json.dumps(log, indent=1))
        if cfg.train.save_checkpoints:
            save_checkpoint(model, cfg, out_dir / f"fold{k}.pt")
        preds.to_csv(fold_file, index=False)
        all_preds.append(preds)
        del model
    oof = pd.concat(all_preds, ignore_index=True)
    if only_folds is None:
        oof.to_csv(out_dir / "oof.csv", index=False)
        summary = summarize(oof)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"[{cfg.name}] out-of-fold: accuracy {summary['accuracy']:.4f}  "
            f"balanced {summary['balanced_accuracy']:.4f}  macro-F1 {summary['macro_f1']:.4f}"
        )
    return oof


def save_checkpoint(model: torch.nn.Module, cfg: ExperimentConfig, path: Path) -> None:
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"config": cfg.to_dict(), "classes": list(CLASSES), "state_dict": state}, path)


def run_from_args(args) -> None:
    cfg = load_config(args.config, args.set)
    df = read_manifest(args.manifest)
    df = df[df["usable"]].reset_index(drop=True)
    out_dir = args.out or Path("runs") / cfg.name
    if args.final:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{cfg.name}] training on all {len(df)} clips")
        model = train_model(df, args.cache, cfg, pick_device(), args.workers)
        save_checkpoint(model, cfg, out_dir / "final.pt")
        print(f"saved {out_dir / 'final.pt'}")
        return
    run_cross_validation(df, args.cache, cfg, out_dir, args.workers, args.folds)
