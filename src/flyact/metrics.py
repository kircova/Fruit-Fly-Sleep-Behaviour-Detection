"""Clip-level metrics with uncertainty estimates that respect the data's structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import CLASSES

PROB_COLS = [f"p_{c}" for c in CLASSES]


def predictions_frame(clips: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Attach class probabilities to the identifying columns of ``clips``."""
    out = clips[["clip_id", "label", "group", "fold", "n_frames"]].reset_index(drop=True).copy()
    out["label"] = out["label"].astype(str)
    out[PROB_COLS] = probs
    out["pred"] = [CLASSES[i] for i in probs.argmax(axis=1)]
    return out


def _scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    labels = np.arange(len(CLASSES))
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
    }


def _codes(preds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lookup = {c: i for i, c in enumerate(CLASSES)}
    return preds["label"].map(lookup).to_numpy(), preds["pred"].map(lookup).to_numpy()


def bootstrap_ci(
    preds: pd.DataFrame, by_group: bool, n_boot: int = 2000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """95% percentile intervals for the headline scores.

    ``by_group=False`` resamples clips, which answers "how sure are we about these clips".
    ``by_group=True`` resamples whole recordings, which answers the question that matters,
    "how would this look on a new set of flies", and is correspondingly wider.
    """
    rng = np.random.default_rng(seed)
    y, pred = _codes(preds)
    if by_group:
        members = [np.flatnonzero(preds["group"].to_numpy() == g) for g in preds["group"].unique()]
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        if by_group:
            picked = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[i] for i in picked])
        else:
            idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < len(CLASSES):
            continue  # a resample without one of the classes has no defined macro score
        for name, value in _scores(y[idx], pred[idx]).items():
            draws.setdefault(name, []).append(value)
    return {k: tuple(np.percentile(v, [2.5, 97.5]).round(4).tolist()) for k, v in draws.items()}


def summarize(preds: pd.DataFrame) -> dict:
    """Headline scores, per-class scores, confusion matrix and per-recording accuracy."""
    y, pred = _codes(preds)
    labels = np.arange(len(CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0
    )
    per_group = (
        preds.assign(correct=(y == pred))
        .groupby("group")
        .agg(clips=("correct", "size"), accuracy=("correct", "mean"), fold=("fold", "first"))
    )
    return {
        **{k: round(float(v), 4) for k, v in _scores(y, pred).items()},
        "n_clips": int(len(preds)),
        "ci95_clips": bootstrap_ci(preds, by_group=False),
        "ci95_recordings": bootstrap_ci(preds, by_group=True),
        "per_class": {
            c: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "per_group": {
            int(g): {
                "clips": int(r.clips),
                "accuracy": round(float(r.accuracy), 4),
                "fold": int(r.fold),
            }
            for g, r in per_group.iterrows()
        },
    }
