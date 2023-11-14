"""Baselines that use no neural network, scored on exactly the same folds.

They answer two questions a video model's accuracy cannot answer on its own:

* **position** -- how far does knowing *where* the fly is get you? Feeding happens at the
  food port, so location alone is a shortcut. If this scored as well as the network, the
  network would not need to have learned anything about behaviour.
* **motion** -- is the task solvable from how much the image changes, how fast, and
  roughly where, without looking at the animal? If so, a deep network is unnecessary.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import CLASSES
from .index import read_manifest
from .metrics import predictions_frame, summarize
from .video import read_cached_frames

POSITION_FEATURES = ["x_mean", "y_mean", "x_std", "y_std", "speed"]
CHUNK = 64  # frames per analysed chunk (2.1 s)
MAX_CHUNKS = 12
BANDS_HZ = [(0.5, 2), (2, 5), (5, 10), (10, 15)]
MOTION_FEATURES = (
    [f"energy_{s}" for s in ("mean", "std", "p10", "p50", "p90", "max")]
    + [f"band_{lo}-{hi}Hz" for lo, hi in BANDS_HZ]
    + ["concentration"]
    + [f"where_{r}{c}" for r in range(4) for c in range(4)]
)


def motion_features(args: tuple[Path, int, float]) -> np.ndarray:
    """Frame-difference statistics of one cached clip (see ``MOTION_FEATURES``)."""
    path, n_frames, fps = args
    length = min(CHUNK, n_frames)
    starts = np.linspace(0, n_frames - length, min(MAX_CHUNKS, 1 + (n_frames - length) // CHUNK))
    energy, bands, maps = [], [], []
    for start in starts.astype(int):
        frames = read_cached_frames(path, np.arange(start, start + length)).astype(np.float32)
        f, h, w = frames.shape
        frames = frames.reshape(f, h // 2, 2, w // 2, 2).mean(axis=(2, 4))  # suppress sensor noise
        diff = np.abs(np.diff(frames, axis=0))
        series = diff.mean(axis=(1, 2))
        energy.append(series)
        maps.append(diff.mean(axis=0))
        if len(series) >= 32:
            power = np.abs(np.fft.rfft(series - series.mean())) ** 2
            freqs = np.fft.rfftfreq(len(series), d=1 / fps)
            total = power[freqs >= BANDS_HZ[0][0]].sum() + 1e-9
            bands.append([power[(freqs >= lo) & (freqs < hi)].sum() / total for lo, hi in BANDS_HZ])
    energy = np.concatenate(energy)
    where = np.mean(maps, axis=0)
    top = np.sort(where.ravel())[-where.size // 20 :].sum() / (where.sum() + 1e-9)
    grid = where.reshape(4, where.shape[0] // 4, 4, where.shape[1] // 4).sum(axis=(1, 3))
    return np.concatenate(
        [
            [energy.mean(), energy.std(), *np.percentile(energy, [10, 50, 90]), energy.max()],
            np.mean(bands, axis=0) if bands else np.full(len(BANDS_HZ), np.nan),
            [top],
            (grid / (grid.sum() + 1e-9)).ravel(),
        ]
    )


def extract_motion_features(df: pd.DataFrame, cache_dir: Path, workers: int) -> pd.DataFrame:
    jobs = [
        (cache_dir / f"{r.clip_id}.mp4", int(r.n_frames), float(r.fps)) for r in df.itertuples()
    ]
    with ProcessPoolExecutor(workers) as pool:
        feats = list(pool.map(motion_features, jobs, chunksize=8))
    return pd.DataFrame(np.stack(feats), columns=MOTION_FEATURES, index=df.index)


def cross_validate(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold predictions of a small gradient-boosted model on tabular features."""
    y = df["label"].cat.codes.to_numpy()
    folds = df["fold"].to_numpy(dtype=int)
    probs = np.zeros((len(df), len(CLASSES)))
    for k in np.unique(folds):
        model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=0
        )
        model.fit(features[folds != k], y[folds != k])
        probs[folds == k] = model.predict_proba(features[folds == k])
    return predictions_frame(df, probs)


def majority(df: pd.DataFrame) -> pd.DataFrame:
    """Always predict the most common class of the training folds."""
    y = df["label"].cat.codes.to_numpy()
    folds = df["fold"].to_numpy(dtype=int)
    probs = np.zeros((len(df), len(CLASSES)))
    for k in np.unique(folds):
        probs[folds == k, np.bincount(y[folds != k]).argmax()] = 1.0
    return predictions_frame(df, probs)


def run_from_args(args) -> None:
    df = read_manifest(args.manifest)
    df = df[df["usable"]].reset_index(drop=True)
    features_file = args.cache / "motion_features.csv"
    if features_file.exists():
        motion = pd.read_csv(features_file, index_col="clip_id").loc[df["clip_id"]]
        motion.index = df.index
    else:
        motion = extract_motion_features(df, args.cache, args.workers)
        motion.assign(clip_id=df["clip_id"]).set_index("clip_id").to_csv(features_file)
    position = df[POSITION_FEATURES]
    runs = {
        "majority": majority(df),
        "position": cross_validate(df, position),
        "motion": cross_validate(df, motion),
        "position_motion": cross_validate(df, pd.concat([position, motion], axis=1)),
    }
    for name, preds in runs.items():
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out / "oof.csv", index=False)
        summary = summarize(preds)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"{name:16s} accuracy {summary['accuracy']:.4f}  "
            f"balanced {summary['balanced_accuracy']:.4f}  macro-F1 {summary['macro_f1']:.4f}"
        )
