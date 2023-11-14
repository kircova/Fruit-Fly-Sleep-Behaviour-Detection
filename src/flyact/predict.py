"""Classify a new video with a trained checkpoint.

The video goes through exactly the same steps as the training clips (fly tracking, crop,
windowing), then every window is classified. The model is closed-set: it always answers
one of the three behaviours it was trained on, including for footage that shows none of
them (a sleeping or walking fly), so its output is only meaningful on bouts that someone
or something has already identified as one of the three.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import CLASSES
from .config import ExperimentConfig, config_from_dict
from .data import effective_stride, to_input, window_indices
from .metrics import PROB_COLS
from .model import FlyNet
from .preprocess import PreprocessConfig, preprocess_clip
from .train import pick_device
from .video import read_cached_frames


def load_model(checkpoint: Path, device: torch.device) -> tuple[FlyNet, ExperimentConfig]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = config_from_dict(state["config"])
    cfg.model.pretrained = False  # the checkpoint already holds every weight
    model = FlyNet(cfg.model, cfg.sampling.num_snippets, len(state["classes"]))
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def classify_video(
    video: Path,
    model: FlyNet,
    cfg: ExperimentConfig,
    device: torch.device,
    hop_seconds: float = 1.0,
    batch_size: int = 32,
) -> pd.DataFrame:
    """One row per window: start and end time in seconds, and class probabilities."""
    with tempfile.TemporaryDirectory() as tmp:
        stats = preprocess_clip("video", video, Path(tmp), PreprocessConfig())
        if not stats["usable"]:
            raise ValueError(f"{video} has only {stats['n_frames']} frames; too short to classify")
        n, fps = int(stats["n_frames"]), float(stats["fps"])
        sc = cfg.sampling
        stride = effective_stride(n, sc)
        span = (sc.num_snippets - 1) * stride + 2 * sc.delta + 1
        hop = max(1, int(round(hop_seconds * fps)))
        starts = list(range(0, max(n - span, 0) + 1, hop))
        probs = []
        for i in range(0, len(starts), batch_size):
            batch = []
            for start in starts[i : i + batch_size]:
                rows = window_indices(n, start, stride, sc)
                unique, inverse = np.unique(rows, return_inverse=True)
                frames = read_cached_frames(Path(tmp) / "video.mp4", unique)
                batch.append(to_input(frames, inverse.reshape(rows.shape), sc.image_size, False))
            probs.append(F.softmax(model(torch.stack(batch).to(device)), dim=1).cpu().numpy())
    out = pd.DataFrame(np.concatenate(probs), columns=PROB_COLS)
    out.insert(0, "end_s", (np.array(starts) + span) / fps)
    out.insert(0, "start_s", np.array(starts) / fps)
    return out


def segments(windows: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive windows that agree into ``(start, end, behaviour, confidence)``."""
    probs = windows[PROB_COLS].to_numpy()
    label = probs.argmax(axis=1)
    run = np.concatenate([[0], np.cumsum(np.diff(label) != 0)])
    rows = []
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        rows.append(
            {
                "start_s": round(float(windows["start_s"].iloc[idx[0]]), 2),
                "end_s": round(float(windows["end_s"].iloc[idx[-1]]), 2),
                "behaviour": CLASSES[label[idx[0]]],
                "confidence": round(float(probs[idx, label[idx[0]]].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def run_from_args(args) -> None:
    device = pick_device()
    model, cfg = load_model(args.checkpoint, device)
    windows = classify_video(args.video, model, cfg, device, args.hop_seconds)
    mean = windows[PROB_COLS].mean().to_numpy()
    print(f"{args.video.name}: {CLASSES[int(mean.argmax())]} (mean probability {mean.max():.2f})")
    print("  " + "  ".join(f"{c}={p:.2f}" for c, p in zip(CLASSES, mean, strict=True)))
    if len(windows) > 1:
        print(segments(windows).to_string(index=False))
    if args.csv:
        windows.round(4).to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
