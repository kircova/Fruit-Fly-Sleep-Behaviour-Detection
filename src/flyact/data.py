"""Datasets: sample windows from the crop cache and turn them into network input."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from .config import SamplingConfig
from .video import read_cached_frames


def effective_stride(n_frames: int, cfg: SamplingConfig, stride: int | None = None) -> int:
    """Largest snippet stride <= the requested one that still fits inside the clip."""
    stride = cfg.stride if stride is None else stride
    room = n_frames - 1 - 2 * cfg.delta
    return max(1, min(stride, room // max(cfg.num_snippets - 1, 1)))


def window_indices(n_frames: int, start: int, stride: int, cfg: SamplingConfig) -> np.ndarray:
    """Frame indices of one window as ``(num_snippets, 3)``.

    Row ``j`` holds the three channels of snippet ``j``. Indices are clamped to the clip, so
    a clip shorter than the window repeats its boundary frames instead of failing.
    """
    centres = start + cfg.delta + stride * np.arange(cfg.num_snippets)
    offsets = np.array([-cfg.delta, 0, cfg.delta]) if cfg.input_mode == "triplet" else np.zeros(3)
    return np.clip(centres[:, None] + offsets[None, :], 0, n_frames - 1).astype(np.int64)


def eval_starts(n_frames: int, cfg: SamplingConfig) -> list[int]:
    """Evenly spaced window starts covering the clip, at most ``cfg.eval_windows`` of them."""
    stride = effective_stride(n_frames, cfg)
    span = (cfg.num_snippets - 1) * stride + 2 * cfg.delta + 1
    last = max(n_frames - span, 0)
    count = min(cfg.eval_windows, 1 + math.ceil(2 * last / span))  # ~50% overlap, then capped
    return [int(round(s)) for s in np.linspace(0, last, count)]


def to_input(
    frames: np.ndarray,
    rows: np.ndarray,
    size: int,
    augment: bool,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """``(F, H, W)`` uint8 frames -> ``(T, 3, size, size)`` float input.

    ``rows`` is ``(T, 3)`` and indexes into ``frames``. One affine transform and one
    photometric transform are drawn per window and applied to every frame alike, so
    augmentation never fabricates motion.
    """

    def rand(lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand((), generator=rng))

    x = torch.from_numpy(frames).float().div_(255).unsqueeze(1)  # (F, 1, H, W)
    theta = torch.eye(2, 3)
    if augment:
        angle = math.radians(rand(-15, 15))
        zoom = rand(0.8, 1.25)  # > 1 zooms in
        flip = -1.0 if rand(0, 1) < 0.5 else 1.0  # mirror left-right; the port stays on top
        cos, sin = math.cos(angle) / zoom, math.sin(angle) / zoom
        theta = torch.tensor(
            [[cos * flip, -sin, rand(-0.08, 0.08)], [sin * flip, cos, rand(-0.08, 0.08)]]
        )
        x = x.clamp_min_(1e-4).pow_(rand(0.7, 1.4))  # gamma: exposure differs between sessions
    grid = F.affine_grid(theta.expand(len(x), 2, 3), (len(x), 1, size, size), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # Standardise per window: recordings differ in brightness by a factor of two.
    x = (x - x.mean()) / (x.std() + 1e-4)
    if augment:
        x = x * rand(0.8, 1.2) + rand(-0.3, 0.3)
    return x[:, 0][torch.from_numpy(rows)]  # (T, 3, size, size)


class WindowDataset(Dataset):
    """Windows cut from cached clips.

    In training mode every item is one randomly placed window of a clip; in evaluation mode
    the items enumerate a fixed set of windows per clip, to be averaged by the caller.
    """

    def __init__(
        self,
        clips: pd.DataFrame,
        cache_dir: Path,
        cfg: SamplingConfig,
        train: bool,
        augment: bool = True,
    ):
        self.clip_ids = clips["clip_id"].tolist()
        self.n_frames = clips["n_frames"].astype(int).tolist()
        self.labels = clips["label"].cat.codes.astype(int).tolist()
        self.cache_dir = Path(cache_dir)
        self.cfg = cfg
        self.train = train
        self.augment = augment and train
        self.items = (
            [(i, -1) for i in range(len(self.clip_ids))]
            if train
            else [(i, s) for i, n in enumerate(self.n_frames) for s in eval_starts(n, cfg)]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, int]:
        clip, start = self.items[item]
        n = self.n_frames[clip]
        cfg = self.cfg
        if self.train:
            # Jittering the stride changes apparent speed by -25%..+33%, which also stops
            # the network from reading clip length off the playback rate of short clips.
            wanted = round(cfg.stride * float(torch.empty(()).uniform_(0.75, 1.33)))
            stride = effective_stride(n, cfg, max(1, wanted))
            span = (cfg.num_snippets - 1) * stride + 2 * cfg.delta + 1
            start = int(torch.randint(0, max(n - span, 0) + 1, ()))
        else:
            stride = effective_stride(n, cfg)
        rows = window_indices(n, start, stride, cfg)
        unique, inverse = np.unique(rows, return_inverse=True)
        frames = read_cached_frames(self.cache_dir / f"{self.clip_ids[clip]}.mp4", unique)
        x = to_input(frames, inverse.reshape(rows.shape), cfg.image_size, self.augment)
        return x, self.labels[clip], clip


def balanced_sampler(labels: list[int], num_samples: int, seed: int) -> WeightedRandomSampler:
    """Draw clips so that every class is seen equally often."""
    counts = np.bincount(labels)
    weights = 1.0 / counts[np.asarray(labels)]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights.tolist(), num_samples, replacement=True, generator=generator
    )
