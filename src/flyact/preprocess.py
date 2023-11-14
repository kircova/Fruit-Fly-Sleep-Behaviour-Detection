"""Build the crop cache: one small fly-centred video per clip, plus per-clip statistics.

Each source clip is decoded twice. The first pass finds the fly; the second cuts a square
window that follows the smoothed trajectory. Two passes keep memory flat regardless of
clip length (the longest clip is 4.4 minutes of 1100x800 video).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .tracking import TrackerConfig, block_mean, crop_origin, detect_fly, smooth_track
from .video import GrayVideoWriter, iter_gray_frames, probe

THUMB_HW = (200, 275)  # every clip's background thumbnail is stored at this size


@dataclass(frozen=True)
class PreprocessConfig:
    # Window side as a fraction of frame width: 572 px on the 1100 px recordings. A fly is
    # about 360 px from head to abdomen tip, so this leaves room for legs, wings and an
    # extended proboscis on either side even when the centre estimate is a little off.
    crop_frac: float = 0.52
    out_size: int = 224
    min_frames: int = 16  # shorter clips are dropped: half a second shows no behaviour
    thumb_samples: int = 48
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


def _resize(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    # Pillow widens the bilinear kernel when shrinking, i.e. it anti-aliases.
    return np.asarray(Image.fromarray(img).resize((hw[1], hw[0]), Image.Resampling.BILINEAR))


def preprocess_clip(clip_id: str, src: Path, cache_dir: Path, cfg: PreprocessConfig) -> dict:
    """Write ``<clip_id>.mp4`` and ``<clip_id>.npz`` into ``cache_dir`` and return clip stats."""
    video_out = cache_dir / f"{clip_id}.mp4"
    stats_out = cache_dir / f"{clip_id}.npz"
    if stats_out.exists():  # written last, so its presence means the clip is finished
        with np.load(stats_out) as z:
            return {k: z[k].item() for k in z.files if z[k].ndim == 0}

    info = probe(src)
    tr = cfg.tracker

    # Pass 1: sparse detections and a decimated set of low-res frames for the thumbnail.
    det_frames, det_xy, thumbs, thumb_stride = [], [], [], 1
    n_frames = 0
    for i, frame in enumerate(iter_gray_frames(src)):
        n_frames = i + 1
        if i % tr.detect_every:
            continue
        small = block_mean(frame, tr.downscale)
        xy = detect_fly(small, tr)
        det_frames.append(i)
        det_xy.append((np.nan, np.nan) if xy is None else xy)
        if (i // tr.detect_every) % thumb_stride == 0:
            thumbs.append(small)
            if len(thumbs) > 2 * cfg.thumb_samples:
                thumbs, thumb_stride = thumbs[::2], thumb_stride * 2

    stats = {
        "clip_id": clip_id,
        "n_frames": n_frames,
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "usable": n_frames >= cfg.min_frames,
    }
    if not stats["usable"]:
        np.savez(stats_out, **stats)
        return stats

    raw = np.full((n_frames, 2), np.nan, dtype=np.float32)
    raw[det_frames] = (np.asarray(det_xy, dtype=np.float32) + 0.5) * tr.downscale
    detected = float(np.mean(~np.isnan(raw[det_frames, 0])))
    track = smooth_track(raw, info.fps, tr)
    if np.isnan(track).any():  # fly never found: fall back to the middle of the frame
        track = np.tile(np.float32([info.width / 2, info.height / 2]), (n_frames, 1))

    # Pass 2: cut the moving window and encode it.
    crop = min(int(round(cfg.crop_frac * info.width)), info.width, info.height)
    tmp = video_out.with_suffix(".tmp.mp4")
    with GrayVideoWriter(tmp, cfg.out_size, fps=int(round(info.fps))) as writer:
        for i, frame in enumerate(iter_gray_frames(src)):
            x0, y0 = crop_origin(track[i], crop, info.width, info.height)
            writer.write(_resize(frame[y0 : y0 + crop, x0 : x0 + crop], (cfg.out_size,) * 2))
    os.replace(tmp, video_out)

    norm = track / np.float32([info.width, info.height])
    stats.update(
        detected_frac=detected,
        crop_px=crop,
        x_mean=float(norm[:, 0].mean()),
        y_mean=float(norm[:, 1].mean()),
        x_std=float(norm[:, 0].std()),
        y_std=float(norm[:, 1].std()),
        # Mean window speed in frame widths per second: > 0 means the fly was walking.
        speed=float(np.linalg.norm(np.diff(track, axis=0), axis=1).mean() * info.fps / info.width)
        if n_frames > 1
        else 0.0,
    )
    thumbnail = _resize(np.median(np.stack(thumbs), axis=0).astype(np.uint8), THUMB_HW)
    np.savez(stats_out, thumbnail=thumbnail, track=track, **stats)
    return stats


def _job(args) -> dict:
    return preprocess_clip(*args)


def build_cache(
    manifest: pd.DataFrame,
    data_root: Path,
    cache_dir: Path,
    cfg: PreprocessConfig = PreprocessConfig(),
    workers: int = 8,
) -> pd.DataFrame:
    """Preprocess every clip in ``manifest``; returns one row of statistics per clip."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (row.clip_id, Path(data_root) / row.rel_path, cache_dir, cfg)
        for row in manifest.itertuples()
    ]
    rows, failed = [], []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(_job, job): job[0] for job in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="preprocess", unit="clip"):
            try:
                rows.append(fut.result())
            except Exception as exc:  # corrupt source file: report it, keep going
                failed.append((futures[fut], repr(exc)))
    for clip_id, err in failed:
        print(f"  could not decode {clip_id}: {err}")
        rows.append({"clip_id": clip_id, "n_frames": 0, "usable": False})
    return pd.DataFrame(rows)


def load_thumbnails(cache_dir: Path, clip_ids: list[str]) -> np.ndarray:
    return np.stack([np.load(cache_dir / f"{c}.npz")["thumbnail"] for c in clip_ids])
