"""Locate the fly so the classifier can look at it at useful resolution.

The camera is fixed and the fly covers roughly 4% of the frame, so a network fed the whole
frame at 160 px would see a proboscis one pixel wide. Instead we find the fly, follow it
with a smoothed window and hand the network a fly-centred crop.

Detection needs no background model, which matters because many clips are a few seconds of
a fly that never moves: the fly is simply the largest blob that is darker than its
surroundings. "Surroundings" is a grey-scale closing with a kernel larger than the fly,
which paints over every dark object smaller than the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class TrackerConfig:
    downscale: int = 4  # detection runs on block-averaged frames
    close_frac: float = 0.22  # closing kernel side, as a fraction of the low-res width
    min_contrast: float = 6.0  # grey levels; floor for the darkness threshold
    rel_contrast: float = 0.35  # threshold as a fraction of the darkest structures
    min_area_frac: float = 0.008  # plausible blob area range, fraction of frame area
    max_area_frac: float = 0.15
    detect_every: int = 3  # frames between detections (the fly cannot teleport)
    median_seconds: float = 1.0  # outlier rejection window
    smooth_seconds: float = 0.5  # Gaussian sigma of the window trajectory


def block_mean(frame: np.ndarray, k: int) -> np.ndarray:
    """Downscale an ``(H, W)`` image by an integer factor using box averaging."""
    h, w = frame.shape
    h, w = h - h % k, w - w % k
    return frame[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3), dtype=np.float32)


def detect_fly(
    small: np.ndarray, cfg: TrackerConfig = TrackerConfig()
) -> tuple[float, float] | None:
    """Return the ``(x, y)`` centroid of the fly in a low-res frame, or None."""
    h, w = small.shape
    k = int(cfg.close_frac * w) | 1
    smooth = ndi.gaussian_filter(small, 1.0)
    darkness = ndi.grey_closing(smooth, size=(k, k)) - smooth
    threshold = max(cfg.min_contrast, cfg.rel_contrast * float(np.percentile(darkness, 99.7)))
    mask = ndi.binary_opening(darkness > threshold, structure=np.ones((3, 3), bool), iterations=2)
    labels, n = ndi.label(mask)
    if n == 0:
        return None
    areas = np.bincount(labels.ravel())[1:]
    plausible = (areas >= cfg.min_area_frac * h * w) & (areas <= cfg.max_area_frac * h * w)
    if not plausible.any():
        return None
    best = 1 + int(np.argmax(np.where(plausible, areas, 0)))
    # The blob includes wings, legs and their shadows, which drag a plain centroid away
    # from the body by up to a third of a body length -- enough to push the head out of
    # the crop. Keep only the thick part of the blob (thorax and abdomen) and centre on it.
    rows, cols = ndi.find_objects(labels)[best - 1]
    blob = np.pad(labels[rows, cols] == best, 1)
    depth = ndi.distance_transform_edt(blob)
    cy, cx = ndi.center_of_mass(depth > 0.5 * depth.max())
    return float(cx) + cols.start - 1, float(cy) + rows.start - 1


def smooth_track(raw: np.ndarray, fps: float, cfg: TrackerConfig = TrackerConfig()) -> np.ndarray:
    """Turn sparse, noisy detections into a smooth per-frame window trajectory.

    ``raw`` is ``(n_frames, 2)`` with NaN where no detection was made. A running median
    rejects frames where the detector locked onto something else, and a Gaussian removes
    the jitter caused by legs and wings changing the blob outline. Without the second step
    the crop window would shake and inject motion the fly never made.
    """
    n = len(raw)
    ok = ~np.isnan(raw[:, 0])
    if not ok.any():
        return np.full((n, 2), np.nan, dtype=np.float32)
    t = np.arange(n)
    dense = np.stack([np.interp(t, t[ok], raw[ok, d]) for d in range(2)], axis=1)
    med = max(3, int(cfg.median_seconds * fps) | 1)
    dense = ndi.median_filter(dense, size=(med, 1), mode="nearest")
    dense = ndi.gaussian_filter1d(dense, cfg.smooth_seconds * fps, axis=0, mode="nearest")
    return dense.astype(np.float32)


def crop_origin(center: np.ndarray, crop: int, width: int, height: int) -> tuple[int, int]:
    """Top-left corner of a ``crop`` x ``crop`` window centred on ``center``, kept in frame."""
    x0 = int(round(float(center[0]) - crop / 2))
    y0 = int(round(float(center[1]) - crop / 2))
    return min(max(x0, 0), max(width - crop, 0)), min(max(y0, 0), max(height - crop, 0))
