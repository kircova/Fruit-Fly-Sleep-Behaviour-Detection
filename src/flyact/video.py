"""Video I/O built on PyAV.

Two kinds of files pass through here:

* raw recordings (large H.264 AVIs) which are only ever read front to back, and
* the crop cache (small H.264 MP4s written by us) which is read with random access
  during training.

The cache is written with a short fixed GOP and no B-frames so that a seek lands at most
``GOP - 1`` frames before the requested frame and presentation order equals decode order.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

CACHE_GOP = 16


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int  # as reported by the container; 0 if unknown


def probe(path: str | Path) -> VideoInfo:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or Fraction(30)
        return VideoInfo(
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=float(rate),
            n_frames=int(stream.frames or 0),
        )


def iter_gray_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield every frame of a video as an ``(H, W)`` uint8 luminance array."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="gray")


class GrayVideoWriter:
    """Write ``(H, W)`` uint8 frames to a seekable H.264 MP4."""

    def __init__(self, path: str | Path, size: int, fps: int = 30, crf: int = 16):
        self._container = av.open(str(path), mode="w")
        stream = self._container.add_stream("libx264", rate=fps)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = CACHE_GOP
        stream.codec_context.max_b_frames = 0
        stream.codec_context.time_base = Fraction(1, fps)
        # scenecut=0 keeps keyframes on the fixed GOP grid, which bounds seek cost.
        stream.options = {"crf": str(crf), "preset": "medium", "x264-params": "scenecut=0"}
        self._stream = stream
        self._n = 0

    def write(self, frame: np.ndarray) -> None:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="gray")
        vf = vf.reformat(format="yuv420p")
        vf.pts = self._n
        vf.time_base = self._stream.codec_context.time_base
        for packet in self._stream.encode(vf):
            self._container.mux(packet)
        self._n += 1

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()

    def __enter__(self) -> GrayVideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_cached_frames(path: str | Path, indices: Sequence[int]) -> np.ndarray:
    """Read the requested frame indices from a cache file as ``(len(indices), H, W)`` uint8.

    Indices may repeat and need not be sorted. One seek is issued to the keyframe at or
    before the smallest index and frames are decoded forward from there.
    """
    wanted = np.asarray(indices, dtype=np.int64)
    if wanted.size == 0:
        raise ValueError("no frame indices requested")
    first, last = int(wanted.min()), int(wanted.max())
    found: dict[int, np.ndarray] = {}
    need = set(wanted.tolist())
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = stream.average_rate or Fraction(30)
        ticks_per_frame = 1 / (fps * stream.time_base)
        if first >= CACHE_GOP:
            container.seek(int(first * ticks_per_frame), stream=stream, backward=True)
        for frame in container.decode(stream):
            idx = int(round(frame.pts / ticks_per_frame))
            if idx in need:
                found[idx] = frame.to_ndarray(format="gray")
            if idx >= last:
                break
    missing = need - found.keys()
    if missing:
        raise OSError(f"{path}: could not decode frames {sorted(missing)[:5]}...")
    return np.stack([found[int(i)] for i in wanted])
