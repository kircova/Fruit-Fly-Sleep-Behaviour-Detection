"""Render the animated figures in the README.

    FLYACT_DATA_ROOT=/path/to/clips python scripts/make_gifs.py

Needs the crop cache, the manifest and the per-fold checkpoints written by
``flyact train --config configs/tsm_triplet.yaml --set train.save_checkpoints=true``.
Every prediction shown comes from the fold model that never saw the clip's recording, so
the animations show held-out behaviour, not a model replaying its training data.
ffmpeg must be on the PATH (it writes far smaller GIFs than Pillow does).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from flyact import CLASSES
from flyact.data import effective_stride, to_input, window_indices
from flyact.index import read_manifest
from flyact.predict import load_model
from flyact.tracking import crop_origin
from flyact.train import pick_device
from flyact.video import iter_gray_frames, read_cached_frames

# Clips are named by content hash, so these ids are stable. All were picked by eye from
# clips the cross-validated model classified, without looking at how the GIF would turn out.
HERO = {"feeding": "36be3a2dc66c", "grooming": "d05e7d630395", "pumping": "b135e913c74a"}
PIPELINE = "37eb62c92099"  # a fly that walks, so the window has something to follow
TIMELINE = "39510c38816e"  # grooming for 20 s, then the fly walks off

SURFACE, PANEL, INK, MUTED = (26, 26, 25), (44, 44, 42), (240, 239, 236), (150, 148, 142)
COLORS = dict(zip(CLASSES, [(57, 135, 229), (217, 89, 38), (25, 158, 112)], strict=True))
TILE = 224


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf" / name), size)


SMALL, BODY, BOLD = font(11), font(12), font(12, bold=True)


class Clip:
    """A cached clip with per-frame class probabilities from its held-out fold model."""

    _models: dict[int, tuple] = {}

    def __init__(self, clip_id: str, manifest, cache: Path, runs: Path, device: torch.device):
        row = manifest[manifest["clip_id"] == clip_id].iloc[0]
        self.row = row
        self.n = int(row["n_frames"])
        self.path = cache / f"{clip_id}.mp4"
        self.track = np.load(cache / f"{clip_id}.npz")["track"]
        fold = int(row["fold"])
        if fold not in self._models:
            self._models[fold] = load_model(runs / f"fold{fold}.pt", device)
        self.probs = self._frame_probabilities(*self._models[fold], device)

    @torch.no_grad()
    def _frame_probabilities(self, model, cfg, device, hop: int = 3) -> np.ndarray:
        sc = cfg.sampling
        stride = effective_stride(self.n, sc)
        span = (sc.num_snippets - 1) * stride + 2 * sc.delta + 1
        starts = np.arange(0, max(self.n - span, 0) + 1, hop)
        out = []
        for i in range(0, len(starts), 32):
            batch = []
            for start in starts[i : i + 32]:
                rows = window_indices(self.n, int(start), stride, sc)
                unique, inverse = np.unique(rows, return_inverse=True)
                frames = read_cached_frames(self.path, unique)
                batch.append(to_input(frames, inverse.reshape(rows.shape), sc.image_size, False))
            out.append(torch.softmax(model(torch.stack(batch).to(device)), 1).cpu().numpy())
        windows = np.concatenate(out)
        centres = starts + span / 2  # a window speaks for the frame in its middle
        frames = np.arange(self.n)
        return np.stack([np.interp(frames, centres, windows[:, k]) for k in range(3)], axis=1)

    def frames(self, indices: np.ndarray) -> np.ndarray:
        return read_cached_frames(self.path, np.clip(indices, 0, self.n - 1))

    def motion_colour(self, indices: np.ndarray, delta: int = 2) -> np.ndarray:
        """Frames ``t - delta, t, t + delta`` as R, G, B: the network's view of motion."""
        stack = np.stack([self.frames(indices + d) for d in (-delta, 0, delta)], axis=-1)
        lo, hi = np.percentile(stack, [0.5, 99.5])
        return (np.clip((stack - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def tile(image: np.ndarray, size: int = TILE) -> Image.Image:
    return Image.fromarray(image).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def draw_bars(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, probs: np.ndarray) -> None:
    """One labelled probability bar per class; the winning class is highlighted."""
    best = int(np.argmax(probs))
    label_w, value_w = 76, 34
    for k, name in enumerate(CLASSES):
        top = y + 19 * k
        ink = INK if k == best else MUTED
        draw.text((x, top), name, font=BOLD if k == best else BODY, fill=ink)
        x0, x1 = x + label_w, x + width - value_w
        draw.rounded_rectangle((x0, top + 4, x1, top + 12), radius=4, fill=PANEL)
        fill = x0 + (x1 - x0) * float(probs[k])
        if fill - x0 >= 4:
            draw.rounded_rectangle((x0, top + 4, fill, top + 12), radius=4, fill=COLORS[name])
        draw.text((x + width, top), f"{probs[k]:.2f}", font=BODY, fill=ink, anchor="ra")


def write_gif(frames: list[Image.Image], path: Path, fps: int, colors: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames):
            frame.save(Path(tmp) / f"{i:04d}.png")
        # One palette for the whole clip; unchanged regions are stored once.
        graph = (
            f"split[a][b];[a]palettegen=max_colors={colors}:stats_mode=full[p];"
            "[b][p]paletteuse=dither=none:diff_mode=rectangle"
        )
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", f"{tmp}/%04d.png",
             "-vf", graph, "-loop", "0", str(path)],
            check=True,
        )  # fmt: skip
    print(f"wrote {path} ({len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB)")


def hero(clips: dict[str, Clip], path: Path, seconds: float = 5.0, step: int = 2) -> None:
    """Three behaviours side by side with the held-out model's live output under each."""
    gap, head, foot = 12, 30, 70
    canvas_w = 3 * TILE + 4 * gap
    count = int(seconds * 30 / step)
    frames = []
    starts = {c: max((clip.n - count * step) // 2, 0) for c, clip in clips.items()}
    crops = {c: clip.frames(starts[c] + step * np.arange(count)) for c, clip in clips.items()}
    for i in range(count):
        canvas = Image.new("RGB", (canvas_w, head + TILE + foot), SURFACE)
        draw = ImageDraw.Draw(canvas)
        for k, (name, clip) in enumerate(clips.items()):
            x = gap + k * (TILE + gap)
            draw.rounded_rectangle((x, 9, x + 10, 19), radius=3, fill=COLORS[name])
            draw.text((x + 17, 7), f"annotated: {name}", font=BOLD, fill=INK)
            canvas.paste(tile(crops[name][i]), (x, head))
            frame_index = min(starts[name] + step * i, clip.n - 1)
            draw_bars(draw, x, head + TILE + 8, TILE, clip.probs[frame_index])
        frames.append(canvas)
    write_gif(frames, path, fps=30 // step)


def pipeline(clip: Clip, data_root: Path, path: Path, seconds: float = 5.0, step: int = 2) -> None:
    """Camera frame with the tracked window, the crop it yields, and its motion-colour view."""
    row = clip.row
    width, height, crop = int(row["width"]), int(row["height"]), int(row["crop_px"])
    count = int(seconds * 30 / step)
    # Show the stretch where the window travels furthest.
    travel = np.linalg.norm(np.diff(clip.track, axis=0), axis=1)
    span = count * step
    start = int(np.argmax(np.convolve(travel, np.ones(span), mode="valid"))) if clip.n > span else 0
    wanted = start + step * np.arange(count)
    raw = {}
    for i, frame in enumerate(iter_gray_frames(data_root / row["rel_path"])):
        if i in set(wanted.tolist()):
            raw[i] = frame
        if i >= wanted[-1]:
            break
    crops, colour = clip.frames(wanted), clip.motion_colour(wanted)
    raw_w = round(TILE * width / height)
    scale = TILE / height
    gap, head = 12, 28
    titles = ["camera frame and tracked window", "fly-centred crop", "three frames as R, G, B"]
    frames = []
    for k, index in enumerate(wanted):
        canvas = Image.new("RGB", (raw_w + 2 * TILE + 4 * gap, head + TILE + gap), SURFACE)
        draw = ImageDraw.Draw(canvas)
        full = Image.fromarray(raw[int(index)]).convert("RGB")
        canvas.paste(full.resize((raw_w, TILE), Image.Resampling.LANCZOS), (gap, head))
        x0, y0 = crop_origin(clip.track[int(index)], crop, width, height)
        box = [
            gap + x0 * scale,
            head + y0 * scale,
            gap + (x0 + crop) * scale,
            head + (y0 + crop) * scale,
        ]
        draw.rectangle(box, outline=COLORS["feeding"], width=2)
        canvas.paste(tile(crops[k]), (raw_w + 2 * gap, head))
        canvas.paste(tile(colour[k]), (raw_w + TILE + 3 * gap, head))
        for title, x in zip(titles, [gap, raw_w + 2 * gap, raw_w + TILE + 3 * gap], strict=True):
            draw.text((x, 8), title, font=BOLD, fill=INK)
        frames.append(canvas)
    write_gif(frames, path, fps=30 // step)


def timeline(clip: Clip, path: Path, seconds: float = 12.0, step: int = 3) -> None:
    """The end of a long clip with the model's output drawn as a running timeline."""
    count = int(seconds * 30 / step)
    start = max(clip.n - count * step, 0)
    wanted = start + step * np.arange(count)
    crops, colour = clip.frames(wanted), clip.motion_colour(wanted)
    gap, head, plot_h, legend_h = 12, 28, 86, 26
    width = 2 * TILE + 3 * gap
    plot_x0, plot_x1 = gap, width - gap
    plot_y0 = head + TILE + gap + legend_h
    probs = clip.probs[wanted]
    xs = np.linspace(plot_x0, plot_x1, count)

    def y_of(p: float) -> float:
        return plot_y0 + plot_h - 4 - p * (plot_h - 8)

    frames = []
    for i in range(count):
        canvas = Image.new("RGB", (width, plot_y0 + plot_h + 22), SURFACE)
        draw = ImageDraw.Draw(canvas)
        draw.text((gap, 8), f"annotated: {clip.row['label']}", font=BOLD, fill=INK)
        draw.text((TILE + 2 * gap, 8), "what the network sees", font=BOLD, fill=INK)
        canvas.paste(tile(crops[i]), (gap, head))
        canvas.paste(tile(colour[i]), (TILE + 2 * gap, head))
        # Legend doubles as a live readout.
        best = int(np.argmax(probs[i]))
        for k, name in enumerate(CLASSES):
            x = gap + k * 150
            y = head + TILE + gap + 4
            draw.rounded_rectangle((x, y + 2, x + 10, y + 12), radius=3, fill=COLORS[name])
            text = f"{name} {probs[i, k]:.2f}"
            draw.text(
                (x + 16, y),
                text,
                font=BOLD if k == best else BODY,
                fill=INK if k == best else MUTED,
            )
        draw.rounded_rectangle((plot_x0, plot_y0, plot_x1, plot_y0 + plot_h), radius=4, fill=PANEL)
        for k, name in enumerate(CLASSES):
            points = [(float(xs[j]), y_of(float(probs[j, k]))) for j in range(i + 1)]
            if len(points) > 1:
                draw.line(points, fill=COLORS[name], width=3, joint="curve")
            cx, cy = float(xs[i]), points[-1][1]
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=COLORS[name], outline=PANEL)
        draw.text((plot_x0, plot_y0 + plot_h + 5), f"{start / 30:.0f} s", font=SMALL, fill=MUTED)
        draw.text(
            (plot_x1, plot_y0 + plot_h + 5),
            f"{clip.n / 30:.0f} s",
            font=SMALL,
            fill=MUTED,
            anchor="ra",
        )
        draw.text(
            (width // 2, plot_y0 + plot_h + 5),
            "predicted probability over time",
            font=SMALL,
            fill=MUTED,
            anchor="ma",
        )
        frames.append(canvas)
    # Thin coloured lines are a tiny share of the pixels; a larger palette keeps their hue.
    write_gif(frames, path, fps=30 // step, colors=192)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--run", type=Path, default=Path("runs/tsm_triplet"))
    parser.add_argument("--out", type=Path, default=Path("docs/media"))
    parser.add_argument("--data-root", default=os.environ.get("FLYACT_DATA_ROOT"))
    args = parser.parse_args()

    device = pick_device()
    manifest = read_manifest(args.manifest)

    def load(clip_id: str) -> Clip:
        return Clip(clip_id, manifest, args.cache, args.run, device)

    hero({name: load(clip_id) for name, clip_id in HERO.items()}, args.out / "behaviours.gif")
    timeline(load(TIMELINE), args.out / "timeline.gif")
    if args.data_root:
        pipeline(load(PIPELINE), Path(args.data_root).expanduser(), args.out / "pipeline.gif")
    else:
        print("FLYACT_DATA_ROOT not set: skipping pipeline.gif, which needs the raw recording")


if __name__ == "__main__":
    main()
