"""Render the clips a run got wrong, most confident mistakes first.

    python scripts/contact_sheet.py runs/tsm_triplet --out errors.jpg

Each row is one clip: six evenly spaced frames of the cached crop, annotated with the
annotated class, the predicted class and the predicted probabilities. Looking at these
is the fastest way to tell model errors from label noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from flyact import CLASSES
from flyact.video import read_cached_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--out", type=Path, default=Path("errors.jpg"))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--label", choices=CLASSES, help="only clips annotated as this class")
    args = parser.parse_args()

    preds = pd.read_csv(args.run / "oof.csv", dtype={"clip_id": str})
    wrong = preds[preds["pred"] != preds["label"]].copy()
    if args.label:
        wrong = wrong[wrong["label"] == args.label]
    wrong["confidence"] = wrong[[f"p_{c}" for c in CLASSES]].max(axis=1)
    wrong = wrong.sort_values("confidence", ascending=False).head(args.limit)

    rows = []
    for clip in wrong.itertuples():
        n = int(clip.n_frames)
        frames = read_cached_frames(args.cache / f"{clip.clip_id}.mp4", np.linspace(0, n - 1, 6))
        strip = Image.fromarray(np.hstack(list(frames))).convert("RGB")
        probs = " ".join(f"{c[0]}={getattr(clip, f'p_{c}'):.2f}" for c in CLASSES)
        text = (
            f"{clip.clip_id}  annotated {clip.label} -> predicted {clip.pred}  [{probs}]  "
            f"recording {clip.group + 1}, {n} frames"
        )
        ImageDraw.Draw(strip).text((4, 4), text, fill=(255, 70, 70))
        rows.append(np.asarray(strip))
    if not rows:
        print("no errors to show")
        return
    Image.fromarray(np.vstack(rows)).save(args.out, quality=85)
    total = int((preds["pred"] != preds["label"]).sum())
    print(f"{total} errors in total; wrote {len(rows)} to {args.out}")


if __name__ == "__main__":
    main()
