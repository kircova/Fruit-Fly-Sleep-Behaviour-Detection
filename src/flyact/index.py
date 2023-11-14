"""Turn a messy folder tree of clips into a de-duplicated manifest.

The source data went through several rounds of curation, each saved as a full copy
(``Action_v3`` ... ``Action_v6``, ``400-Feeding``, zip exports, ...) and each renaming the
files, so neither folder nor file name identifies a clip. The only stable identity is the
content hash, which is what this module keys everything on.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from . import CLASSES

# Bout clips are AVI. The MP4s that sit next to them are compilations of many bouts made
# for presentations; indexing them would put the same frames in the dataset twice.
CLIP_SUFFIXES = (".avi",)

# Class keywords, tested against the file name first and then against parent folders.
# "prob" covers the original "prob_<idx>_<start>_<end>" names of proboscis-pumping bouts.
_LABEL_PATTERNS = {
    "feeding": re.compile(r"feed", re.I),
    "grooming": re.compile(r"groom", re.I),
    "pumping": re.compile(r"pump|(?:^|[_/])prob", re.I),
}

# Names assigned by the script that cut bouts out of the long recordings:
#   <behaviour>_<bout index>_<first frame>_<last frame>.avi
_BOUT_NAME = re.compile(r"^([A-Za-z]+)_(\d+)_(\d+)_(\d+)$")


def file_digest(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_label(rel_path: Path) -> str | None:
    """Class of a clip from its path, or None if the path names no class."""
    for text in (rel_path.name, *reversed(rel_path.parts[:-1])):
        hits = [label for label, pat in _LABEL_PATTERNS.items() if pat.search(text)]
        if len(hits) == 1:
            return hits[0]
    return None


def parse_bout_name(stem: str) -> tuple[int, int, int] | None:
    """``(bout index, first frame, last frame)`` if ``stem`` is an original bout name."""
    m = _BOUT_NAME.match(stem)
    if not m:
        return None
    idx, start, stop = (int(g) for g in m.groups()[1:])
    return (idx, start, stop) if stop > start else None


def build_index(
    data_root: Path, workers: int = 8, suffixes: tuple[str, ...] = CLIP_SUFFIXES
) -> pd.DataFrame:
    """Scan ``data_root`` and return one row per unique clip."""
    data_root = Path(data_root)
    files = sorted(p for p in data_root.rglob("*") if p.suffix.lower() in suffixes and p.is_file())
    with ProcessPoolExecutor(workers) as pool:
        digests = list(pool.map(file_digest, files, chunksize=32))

    copies: dict[str, list[Path]] = defaultdict(list)
    for path, digest in zip(files, digests, strict=True):
        copies[digest].append(path.relative_to(data_root))

    rows = []
    for digest, paths in copies.items():
        labels = {infer_label(p) for p in paths} - {None}
        if len(labels) != 1:
            # Unlabelled footage (e.g. demo videos) or a clip filed under two classes.
            continue
        bout = next((b for b in (parse_bout_name(p.stem) for p in paths) if b), None)
        rows.append(
            {
                "clip_id": digest[:12],
                "label": labels.pop(),
                "rel_path": paths[0].as_posix(),
                "n_copies": len(paths),
                "bout_idx": bout[0] if bout else pd.NA,
                "src_start": bout[1] if bout else pd.NA,
                "src_stop": bout[2] if bout else pd.NA,
                # Copies keep the timestamp of the moment the bout was cut from its source
                # recording; the earliest one is the original. Used only to cross-check the
                # session clustering (see sessions.py).
                "cut_time": int(min((data_root / p).stat().st_mtime for p in paths)),
            }
        )
    df = pd.DataFrame(rows)
    if df["clip_id"].duplicated().any():
        raise RuntimeError("12-character clip ids collide; lengthen the prefix")
    df["label"] = pd.Categorical(df["label"], categories=list(CLASSES))
    for col in ("bout_idx", "src_start", "src_stop"):
        df[col] = df[col].astype("Int64")
    return df.sort_values(["label", "rel_path"]).reset_index(drop=True)


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"clip_id": str})
    df["label"] = pd.Categorical(df["label"], categories=list(CLASSES))
    for col in ("bout_idx", "src_start", "src_stop", "cut_batch", "session", "group", "fold"):
        if col in df:
            df[col] = df[col].astype("Int64")
    return df
