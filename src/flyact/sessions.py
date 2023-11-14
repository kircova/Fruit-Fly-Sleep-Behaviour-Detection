"""Recover which recording each clip was cut from, and build leak-free folds.

Every clip is a bout cut out of one of roughly fifteen many-hour recordings, one fly per
recording. Clips from the same recording share the fly, the chamber, the lighting and often
the exact pose, so a model evaluated on a random split is rewarded for recognising the
*recording* rather than the *behaviour*. The file names no longer say which recording a
clip came from, so we reconstruct it from the pixels.

What identifies a recording is the chamber itself: it is 3D printed, its walls have a
ragged outline, and it sits a few pixels differently in every experiment. The signature is
therefore the edge map of the outer band of the frame, where the walls and food port are.
Two things make it robust:

* edges are normalised per clip, so the lights-on and lights-off phases of one experiment
  look alike, and
* distances are trimmed means, so a fly standing in front of a wall only affects pixels
  that are then ignored.

The clustering can be checked against information it never saw: clips that still carry
their original names were cut from the recordings in batches, one recording at a time,
and the batch is visible in the file timestamps. ``validate_against_cut_batches`` reports
how well the two agree, and ``merge_by_cut_batch`` fuses the rare sessions that the
metadata says belong together, so that the final groups err on the side of being too
coarse (safe) rather than too fine (leaky).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import StratifiedGroupKFold

# Width of the outer band (fraction of height / width) that holds walls and food port.
BAND = (0.17, 0.145)
TRIM = 0.7  # use the 70% best-matching pixels of every comparison
# Same-recording and different-recording distances are bimodal with an empty gap between
# roughly 0.025 and 0.029 (see `flyact sessions` output); the threshold sits in the gap.
DEFAULT_THRESHOLD = 0.027
CUT_BATCH_GAP_S = 600


def wall_signature(thumbnail: np.ndarray) -> np.ndarray:
    """Normalised edge strength of the outer band of a background thumbnail, as uint8."""
    img = ndi.gaussian_filter(thumbnail.astype(np.float32), 1.0)
    edges = np.hypot(ndi.sobel(img, axis=0), ndi.sobel(img, axis=1))
    edges = np.clip(edges / (np.percentile(edges, 99) + 1e-6), 0, 1.5)
    h, w = edges.shape
    bh, bw = int(BAND[0] * h), int(BAND[1] * w)
    band = np.ones((h, w), bool)
    band[bh : h - bh, bw : w - bw] = False
    return np.round(edges[band] * 170).astype(np.uint8)


_SIG: np.ndarray | None = None


def _init_worker(sig: np.ndarray) -> None:
    global _SIG
    _SIG = sig


def _row(i: int) -> np.ndarray:
    assert _SIG is not None
    diff = np.abs(_SIG[i + 1 :].astype(np.int16) - _SIG[i].astype(np.int16))
    k = int(TRIM * diff.shape[1])
    return np.partition(diff, k, axis=1)[:, :k].mean(axis=1, dtype=np.float32) / 170.0


def pairwise_distances(signatures: np.ndarray, workers: int = 8) -> np.ndarray:
    """Condensed matrix of trimmed mean absolute differences between signatures."""
    n = len(signatures)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(signatures,)) as pool:
        rows = list(pool.map(_row, range(n - 1), chunksize=16))
    return np.concatenate(rows)


def frame_size_mismatch(df: pd.DataFrame) -> np.ndarray:
    """Condensed 0/1 matrix: 1 where two clips have different frame sizes.

    Added to the signature distances as a cannot-link constraint: one camera setting per
    recording, so clips of different size were never cut from the same one.
    """
    size_id = df.groupby(["width", "height"]).ngroup().to_numpy()
    return pdist(size_id[:, None], metric="hamming")


def cluster_sessions(
    condensed: np.ndarray, threshold: float = DEFAULT_THRESHOLD, min_size: int = 5
) -> np.ndarray:
    """Average-linkage clustering; returns session numbers ordered by size (0 = largest).

    A handful of clips end up alone because something unusual covers the walls (the fly
    walking along the rim, for instance). A recording with fewer than ``min_size`` clips is
    not plausible, so those clips join the session of their nearest neighbour instead of
    forming "recordings" of their own that could land in a different fold.
    """
    raw = fcluster(linkage(condensed, method="average"), threshold, criterion="distance")
    sizes = pd.Series(raw).value_counts()
    small = np.isin(raw, sizes.index[sizes < min_size])
    if small.any() and not small.all():
        dist = squareform(condensed)
        nearest = dist[np.ix_(small, ~small)].argmin(axis=1)
        raw[small] = raw[~small][nearest]
    order = pd.Series(raw).value_counts().index
    rank = {label: i for i, label in enumerate(order)}
    return np.array([rank[label] for label in raw])


def cut_batches(df: pd.DataFrame) -> pd.Series:
    """Batch number for clips that kept their original bout name, NA for the rest."""
    named = df[df["bout_idx"].notna()].sort_values("cut_time")
    batch = (named["cut_time"].diff() > CUT_BATCH_GAP_S).cumsum()
    return batch.reindex(df.index).astype("Int64")


def validate_against_cut_batches(df: pd.DataFrame, condensed: np.ndarray) -> dict:
    """Compare the pixel-based sessions with the batches recorded in file timestamps."""
    batch = df["cut_batch"]
    named = batch.notna().to_numpy()
    if named.sum() < 2:  # no clip kept its original name: nothing to check against
        return {"named_clips": int(named.sum())}
    dist = squareform(condensed)
    b = batch.to_numpy(dtype=float, na_value=np.nan)
    same = (b[:, None] == b[None, :]) & named[:, None] & named[None, :]
    upper = np.triu(np.ones_like(same), 1)
    per_batch = df[named].groupby("cut_batch")["session"].nunique()
    return {
        "named_clips": int(named.sum()),
        "cut_batches": int(per_batch.size),
        "batches_in_one_session": int((per_batch == 1).sum()),
        "same_batch_distance_p50": float(np.median(dist[same & upper])),
        "same_batch_distance_p90": float(np.percentile(dist[same & upper], 90)),
        "all_pairs_distance_p50": float(np.median(condensed)),
    }


def merge_by_cut_batch(df: pd.DataFrame) -> np.ndarray:
    """Group id per clip: sessions joined whenever one cutting batch spans both."""
    parent = {s: s for s in df["session"].unique()}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for _, sessions in df[df["cut_batch"].notna()].groupby("cut_batch")["session"]:
        roots = sorted({find(s) for s in sessions})
        for other in roots[1:]:
            parent[other] = roots[0]
    roots = df["session"].map(find)
    order = roots.value_counts().index
    rank = {root: i for i, root in enumerate(order)}
    return roots.map(rank).to_numpy()


def assign_folds(df: pd.DataFrame, n_splits: int = 5, n_seeds: int = 200) -> np.ndarray:
    """Group-disjoint folds whose class mix is as even as the groups allow.

    ``StratifiedGroupKFold`` is greedy and its result depends on the shuffle, so we draw
    many partitions and keep the one whose per-fold class counts are closest to uniform.
    The search only looks at labels and group ids, never at model outputs.
    """
    y = df["label"].cat.codes.to_numpy()
    groups = df["group"].to_numpy()
    ideal = np.bincount(y) / n_splits
    best, best_cost = None, np.inf
    for seed in range(n_seeds):
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        fold = np.empty(len(df), dtype=int)
        for k, (_, test) in enumerate(splitter.split(y, y, groups)):
            fold[test] = k
        counts = np.stack(
            [np.bincount(y[fold == k], minlength=len(ideal)) for k in range(n_splits)]
        )
        cost = float((((counts - ideal) / ideal) ** 2).sum())
        if cost < best_cost:
            best, best_cost = fold, cost
    assert best is not None
    # Relabel so that fold 0 holds the most clips; purely cosmetic but stable.
    order = pd.Series(best).value_counts().index
    rank = {f: i for i, f in enumerate(order)}
    return np.array([rank[f] for f in best])
