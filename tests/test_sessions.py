import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from flyact import CLASSES
from flyact.sessions import (
    assign_folds,
    cluster_sessions,
    cut_batches,
    frame_size_mismatch,
    merge_by_cut_batch,
)


def toy_manifest():
    rng = np.random.default_rng(0)
    rows = []
    for group in range(9):
        for _ in range(int(rng.integers(20, 120))):
            rows.append({"group": group, "label": CLASSES[int(rng.integers(0, 3))]})
    df = pd.DataFrame(rows)
    df["label"] = pd.Categorical(df["label"], categories=list(CLASSES))
    return df


def test_folds_never_split_a_group():
    df = toy_manifest()
    df["fold"] = assign_folds(df, n_splits=3, n_seeds=20)
    assert df.groupby("group")["fold"].nunique().max() == 1
    assert sorted(df["fold"].unique()) == [0, 1, 2]


def test_folds_are_deterministic():
    df = toy_manifest()
    assert (assign_folds(df, 3, 20) == assign_folds(df, 3, 20)).all()


def test_cluster_sessions_orders_by_size():
    points = np.concatenate([np.zeros((5, 3)), np.ones((12, 3)), np.full((7, 3), 5.0)])
    sessions = cluster_sessions(pdist(points), threshold=0.5)
    assert list(np.bincount(sessions)) == [12, 7, 5]


def test_cut_batches_split_on_time_gaps_and_ignore_unnamed_clips():
    df = pd.DataFrame(
        {
            "bout_idx": pd.array([0, 1, 2, pd.NA, 0], dtype="Int64"),
            "cut_time": [1000, 1010, 5000, 1005, 5020],
        }
    )
    assert cut_batches(df).tolist() == [0, 0, 1, pd.NA, 1]


def test_merge_by_cut_batch_joins_sessions_that_share_a_batch():
    df = pd.DataFrame(
        {
            "session": [0, 0, 1, 1, 2, 3, 3],
            "cut_batch": pd.array([pd.NA, 7, 7, pd.NA, pd.NA, 8, 8], dtype="Int64"),
        }
    )
    group = merge_by_cut_batch(df)
    assert group[0] == group[1] == group[2] == group[3]  # sessions 0 and 1 share batch 7
    assert len({group[0], group[4], group[5]}) == 3  # the others stay apart


def test_stray_clips_join_their_nearest_session():
    points = np.concatenate([np.zeros((8, 3)), np.ones((8, 3)), [[1.4, 1.4, 1.4]]])
    sessions = cluster_sessions(pdist(points), threshold=0.5)
    assert list(np.bincount(sessions)) == [9, 8]  # the stray point joined the cluster at 1
    assert sessions[-1] == sessions[8]


def test_clips_of_different_frame_size_never_share_a_session():
    df = pd.DataFrame({"width": [1100] * 6 + [1332] * 6, "height": [800] * 6 + [920] * 6})
    identical_walls = np.zeros(len(df) * (len(df) - 1) // 2)
    sessions = cluster_sessions(identical_walls + frame_size_mismatch(df), threshold=0.5)
    assert len(set(sessions[:6])) == 1 and len(set(sessions[6:])) == 1
    assert sessions[0] != sessions[6]
