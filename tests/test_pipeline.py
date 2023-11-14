"""End-to-end run of every CLI command on a tiny synthetic dataset."""

import json

import av
import numpy as np
import pandas as pd
import pytest

from flyact import CLASSES
from flyact.cli import main

W, H, N_FRAMES = 440, 320, 40


def write_clip(path, rng, wall, behaviour):
    """A dark elliptical "fly" in a grey chamber whose wall texture identifies the recording."""
    yy, xx = np.mgrid[:H, :W]
    cx, cy = rng.integers(140, 300), rng.integers(110, 210)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        for t in range(N_FRAMES):
            frame = np.full((H, W), 125, dtype=np.uint8)
            frame[:40], frame[-40:], frame[:, :50], frame[:, -50:] = (
                wall[:40],
                wall[-40:],
                wall[:, :50],
                wall[:, -50:],
            )
            wobble = {"feeding": 0, "grooming": 6 * np.sin(t * 2.0), "pumping": 6 * np.sin(t * 0.4)}
            body = ((xx - cx - wobble[behaviour]) / 45) ** 2 + ((yy - cy) / 18) ** 2 <= 1
            frame[body] = 70
            video_frame = av.VideoFrame.from_ndarray(frame, format="gray").reformat(
                format="yuv420p"
            )
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("flyact")
    data = root / "raw"
    rng = np.random.default_rng(0)
    for recording in range(2):
        wall = rng.integers(40, 220, (H, W)).astype(np.uint8)  # very different per recording
        for behaviour in CLASSES:
            folder = data / f"rec{recording}" / behaviour.capitalize()
            folder.mkdir(parents=True)
            for i in range(4):
                write_clip(folder / f"v_{behaviour}_g01_c{i}.avi", rng, wall, behaviour)
    # A renamed duplicate, as the real data has by the thousand.
    first = data / "rec0" / "Feeding" / "v_feeding_g01_c0.avi"
    (data / "copy").mkdir()
    (data / "copy" / "v_Feeding_g01_c99.avi").write_bytes(first.read_bytes())
    return root


def test_full_pipeline(workspace, monkeypatch, capsys):
    monkeypatch.chdir(workspace)
    paths = ["--manifest", "manifest.csv", "--cache", "cache"]

    main(["index", "--data-root", "raw", "--workers", "1", *paths])
    main(["preprocess", "--data-root", "raw", "--workers", "2", *paths])
    main(["sessions", "--folds", "2", "--workers", "1", *paths])

    df = pd.read_csv("manifest.csv")
    assert len(df) == 24 and df["n_copies"].max() == 2  # the duplicate was folded in
    assert df["usable"].all() and (df["n_frames"] == N_FRAMES).all()
    assert df["detected_frac"].min() > 0.9
    # The two synthetic recordings are recovered exactly, and folds keep them apart.
    truth = df["rel_path"].str.extract(r"rec(\d)")[0].fillna("0")
    assert df.groupby(truth)["group"].nunique().max() == 1 and df["group"].nunique() == 2
    assert df.groupby("group")["fold"].nunique().max() == 1

    (workspace / "tiny.yaml").write_text(
        "sampling: {num_snippets: 4, stride: 2, delta: 1, image_size: 64, eval_windows: 2}\n"
        "model: {pretrained: false}\n"
        "train: {epochs: 1, batch_size: 4}\n"
    )
    train = ["train", "--config", "tiny.yaml", "--workers", "0", *paths]
    main([*train, "--out", "runs/tiny"])
    summary = json.loads((workspace / "runs/tiny/summary.json").read_text())
    oof = pd.read_csv("runs/tiny/oof.csv")
    assert summary["n_clips"] == 24 and sorted(oof["clip_id"]) == sorted(df["clip_id"])
    assert np.allclose(oof[[f"p_{c}" for c in CLASSES]].sum(axis=1), 1, atol=1e-4)

    main([*train, "--out", "runs/tiny", "--final"])
    capsys.readouterr()
    clip = str(workspace / "raw/rec1/Grooming/v_grooming_g01_c2.avi")
    main(["predict", clip, "--checkpoint", "runs/tiny/final.pt", "--csv", "windows.csv"])
    assert capsys.readouterr().out.split(":")[1].split()[0] in CLASSES
    windows = pd.read_csv("windows.csv")
    assert {"start_s", "end_s", *[f"p_{c}" for c in CLASSES]} <= set(windows.columns)
