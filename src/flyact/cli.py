"""Command line entry point: ``flyact <command> --help``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .index import read_manifest

DEFAULT_MANIFEST = Path("data/manifest.csv")
DEFAULT_CACHE = Path("cache")


def _data_root(arg: str | None) -> Path:
    root = arg or os.environ.get("FLYACT_DATA_ROOT")
    if not root:
        raise SystemExit("pass --data-root or set FLYACT_DATA_ROOT to the folder holding the clips")
    return Path(root).expanduser()


def cmd_index(args: argparse.Namespace) -> None:
    from .index import build_index

    df = build_index(_data_root(args.data_root), workers=args.workers)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.manifest, index=False)
    print(f"{int(df['n_copies'].sum())} files -> {len(df)} unique labelled clips")
    print(df["label"].value_counts().to_string())


def cmd_preprocess(args: argparse.Namespace) -> None:
    from .preprocess import build_cache

    df = read_manifest(args.manifest)
    stats = build_cache(df, _data_root(args.data_root), args.cache, workers=args.workers)
    stat_cols = [c for c in stats.columns if c != "clip_id"]
    df = df.drop(columns=[c for c in stat_cols if c in df]).merge(stats, on="clip_id", how="left")
    df["usable"] = df["usable"].fillna(False).astype(bool)
    df.to_csv(args.manifest, index=False)
    ok = df[df["usable"]]
    print(f"{len(ok)} usable clips, {len(df) - len(ok)} dropped (undecodable or too short)")
    hours = ok["n_frames"].sum() / ok["fps"].mean() / 3600
    print(f"{int(ok['n_frames'].sum())} frames = {hours:.2f} h")
    print(f"fly found in {ok['detected_frac'].mean():.1%} of sampled frames")


def cmd_sessions(args: argparse.Namespace) -> None:
    from . import sessions as S
    from .preprocess import load_thumbnails

    df = read_manifest(args.manifest)
    use = df["usable"].to_numpy()
    sub = df[use].reset_index(drop=True)
    thumbs = load_thumbnails(args.cache, sub["clip_id"].tolist())
    signatures = np.stack([S.wall_signature(t) for t in thumbs])
    condensed = S.pairwise_distances(signatures, workers=args.workers)
    np.save(args.cache / "wall_distances.npy", condensed)  # kept for the report figure
    condensed = condensed + S.frame_size_mismatch(sub)

    threshold = args.threshold or S.DEFAULT_THRESHOLD
    hist, edges = np.histogram(condensed, bins=40, range=(0, 0.08))
    print("pairwise wall-signature distances (the gap separates recordings):")
    for count, edge in zip(hist, edges, strict=False):
        mark = " <- threshold" if edge <= threshold < edge + 0.002 else ""
        print(f"  {edge:.3f} {'#' * int(50 * count / hist.max())}{mark}")

    sub["session"] = S.cluster_sessions(condensed, threshold)
    sub["cut_batch"] = S.cut_batches(sub)
    report = S.validate_against_cut_batches(sub, condensed)
    print("check against cutting batches (never used for clustering):")
    for key, value in report.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    sub["group"] = S.merge_by_cut_batch(sub)
    sub["fold"] = S.assign_folds(sub, n_splits=args.folds)

    new_cols = ["session", "cut_batch", "group", "fold"]
    df = df.drop(columns=[c for c in new_cols if c in df])
    df = df.merge(sub[["clip_id", *new_cols]], on="clip_id", how="left")
    for col in new_cols:
        df[col] = df[col].astype("Int64")
    df.to_csv(args.manifest, index=False)

    print(f"{sub['session'].nunique()} sessions -> {sub['group'].nunique()} groups")
    print("clips per group and class:")
    table = pd.crosstab(sub["group"], sub["label"])
    table["fold"] = sub.groupby("group")["fold"].first()
    print(table.to_string())
    print("clips per fold and class:")
    print(pd.crosstab(sub["fold"], sub["label"], margins=True).to_string())


def cmd_train(args: argparse.Namespace) -> None:
    from .train import run_from_args

    run_from_args(args)


def cmd_baselines(args: argparse.Namespace) -> None:
    from .baselines import run_from_args

    run_from_args(args)


def cmd_report(args: argparse.Namespace) -> None:
    from .report import run_from_args

    run_from_args(args)


def cmd_predict(args: argparse.Namespace) -> None:
    from .predict import run_from_args

    run_from_args(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flyact", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, data_root: bool = False) -> None:
        p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        p.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 4))
        if data_root:
            p.add_argument("--data-root", help="folder with the raw clips (or $FLYACT_DATA_ROOT)")

    p = sub.add_parser("index", help="scan the raw data, de-duplicate, write the manifest")
    common(p, data_root=True)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("preprocess", help="track the fly and build the crop cache")
    common(p, data_root=True)
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("sessions", help="recover recording sessions and assign folds")
    common(p)
    p.add_argument("--threshold", type=float, help="wall-signature distance separating recordings")
    p.add_argument("--folds", type=int, default=5)
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser("train", help="train and evaluate a model with grouped cross-validation")
    common(p)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, help="run directory (default: runs/<config name>)")
    p.add_argument("--folds", type=int, nargs="*", help="only run these test folds")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    p.add_argument("--final", action="store_true", help="train one model on all the data")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("baselines", help="non-neural baselines under the same folds")
    common(p)
    p.add_argument("--out", type=Path, default=Path("runs/baselines"))
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("report", help="aggregate runs into tables and figures")
    common(p, data_root=True)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("docs"))
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("predict", help="classify a new video with a trained model")
    p.add_argument("video", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hop-seconds", type=float, default=1.0)
    p.add_argument("--csv", type=Path, help="write per-window probabilities here")
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
