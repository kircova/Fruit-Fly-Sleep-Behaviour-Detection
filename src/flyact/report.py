"""Turn finished runs into the tables and figures used in the README."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score

from . import CLASSES
from .index import read_manifest
from .sessions import DEFAULT_THRESHOLD

# Display name per run directory, in the order they appear in the results table.
RUNS = [
    ("baselines/majority", "Majority class"),
    ("baselines/position", "Fly position only"),
    ("baselines/motion", "Motion statistics"),
    ("baselines/position_motion", "Position + motion statistics"),
    ("static_2d", "ResNet-18 on single frames"),
    ("triplet_2d", "ResNet-18 on frame triplets"),
    ("static_tsm", "ResNet-18 + temporal shift on single frames"),
    ("tsm_triplet", "ResNet-18 + temporal shift on frame triplets"),
]
MAIN_RUN = "tsm_triplet"
LEAKAGE_RUN = "random_split"

# Palette: three categorical hues (validated for colour-vision deficiency as a set), one
# sequential hue, and neutral inks. Text is never drawn in a series colour.
CLASS_COLORS = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a"], strict=True))
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
BLUE = "#2a78d6"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {path}")


def load_runs(runs_dir: Path) -> dict[str, dict]:
    found = {}
    for name, _ in [*RUNS, (LEAKAGE_RUN, "")]:
        summary = runs_dir / name / "summary.json"
        if summary.exists():
            found[name] = json.loads(summary.read_text())
    return found


def results_table(summaries: dict[str, dict]) -> str:
    """Markdown table of every finished run, one row each."""
    lines = [
        "| Model | Accuracy | Balanced acc. | Macro-F1 | 95% CI (recordings) "
        "| F1 feeding | F1 grooming | F1 pumping |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, title in RUNS:
        s = summaries.get(name)
        if s is None:
            continue
        lo, hi = s["ci95_recordings"]["macro_f1"]
        f1 = [s["per_class"][c]["f1"] for c in CLASSES]
        bold = "**" if name == MAIN_RUN else ""
        lines.append(
            f"| {bold}{title}{bold} | {s['accuracy']:.3f} | {s['balanced_accuracy']:.3f} "
            f"| {bold}{s['macro_f1']:.3f}{bold} | {lo:.2f} to {hi:.2f} "
            f"| {f1[0]:.3f} | {f1[1]:.3f} | {f1[2]:.3f} |"
        )
    return "\n".join(lines)


def length_table(oof: pd.DataFrame, fps: float = 30.0) -> str:
    """Accuracy by clip duration: do short clips, which get fewer windows, fare worse?"""
    seconds = oof["n_frames"] / fps
    bucket = pd.cut(
        seconds, [0, 2, 5, 15, np.inf], labels=["< 2 s", "2 to 5 s", "5 to 15 s", "> 15 s"]
    )
    correct = oof["pred"] == oof["label"]
    lines = ["| Clip length | Clips | Accuracy |", "|---|---|---|"]
    for name, idx in correct.groupby(bucket, observed=True).groups.items():
        lines.append(f"| {name} | {len(idx)} | {correct.loc[idx].mean():.3f} |")
    return "\n".join(lines)


def motion_table(oof: pd.DataFrame, motion: pd.DataFrame) -> str:
    """Clips and accuracy per class at five levels of overall motion in the crop."""
    df = oof.merge(motion[["clip_id", "energy_mean"]], on="clip_id")
    df["correct"] = df["pred"] == df["label"]
    names = ["stillest fifth", "2nd", "3rd", "4th", "most motion"]
    df["level"] = pd.qcut(df["energy_mean"], 5, labels=names)
    lines = [
        "| Motion in the clip | " + " | ".join(f"{c}: clips (accuracy)" for c in CLASSES) + " |",
        "|---|" + "---|" * len(CLASSES),
    ]
    for level in names:
        cells = []
        for cls in CLASSES:
            part = df[(df["level"] == level) & (df["label"] == cls)]
            cells.append(f"{len(part)} ({part['correct'].mean():.0%})" if len(part) else "0")
        lines.append(f"| {level} | " + " | ".join(cells) + " |")
    auc = roc_auc_score(df["label"] == "pumping", -df["energy_mean"])
    lines.append(f"\nLow motion alone separates pumping from the rest with AUC {auc:.3f}.")
    return "\n".join(lines)


def fig_dataset(df: pd.DataFrame, path: Path) -> None:
    """Clips per recording, split by class: the imbalance that makes random splits unsafe."""
    table = pd.crosstab(df["group"], df["label"]).sort_values(by=list(CLASSES), ascending=False)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7.6, 0.27 * len(table) + 1.3))
    y = np.arange(len(table))
    left = np.zeros(len(table))
    for cls in CLASSES:
        values = table[cls].to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            height=0.62,
            color=CLASS_COLORS[cls],
            label=cls,
            edgecolor=SURFACE,
            linewidth=1.2,
        )
        left += values
    for yi, total in zip(y, left, strict=True):
        ax.text(total + 6, yi, f"{int(total)}", va="center", ha="left", color=INK_2, fontsize=9)
    ax.set_yticks(y, [f"recording {g + 1}" for g in table.index])
    ax.invert_yaxis()
    ax.set_xlabel("clips")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, left.max() * 1.08)
    ax.legend(loc="lower right", ncols=3, handlelength=1.0, columnspacing=1.4)
    ax.set_title("Two recordings hold more than half of the clips; grooming appears in three")
    _save(fig, path)


def fig_sessions(df: pd.DataFrame, distances: np.ndarray, threshold: float, path: Path) -> None:
    """Wall-signature distances: all pairs of clips versus pairs known to share a recording."""
    from scipy.spatial.distance import squareform

    batch = df["cut_batch"].to_numpy(dtype=float, na_value=np.nan)
    same = squareform((batch[:, None] == batch[None, :]).astype(np.uint8), checks=False) > 0
    bins = np.linspace(0, 0.09, 61)
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 3.9), sharex=True)
    panels = [
        (distances, MUTED, f"All {len(distances):,} pairs of clips"),
        (distances[same], BLUE, f"The {int(same.sum()):,} pairs cut from a recording in one batch"),
    ]
    for ax, (values, color, title) in zip(axes, panels, strict=True):
        ax.hist(values, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(threshold, color=INK, linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    axes[0].text(
        threshold + 0.001, axes[0].get_ylim()[1] * 0.82, "threshold", color=INK_2, fontsize=9
    )
    axes[1].set_xlabel("distance between chamber-wall signatures")
    fig.subplots_adjust(hspace=0.45)
    _save(fig, path)


def fig_results(summaries: dict[str, dict], path: Path) -> None:
    """Macro-F1 of every method with a recording-level bootstrap interval."""
    rows = [(title, summaries[name]) for name, title in RUNS if name in summaries]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(rows) + 1.2))
    y = np.arange(len(rows))
    for yi, (_, s) in zip(y, rows, strict=True):
        lo, hi = s["ci95_recordings"].get("macro_f1", (s["macro_f1"], s["macro_f1"]))
        ax.plot([lo, hi], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round", alpha=0.45)
        ax.plot(
            s["macro_f1"],
            yi,
            "o",
            color=BLUE,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ax.text(
            max(hi, s["macro_f1"]) + 0.015,
            yi,
            f"{s['macro_f1']:.3f}",
            va="center",
            color=INK,
            fontsize=9.5,
        )
    ax.set_yticks(y, [title for title, _ in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("macro-F1 on held-out recordings (line: 95% interval over recordings)")
    ax.set_title("What each ingredient buys, measured on flies the model never saw")
    _save(fig, path)


def fig_confusion(summary: dict, path: Path) -> None:
    """Row-normalised confusion matrix of the main model with raw counts."""
    counts = np.array(summary["confusion_matrix"])
    share = counts / counts.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ink = "#ffffff" if share[i, j] > 0.55 else INK
            ax.text(
                j,
                i - 0.08,
                f"{share[i, j]:.0%}",
                ha="center",
                va="center",
                color=ink,
                fontsize=12,
                fontweight="bold",
            )
            ax.text(
                j, i + 0.2, f"{counts[i, j]}", ha="center", va="center", color=ink, fontsize=8.5
            )
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("annotated")
    for spine in ax.spines.values():
        spine.set_visible(False)
    # White gaps between cells instead of drawn borders.
    ax.set_xticks(np.arange(-0.5, len(CLASSES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASSES)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Out-of-fold confusion (share of row, clips)")
    _save(fig, path)


def fig_per_recording(summary: dict, path: Path) -> None:
    """Accuracy on each held-out recording: the spread a single number hides."""
    rows = sorted(summary["per_group"].items(), key=lambda kv: -kv[1]["clips"])
    rows = [(int(g), r) for g, r in rows if r["clips"] >= 10]
    fig, ax = plt.subplots(figsize=(7.6, 0.27 * len(rows) + 1.3))
    y = np.arange(len(rows))
    acc = np.array([r["accuracy"] for _, r in rows])
    ax.hlines(y, 0, acc, color=BLUE, linewidth=2, alpha=0.45)
    ax.plot(acc, y, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for yi, (_, r) in zip(y, rows, strict=True):
        ax.text(
            1.03, yi, f"{r['accuracy']:.0%} of {r['clips']}", va="center", color=INK_2, fontsize=9
        )
    ax.set_yticks(y, [f"recording {g + 1}" for g, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.22)
    ax.set_xticks(np.arange(0, 1.01, 0.2), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("accuracy when this recording is held out")
    ax.set_title("Accuracy by held-out recording (recordings with at least 10 clips)")
    _save(fig, path)


def fig_examples(df: pd.DataFrame, cache_dir: Path, path: Path, per_class: int = 2) -> None:
    """A few network inputs per class: eight snippets of one window, motion shown as colour."""
    from .config import SamplingConfig
    from .data import to_input, window_indices
    from .video import read_cached_frames

    cfg = SamplingConfig(image_size=224)
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(
        len(CLASSES) * per_class,
        cfg.num_snippets,
        figsize=(cfg.num_snippets * 1.15, len(CLASSES) * per_class * 1.15 + 0.5),
    )
    row = 0
    for cls in CLASSES:
        pool = df[(df["label"] == cls) & (df["n_frames"] >= 2 * cfg.span)]
        for _, clip in pool.sample(per_class, random_state=int(rng.integers(1 << 30))).iterrows():
            n = int(clip["n_frames"])
            rows = window_indices(n, (n - cfg.span) // 2, cfg.stride, cfg)
            unique, inverse = np.unique(rows, return_inverse=True)
            frames = read_cached_frames(cache_dir / f"{clip['clip_id']}.mp4", unique)
            x = to_input(frames, inverse.reshape(rows.shape), cfg.image_size, augment=False)
            x = ((x - x.min()) / (x.max() - x.min())).permute(0, 2, 3, 1).numpy()
            for j in range(cfg.num_snippets):
                axes[row, j].imshow(x[j])
                axes[row, j].set_xticks([])
                axes[row, j].set_yticks([])
                for spine in axes[row, j].spines.values():
                    spine.set_visible(False)
            axes[row, 0].set_ylabel(cls, color=INK, fontsize=9)
            row += 1
    fig.suptitle(
        "One 2-second window per row. Each tile stacks three frames 67 ms apart as R, G, B:\n"
        "whatever is grey stood still, whatever has colour fringes moved.",
        x=0.01,
        ha="left",
        fontsize=9.5,
        color=INK,
    )
    fig.subplots_adjust(wspace=0.03, hspace=0.03, top=0.9, left=0.04, right=0.995)
    _save(fig, path)


def fig_method(df: pd.DataFrame, data_root: Path, cache_dir: Path, path: Path) -> None:
    """Raw frame with the tracked window, the crop, and the crop's three-frame input."""
    from matplotlib.patches import Rectangle

    from .tracking import crop_origin
    from .video import iter_gray_frames, read_cached_frames

    # A grooming fly away from the walls, so the window is not clamped to the frame edge.
    pool = df[
        (df["label"] == "grooming")
        & df["n_frames"].between(150, 600)
        & df["y_mean"].between(0.4, 0.6)
        & df["x_mean"].between(0.35, 0.65)
    ]
    clip = pool.sort_values("clip_id").iloc[0]
    t = int(clip["n_frames"]) // 2
    raw = next(f for i, f in enumerate(iter_gray_frames(data_root / clip["rel_path"])) if i == t)
    track = np.load(cache_dir / f"{clip['clip_id']}.npz")["track"]
    crop = int(clip["crop_px"])
    x0, y0 = crop_origin(track[t], crop, int(clip["width"]), int(clip["height"]))
    frames = read_cached_frames(cache_dir / f"{clip['clip_id']}.mp4", [t - 2, t, t + 2])
    rgb = np.stack(frames, axis=-1).astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), width_ratios=[1.375, 1, 1])
    axes[0].imshow(raw, cmap="gray", vmin=0, vmax=255)
    axes[0].add_patch(Rectangle((x0, y0), crop, crop, fill=False, edgecolor=BLUE, linewidth=2))
    axes[0].set_title("1. Raw frame with the tracked window")
    axes[1].imshow(frames[1], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("2. Fly-centred crop")
    axes[2].imshow(rgb)
    axes[2].set_title("3. Three frames as R, G, B")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(wspace=0.04)
    _save(fig, path)


def run_from_args(args) -> None:
    _style()
    df = read_manifest(args.manifest)
    df = df[df["usable"]].reset_index(drop=True)
    figures = args.out / "figures"
    summaries = load_runs(args.runs)

    fig_dataset(df, figures / "dataset.png")
    distances = args.cache / "wall_distances.npy"
    if distances.exists():
        fig_sessions(df, np.load(distances), DEFAULT_THRESHOLD, figures / "sessions.png")
    if (args.cache / f"{df['clip_id'].iloc[0]}.mp4").exists():
        fig_examples(df, args.cache, figures / "inputs.png")
        data_root = args.data_root or os.environ.get("FLYACT_DATA_ROOT")
        if data_root:
            fig_method(df, Path(data_root).expanduser(), args.cache, figures / "method.png")
    if summaries:
        fig_results(summaries, figures / "results.png")
        table = results_table(summaries)
        oof_file = args.runs / MAIN_RUN / "oof.csv"
        if oof_file.exists():
            oof = pd.read_csv(oof_file, dtype={"clip_id": str})
            table += "\n\n" + length_table(oof)
            motion_file = args.cache / "motion_features.csv"
            if motion_file.exists():
                motion = pd.read_csv(motion_file, dtype={"clip_id": str})
                table += "\n\n" + motion_table(oof, motion)
        (args.out / "results.md").write_text(table + "\n")
        print(table)
    if MAIN_RUN in summaries:
        fig_confusion(summaries[MAIN_RUN], figures / "confusion.png")
        fig_per_recording(summaries[MAIN_RUN], figures / "per_recording.png")
    if LEAKAGE_RUN in summaries and MAIN_RUN in summaries:
        a, b = summaries[LEAKAGE_RUN], summaries[MAIN_RUN]
        print(
            f"random split: accuracy {a['accuracy']:.3f}, macro-F1 {a['macro_f1']:.3f}  |  "
            f"held-out recordings: accuracy {b['accuracy']:.3f}, macro-F1 {b['macro_f1']:.3f}"
        )
