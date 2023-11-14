import numpy as np
import pandas as pd

from flyact import CLASSES
from flyact.metrics import paired_difference, predictions_frame, summarize


def test_summarize_on_a_hand_checked_example():
    labels = ["feeding"] * 4 + ["grooming"] * 4 + ["pumping"] * 2
    clips = pd.DataFrame(
        {
            "clip_id": [f"c{i}" for i in range(10)],
            "label": pd.Categorical(labels, categories=list(CLASSES)),
            "group": [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
            "fold": [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
            "n_frames": 100,
        }
    )
    predicted = [0, 0, 0, 2, 1, 1, 1, 1, 2, 0]  # two mistakes
    preds = predictions_frame(clips, np.eye(3)[predicted])
    s = summarize(preds)

    assert s["accuracy"] == 0.8
    assert s["balanced_accuracy"] == round((3 / 4 + 4 / 4 + 1 / 2) / 3, 4)
    assert s["confusion_matrix"] == [[3, 0, 1], [0, 4, 0], [1, 0, 1]]
    assert s["per_class"]["grooming"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 4}
    assert s["per_group"][0] == {"clips": 5, "accuracy": 1.0, "fold": 0}
    assert s["per_group"][1]["accuracy"] == 0.6
    lo, hi = s["ci95_clips"]["accuracy"]
    assert lo <= 0.8 <= hi


def make_preds(rng, n_groups, accuracy):
    rows = []
    for group in range(n_groups):
        for i in range(60):
            label = CLASSES[i % 3]
            wrong = CLASSES[(i + 1) % 3]
            rows.append(
                (f"g{group}c{i}", label, group, label if rng.random() < accuracy else wrong)
            )
    df = pd.DataFrame(rows, columns=["clip_id", "label", "group", "pred"])
    return df


def test_paired_difference_separates_real_gaps_from_ties():
    rng = np.random.default_rng(0)
    strong, weak = make_preds(rng, 12, 0.95), make_preds(rng, 12, 0.75)
    point, lo, hi = paired_difference(strong, weak)
    assert lo > 0 and lo < point < hi  # a 20-point gap is not noise

    point, lo, hi = paired_difference(strong, strong.sample(frac=1, random_state=0))
    assert point == 0 and lo == 0 and hi == 0  # same predictions in another row order
