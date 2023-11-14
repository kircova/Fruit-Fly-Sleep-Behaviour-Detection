import numpy as np
import pandas as pd

from flyact import CLASSES
from flyact.metrics import predictions_frame, summarize


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
