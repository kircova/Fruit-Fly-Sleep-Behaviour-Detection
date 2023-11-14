from pathlib import Path

import pytest

from flyact.index import build_index, infer_label, parse_bout_name


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("Actions/400-Feeding/v_Feeding_g01_c0.avi", "feeding"),
        ("Actions/Feeding 2/v_feeding_g01_17.avi", "feeding"),
        ("Actions/Action_v5/v_Grooming_g01_c12.avi", "grooming"),
        ("Files/Video Files/pump/prob_12_638302_638358.avi", "pumping"),
        ("Files/Video Files/pump/ProbPumbping_0_33461_33494.avi", "pumping"),
        ("Files/Video Files/Grooming/grooming_47_231326_233398.avi", "grooming"),
        ("Motion-NoMotion/output.avi", None),
    ],
)
def test_infer_label(path, label):
    assert infer_label(Path(path)) == label


def test_file_name_wins_over_folder():
    # Mixed-class folders exist (Action_v5 holds all three classes).
    assert infer_label(Path("Feeding/v_Pumping_g01_c3.avi")) == "pumping"


def test_parse_bout_name():
    assert parse_bout_name("prob_12_638302_638358") == (12, 638302, 638358)
    assert parse_bout_name("grooming_7_11704_14147") == (7, 11704, 14147)
    assert parse_bout_name("v_Feeding_g01_c0") is None
    assert parse_bout_name("prob_1_500_400") is None  # stop before start is not a bout name


def test_build_index_deduplicates_by_content(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    (tmp_path / "v1" / "grooming_3_100_250.avi").write_bytes(b"clip-a")
    (tmp_path / "v2" / "v_Grooming_g01_c0.avi").write_bytes(b"clip-a")  # renamed copy
    (tmp_path / "v2" / "v_Feeding_g01_c0.avi").write_bytes(b"clip-b")
    (tmp_path / "v2" / "notes.avi").write_bytes(b"unlabelled")
    (tmp_path / "feeding.mp4").write_bytes(b"compilation")  # not a bout clip

    df = build_index(tmp_path, workers=1)

    assert sorted(df["label"]) == ["feeding", "grooming"]
    grooming = df[df["label"] == "grooming"].iloc[0]
    assert grooming["n_copies"] == 2
    # The original bout name survives even though the canonical path is another copy.
    assert (grooming["bout_idx"], grooming["src_start"], grooming["src_stop"]) == (3, 100, 250)
