import numpy as np
import pytest
import torch

from flyact.config import SamplingConfig
from flyact.data import (
    balanced_sampler,
    effective_stride,
    eval_starts,
    to_input,
    window_indices,
)

CFG = SamplingConfig()  # 8 snippets, stride 8, delta 2 -> 61-frame windows


def test_span_matches_the_indices_actually_used():
    rows = window_indices(1000, start=0, stride=CFG.stride, cfg=CFG)
    assert rows.shape == (8, 3)
    assert rows.min() == 0 and rows.max() == CFG.span - 1
    assert (np.diff(rows, axis=1) == CFG.delta).all()
    assert (np.diff(rows[:, 1]) == CFG.stride).all()


@pytest.mark.parametrize("n_frames", [16, 20, 33, 60, 61, 62, 500])
def test_windows_never_leave_the_clip(n_frames):
    stride = effective_stride(n_frames, CFG)
    for start in eval_starts(n_frames, CFG):
        rows = window_indices(n_frames, start, stride, CFG)
        assert rows.min() >= 0 and rows.max() <= n_frames - 1
    # Clips that can hold a full window must not be squeezed.
    assert (stride == CFG.stride) == (n_frames >= CFG.span)


def test_short_clip_is_sampled_denser_not_padded():
    n_frames = 33
    stride = effective_stride(n_frames, CFG)
    rows = window_indices(n_frames, 0, stride, CFG)
    assert stride == 4
    assert len(np.unique(rows[:, 1])) == CFG.num_snippets  # eight distinct centre frames


def test_eval_windows_cover_the_clip_and_respect_the_cap():
    assert eval_starts(61, CFG) == [0]
    starts = eval_starts(5000, CFG)
    assert len(starts) == CFG.eval_windows
    assert starts[0] == 0 and starts[-1] == 5000 - CFG.span


def test_static_mode_repeats_the_centre_frame():
    cfg = SamplingConfig(input_mode="static")
    rows = window_indices(1000, 0, cfg.stride, cfg)
    assert (rows[:, 0] == rows[:, 1]).all() and (rows[:, 1] == rows[:, 2]).all()


def test_to_input_shape_and_standardisation():
    frames = np.random.default_rng(0).integers(0, 255, (24, 224, 224), dtype=np.uint8)
    rows = np.arange(24).reshape(8, 3)
    x = to_input(frames, rows, size=160, augment=False)
    assert x.shape == (8, 3, 160, 160)
    assert abs(float(x.mean())) < 1e-3 and abs(float(x.std()) - 1) < 1e-2


def test_augmentation_is_identical_for_every_frame_of_a_window():
    # If all frames are the same image, they must still be the same after augmentation:
    # otherwise augmentation would be inventing motion.
    image = np.random.default_rng(1).integers(0, 255, (224, 224), dtype=np.uint8)
    frames = np.repeat(image[None], 24, axis=0)
    rows = np.arange(24).reshape(8, 3)
    torch.manual_seed(0)
    x = to_input(frames, rows, size=160, augment=True)
    assert torch.allclose(x, x[0, 0].expand_as(x))


def test_balanced_sampler_equalises_classes():
    labels = [0] * 800 + [1] * 150 + [2] * 50
    drawn = np.array(labels)[list(balanced_sampler(labels, 6000, seed=0))]
    assert np.abs(np.bincount(drawn) / 6000 - 1 / 3).max() < 0.03
