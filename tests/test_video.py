import numpy as np
import pytest

from flyact.video import CACHE_GOP, GrayVideoWriter, iter_gray_frames, probe, read_cached_frames

N, SIZE = 100, 64


def frame_with_marker(i):
    """Frame i carries a bright block whose position encodes i."""
    img = np.full((SIZE, SIZE), 60, dtype=np.uint8)
    row, col = divmod(i, 10)
    img[row * 6 : row * 6 + 4, col * 6 : col * 6 + 4] = 230
    return img


def decode_marker(img):
    rows, cols = np.nonzero(img > 150)
    return int(round(rows.min() / 6)) * 10 + int(round(cols.min() / 6))


@pytest.fixture(scope="module")
def cache_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("cache") / "clip.mp4"
    with GrayVideoWriter(path, SIZE) as writer:
        for i in range(N):
            writer.write(frame_with_marker(i))
    return path


def test_roundtrip_preserves_every_frame_in_order(cache_file):
    assert probe(cache_file).n_frames == N
    assert [decode_marker(f) for f in iter_gray_frames(cache_file)] == list(range(N))


def test_random_access_returns_the_requested_frames(cache_file):
    rng = np.random.default_rng(0)
    for _ in range(25):
        start = int(rng.integers(0, N - 30))
        indices = start + np.arange(0, 30, 4)
        frames = read_cached_frames(cache_file, indices)
        assert [decode_marker(f) for f in frames] == indices.tolist()


def test_random_access_handles_gop_boundaries_repeats_and_order(cache_file):
    indices = [CACHE_GOP, CACHE_GOP - 1, CACHE_GOP + 1, CACHE_GOP, N - 1, 0]
    frames = read_cached_frames(cache_file, indices)
    assert [decode_marker(f) for f in frames] == indices


def test_reading_past_the_end_fails_loudly(cache_file):
    with pytest.raises(OSError):
        read_cached_frames(cache_file, [N + 5])
