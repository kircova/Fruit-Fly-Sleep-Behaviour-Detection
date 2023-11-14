import numpy as np

from flyact.tracking import block_mean, crop_origin, detect_fly, smooth_track


def synthetic_frame(cx=150, cy=90, wing=True):
    """Low-res frame: grey chamber, dark elliptical body, long thin wing to the left."""
    h, w = 200, 275
    yy, xx = np.mgrid[:h, :w]
    img = np.full((h, w), 130.0, dtype=np.float32)
    img[((xx - cx) / 40) ** 2 + ((yy - cy) / 16) ** 2 <= 1] = 85
    if wing:
        img[cy + 10 : cy + 14, cx - 110 : cx - 30] = 95
    img[5:12, 130:150] = 80  # the food port: dark, but much smaller than a fly
    return img


def test_block_mean_shape_and_value():
    frame = np.arange(8 * 12, dtype=np.uint8).reshape(8, 12)
    small = block_mean(frame, 4)
    assert small.shape == (2, 3)
    assert small[0, 0] == frame[:4, :4].mean()


def test_detect_fly_finds_body():
    x, y = detect_fly(synthetic_frame())
    assert abs(x - 150) < 4 and abs(y - 90) < 4


def test_detection_centres_on_body_not_wing():
    # A plain centroid of the blob would be dragged ~10 px towards the wing.
    x_with, _ = detect_fly(synthetic_frame(wing=True))
    x_without, _ = detect_fly(synthetic_frame(wing=False))
    assert abs(x_with - x_without) < 3


def test_detect_fly_returns_none_on_empty_chamber():
    assert detect_fly(np.full((200, 275), 130.0, dtype=np.float32)) is None


def test_smooth_track_interpolates_and_rejects_outliers():
    raw = np.full((90, 2), np.nan, dtype=np.float32)
    raw[::3] = [500.0, 300.0]
    raw[30] = [50.0, 700.0]  # one frame where the detector latched onto something else
    track = smooth_track(raw, fps=30)
    assert track.shape == (90, 2)
    assert np.abs(track - [500.0, 300.0]).max() < 1.0


def test_smooth_track_without_any_detection_is_nan():
    assert np.isnan(smooth_track(np.full((10, 2), np.nan, dtype=np.float32), fps=30)).all()


def test_crop_origin_stays_inside_frame():
    assert crop_origin(np.array([10.0, 10.0]), 572, 1100, 800) == (0, 0)
    assert crop_origin(np.array([1090.0, 790.0]), 572, 1100, 800) == (528, 228)
    assert crop_origin(np.array([550.0, 400.0]), 572, 1100, 800) == (264, 114)
