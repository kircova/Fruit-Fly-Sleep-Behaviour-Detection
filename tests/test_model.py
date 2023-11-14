import torch

from flyact.config import ModelConfig
from flyact.model import FlyNet, temporal_shift


def test_temporal_shift_moves_the_right_channels():
    b, t, c = 2, 4, 16
    x = torch.arange(b * t, dtype=torch.float32).view(b * t, 1, 1, 1).expand(b * t, c, 1, 1)
    out = temporal_shift(x, num_snippets=t, fold_div=8).view(b, t, c)
    index = torch.arange(b * t, dtype=torch.float32).view(b, t)
    fold = c // 8
    # First fold reads the next snippet, zero-padded at the end of the window.
    assert torch.equal(out[:, :-1, 0], index[:, 1:]) and (out[:, -1, :fold] == 0).all()
    # Second fold reads the previous snippet, zero-padded at the start.
    assert torch.equal(out[:, 1:, fold], index[:, :-1]) and (out[:, 0, fold : 2 * fold] == 0).all()
    # Everything else is untouched, and nothing leaks between windows of the batch.
    assert torch.equal(out[:, :, 2 * fold :], index[:, :, None].expand(b, t, c - 2 * fold))


def make(temporal):
    torch.manual_seed(0)
    cfg = ModelConfig(temporal=temporal, pretrained=False, dropout=0.0)
    return FlyNet(cfg, num_snippets=4).eval()


def test_forward_shape():
    assert make("tsm")(torch.randn(2, 4, 3, 64, 64)).shape == (2, 3)


def test_only_the_tsm_model_is_sensitive_to_snippet_order():
    x = torch.randn(1, 4, 3, 64, 64)
    shuffled = x[:, [2, 0, 3, 1]]
    with torch.no_grad():
        plain, tsm = make("none"), make("tsm")
        assert torch.allclose(plain(x), plain(shuffled), atol=1e-5)
        assert not torch.allclose(tsm(x), tsm(shuffled), atol=1e-5)
