"""Training throughput of candidate video backbones on this machine.

    python scripts/benchmark_backbones.py

Times forward + backward + optimiser step on random input, in clips per second. This is
the measurement behind the choice of a 2D backbone over a 3D CNN: on
an Apple-silicon GPU the 3D convolutions are an order of magnitude slower per clip.
"""

from __future__ import annotations

import time

import torch
from torchvision.models import video

from flyact.config import ModelConfig
from flyact.model import FlyNet
from flyact.train import pick_device


class SnippetsAsClip(torch.nn.Module):
    """Adapter so FlyNet takes the same ``(B, 3, T, H, W)`` input as the video models."""

    def __init__(self, num_snippets: int, temporal: str = "none"):
        super().__init__()
        self.net = FlyNet(ModelConfig(pretrained=False, temporal=temporal), num_snippets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2))


# (name, constructor, (frames, size), batch size). The 224 px models get a smaller batch:
# at 8 clips Swin3D-T no longer fits in 24 GB of unified memory and starts swapping.
CANDIDATES = [
    ("ResNet-18 on frame triplets (this repo)", lambda: SnippetsAsClip(8), (8, 160), 8),
    ("  ... with temporal shift", lambda: SnippetsAsClip(8, "tsm"), (8, 160), 8),
    ("R(2+1)D-18", video.r2plus1d_18, (16, 112), 8),
    ("MC3-18", video.mc3_18, (16, 112), 8),
    ("R3D-18", video.r3d_18, (16, 112), 8),
    ("S3D", video.s3d, (16, 224), 4),
    ("Swin3D-T", video.swin3d_t, (16, 224), 4),
]


def clips_per_second(
    make, frames: int, size: int, batch: int, device: torch.device, steps: int = 8
) -> float:
    model = make().to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(batch, 3, frames, size, size, device=device)
    y = torch.randint(0, 3, (batch,), device=device)
    for step in range(steps + 2):
        if step == 2:  # the first steps include allocation and kernel compilation
            _sync(device)
            start = time.perf_counter()
        loss = torch.nn.functional.cross_entropy(model(x)[:, :3], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _sync(device)
    return batch * steps / (time.perf_counter() - start)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    device = pick_device()
    print(f"device: {device}\n")
    print(f"{'model':40s} {'input (T x H x W)':>18s} {'batch':>6s} {'params':>8s} {'clips/s':>8s}")
    for name, make, (frames, size), batch in CANDIDATES:
        params = sum(p.numel() for p in make().parameters()) / 1e6
        speed = clips_per_second(make, frames, size, batch, device)
        shape = f"{frames} x {size} x {size}"
        print(f"{name:40s} {shape:>18s} {batch:6d} {params:7.1f}M {speed:8.1f}", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
