"""Experiment configuration: plain dataclasses, loaded from YAML, overridable from the CLI."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SamplingConfig:
    """How a clip is turned into network input.

    A window is ``num_snippets`` snippets spaced ``stride`` frames apart. Each snippet is
    three frames ``delta`` apart, stacked as the three input channels. The footage is
    monochrome, so the channels an ImageNet backbone would spend on colour are free to
    carry time instead: within a snippet the network sees ~130 ms of motion directly, and
    across snippets it sees ~2 s of behaviour.
    """

    num_snippets: int = 8
    stride: int = 8
    delta: int = 2
    input_mode: str = "triplet"  # "triplet" | "static" (centre frame repeated three times)
    image_size: int = 160
    eval_windows: int = 10  # windows averaged per clip at test time

    @property
    def span(self) -> int:
        """Number of source frames covered by one window."""
        return (self.num_snippets - 1) * self.stride + 2 * self.delta + 1


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    temporal: str = "tsm"  # "tsm" | "none"
    pretrained: bool = True
    dropout: float = 0.3


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    warmup_epochs: float = 1.0
    augment: bool = True
    eval_every: int = 0  # epochs between monitoring evaluations of the test fold (0 = off)
    seed: int = 0
    save_checkpoints: bool = False


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    # "group": test folds are whole recordings (the honest protocol).
    # "random": clips are split at random, ignoring recordings (kept to measure leakage).
    split: str = "group"
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes"}
    if current is None:
        return yaml.safe_load(value)
    return type(current)(value)


def _apply(obj: Any, updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            _apply(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Defaults, then the YAML file, then ``section.key=value`` overrides."""
    cfg = ExperimentConfig()
    if path is not None:
        _apply(cfg, yaml.safe_load(Path(path).read_text()) or {})
        if "name" not in (yaml.safe_load(Path(path).read_text()) or {}):
            cfg.name = Path(path).stem
    for item in overrides or []:
        dotted, _, raw = item.partition("=")
        *parents, leaf = dotted.split(".")
        target = cfg
        for part in parents:
            target = getattr(target, part)
        setattr(target, leaf, _coerce(raw, getattr(target, leaf)))
    return cfg


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    _apply(cfg, data)
    return cfg
