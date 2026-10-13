"""Load configs/default.yaml into an attribute-accessible, read-only object.

CLI overrides use ``--set dotted.key=value``; the value is parsed as YAML, so
``--set train.batch_size=8`` gives an int and ``--set runtime.pretrained=false``
a bool.
"""
from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
MODEL_NAMES = ("resnet50", "efficientnet_b0", "vit", "cae", "patchcore")


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        # Guard against recursion while unpickling (``data`` not yet set).
        if name == "data" or name.startswith("__"):
            raise AttributeError(name)
        try:
            value = self.data[name]
        except KeyError as exc:
            raise AttributeError(f"config has no key '{name}'") from exc
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(name)

    def __contains__(self, name: str) -> bool:
        return name in self.data

    def get(self, name: str, default: Any = None) -> Any:
        return self.__getattr__(name) if name in self.data else default

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def path(self, key: str) -> Path:
        """Resolve an entry of ``paths:`` against the repository root."""
        return resolve(self.data["paths"][key])

    @property
    def data_root(self) -> Path:  # type: ignore[override]
        return resolve(self.data["data_root"])

    @property
    def cache_dir(self) -> Path:
        return resolve(self.data["cache"]["dir"])


def resolve(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def _apply_override(data: dict[str, Any], assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"--set expects key=value, got '{assignment}'")
    key, raw = assignment.split("=", 1)
    node = data
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set '{key}': '{part}' is not a mapping")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    for assignment in overrides or []:
        _apply_override(data, assignment)
    return Config(data)


def base_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the options every script shares."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config file")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config entry, e.g. --set train.batch_size=8 (repeatable)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def add_model_args(parser: argparse.ArgumentParser, seed: bool = True) -> None:
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--category", required=True, help="dataset category, e.g. can")
    if seed:
        parser.add_argument(
            "--seed", type=int, default=None,
            help="single seed; default runs every seed in train.seeds, one model at a time",
        )


def setup(args: argparse.Namespace) -> Config:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config, args.overrides)
    if getattr(args, "category", None) and args.category not in cfg.categories:
        raise SystemExit(f"category '{args.category}' not in config categories {cfg.categories}")
    return cfg


def seeds_for(args: argparse.Namespace, cfg: Config) -> list[int]:
    return [args.seed] if args.seed is not None else list(cfg.train.seeds)
