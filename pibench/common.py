"""Shared helpers: config loading, seeding, stable ids, dataset IO."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

DATASET_COLUMNS = [
    "example_id",
    "source",
    "slice",
    "tool_name",
    "user_task",
    "tool_output_text",
    "label",
    "attack_category",
    "setting",
    "attack_goal",
    "suite",
    "user_task_id",
    "injection_task_id",
    "attack_name",
    "n_occurrences",
    "paired",
    "notes",
]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def stable_id(*parts: str) -> str:
    """Deterministic short id from the parts (source, label, text)."""
    h = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def dataset_path(cfg: dict) -> Path:
    return ROOT / cfg["paths"]["processed_dir"] / "dataset.parquet"


def load_dataset(cfg: dict) -> pd.DataFrame:
    path = dataset_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run `make dataset` (build_dataset.py) first")
    return pd.read_parquet(path)


def dedupe_rows(rows: list[dict]) -> list[dict]:
    """Collapse rows with identical (source, label, text), keeping the first row's metadata and counting occurrences."""
    merged: dict[tuple, dict] = {}
    for row in rows:
        key = (row["source"], row["label"], row["tool_output_text"])
        if key in merged:
            prev = merged[key]
            merged[key] = {
                **prev,
                "n_occurrences": prev["n_occurrences"] + row.get("n_occurrences", 1),
                "paired": bool(prev.get("paired")) or bool(row.get("paired")),
            }
        else:
            merged[key] = {**row, "n_occurrences": row.get("n_occurrences", 1)}
    return list(merged.values())


def finalize_rows(rows: list[dict]) -> pd.DataFrame:
    """Fill defaults, assign stable ids, and order columns."""
    out = []
    for row in rows:
        full = {col: row.get(col) for col in DATASET_COLUMNS}
        full["example_id"] = stable_id(row["source"], row["slice"], row["label"], row["tool_output_text"])
        full["n_occurrences"] = int(row.get("n_occurrences", 1))
        full["paired"] = bool(row.get("paired", False))
        out.append(full)
    df = pd.DataFrame(out, columns=DATASET_COLUMNS)
    if df["example_id"].duplicated().any():
        raise ValueError("example_id collision after dedup, check dedupe logic")
    return df
