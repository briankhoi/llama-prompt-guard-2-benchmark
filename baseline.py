"""Keyword/regex baseline, written in the same score-CSV format as score.py so evaluate.py treats it as another model.

Usage: python baseline.py
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd

from pibench.common import ROOT, load_config, load_dataset


def compile_variant(cfg: dict, groups: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for g in groups for p in cfg["baseline"]["patterns"][g]]


def score_text(patterns: list[re.Pattern], text: str) -> tuple[int, list[int]]:
    hits = [i for i, p in enumerate(patterns) if p.search(text)]
    return len(hits), hits


def run_variant(name: str, patterns: list[re.Pattern], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for eid, text in zip(df["example_id"], df["tool_output_text"]):
        start = time.perf_counter()
        n_hits, hits = score_text(patterns, text)
        latency_ms = (time.perf_counter() - start) * 1000
        rows.append(
            {
                "example_id": eid,
                "model": name,
                "mode": "full",
                "score": float(n_hits),
                "matched_patterns": ";".join(map(str, hits)),
                "n_windows": 1,
                "latency_ms": latency_ms,
                "device": "cpu",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = load_dataset(cfg)
    out_dir = ROOT / cfg["paths"]["scores_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, groups in cfg["baseline"]["variants"].items():
        scores = run_variant(name, compile_variant(cfg, groups), df)
        scores.to_csv(out_dir / f"{name}.csv", index=False)
        lat = scores["latency_ms"].to_numpy()
        meta = {
            "model": name,
            "pattern_groups": groups,
            "device": "cpu",
            "latency": {"full": {"mean_ms": float(lat.mean()), "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95))}},
        }
        (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{name}: flagged {int((scores.score >= 1).sum())}/{len(scores)}")


if __name__ == "__main__":
    main()
