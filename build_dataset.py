"""Build the labeled tool-output dataset from InjecAgent, AgentDojo, and the hand-written hard negatives.

Usage: python build_dataset.py [--config config.yaml] [--skip-agentdojo]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pibench import injecagent
from pibench.common import ROOT, dataset_path, dedupe_rows, finalize_rows, load_config, seed_everything

STATS_TOKENIZER = "protectai/deberta-v3-base-prompt-injection-v2"


def _check_repo_commit(repo_dir: Path, expected: str) -> None:
    if not repo_dir.exists():
        raise FileNotFoundError(f"{repo_dir} missing, run `make fetch` to clone the pinned sources")
    head = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != expected:
        raise RuntimeError(f"{repo_dir} is at {head}, expected pinned commit {expected}")


def _check_agentdojo_version(expected: str) -> None:
    import importlib.metadata

    installed = importlib.metadata.version("agentdojo")
    if installed != expected:
        raise RuntimeError(f"agentdojo {installed} installed, config pins {expected}")


def build_injecagent(cfg: dict) -> tuple[list[dict], dict]:
    src = cfg["sources"]["injecagent"]
    repo = ROOT / cfg["paths"]["external_dir"] / "InjecAgent"
    _check_repo_commit(repo, src["commit"])
    pos, pos_checks = injecagent.load_positive_rows(repo, src)
    if pos_checks["template_mismatch"] or pos_checks["slot_not_instruction"]:
        raise RuntimeError(f"InjecAgent responses don't match template + attacker instruction: {pos_checks}")
    neg, neg_checks = injecagent.load_negative_rows(repo, src, ROOT / cfg["paths"]["injecagent_fillers"])
    return pos + neg, {"positives_raw": len(pos), "negatives_raw": len(neg), **pos_checks, **neg_checks}


def build_agentdojo(cfg: dict) -> tuple[list[dict], dict]:
    from pibench import agentdojo_extract

    src = cfg["sources"]["agentdojo"]
    _check_agentdojo_version(src["package_version"])
    rows, stats = agentdojo_extract.extract(src)
    return rows, stats


def build_hard_negatives(cfg: dict) -> list[dict]:
    items = yaml.safe_load((ROOT / cfg["paths"]["hard_negatives"]).read_text())
    return [
        {
            "source": "hard_negatives",
            "slice": "hard_negatives",
            "tool_name": it["tool_name"],
            "user_task": it["user_task"],
            "tool_output_text": it["text"].strip(),
            "label": "benign",
            "attack_category": "none",
            "notes": f"hard_negative_category={it['category']}",
        }
        for it in items
    ]


def _token_lengths(texts: pd.Series) -> np.ndarray:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(STATS_TOKENIZER)
    return np.array([len(ids) for ids in tok(list(texts), add_special_tokens=True)["input_ids"]])


def print_stats(df: pd.DataFrame) -> str:
    lines = [f"Total rows: {len(df)}", ""]
    lines.append(df.groupby(["source", "label"]).size().to_frame("n").to_markdown())
    lines.append("")
    lines.append(df.groupby(["source", "label", "attack_category"]).size().to_frame("n").to_markdown())
    lines.append("")
    lines.append(df[df.source == "injecagent"].groupby(["label", "setting", "attack_goal"], dropna=False).size().to_frame("n").to_markdown())
    lines.append("")
    lines.append(f"Token lengths (tokenizer: {STATS_TOKENIZER}, incl. special tokens)")
    desc = df.groupby(["source", "label"])["n_tokens_stats"].describe(percentiles=[0.5, 0.9, 0.99])
    lines.append(desc.round(1).to_markdown())
    lines.append("")
    ad = df[df.source == "agentdojo"]
    lines.append("AgentDojo token lengths by `paired` (benign paired = clean version of an output that gets injected):")
    lines.append(ad.groupby(["label", "paired"])["n_tokens_stats"].describe(percentiles=[0.5, 0.9]).round(1).to_markdown())
    lines.append("")
    over = df.assign(over_512=df.n_tokens_stats > 512).groupby(["source", "label"])["over_512"].agg(["sum", "mean"])
    lines.append("Rows over 512 tokens:")
    lines.append(over.round(3).to_markdown())
    text = "\n".join(lines)
    print(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--skip-agentdojo", action="store_true", help="build without AgentDojo (e.g. if extraction is blocked)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    ia_rows, ia_checks = build_injecagent(cfg)
    print("[injecagent]", json.dumps(ia_checks))
    rows = list(ia_rows)
    build_meta = {"injecagent": ia_checks}
    if not args.skip_agentdojo:
        ad_rows, ad_stats = build_agentdojo(cfg)
        print("[agentdojo] runs:", ad_stats["runs"], "label disagreements:", ad_stats["label_disagreements"],
              "replay errors:", len(ad_stats["benign_replay_errors"]) + len(ad_stats["attack_replay_errors"]))
        rows.extend(ad_rows)
        build_meta["agentdojo"] = ad_stats
    rows.extend(build_hard_negatives(cfg))

    raw_counts = pd.DataFrame(rows).groupby(["source", "label"]).size().to_dict()
    df = finalize_rows(dedupe_rows(rows))
    df["n_tokens_stats"] = _token_lengths(df["tool_output_text"])
    build_meta["raw_counts_before_dedup"] = {f"{k[0]}/{k[1]}": int(v) for k, v in raw_counts.items()}
    build_meta["cross_label_duplicate_texts"] = int(df.duplicated(["source", "tool_output_text"], keep=False).sum())

    out = dataset_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)
    stats_text = print_stats(df)
    (out.parent / "dataset_stats.md").write_text(stats_text + "\n")
    (out.parent / "build_meta.json").write_text(json.dumps(build_meta, indent=2, default=str))
    print(f"\nWrote {len(df)} rows to {out} (+ .csv, dataset_stats.md, build_meta.json)")
    print("Raw counts before dedup:", build_meta["raw_counts_before_dedup"])
    print("Texts appearing with both labels in the same source:", build_meta["cross_label_duplicate_texts"])


if __name__ == "__main__":
    main()
