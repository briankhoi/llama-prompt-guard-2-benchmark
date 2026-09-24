"""Load scores, join them to the dataset, define evaluation slices, and locate injections inside tool outputs."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from pibench.common import ROOT

SHORT_NAMES = {
    "meta-llama/Llama-Prompt-Guard-2-22M": "PG2-22M",
    "meta-llama/Llama-Prompt-Guard-2-86M": "PG2-86M",
    # Context variant (score.py --context): user task + tool name serialized before the output.
    "meta-llama/Llama-Prompt-Guard-2-22M+ctx": "PG2-22M+ctx",
    "meta-llama/Llama-Prompt-Guard-2-86M+ctx": "PG2-86M+ctx",
    "protectai/deberta-v3-base-prompt-injection-v2": "protectai-v2",
    "keyword_generic": "keyword-generic",
    "keyword_template_aware": "keyword-template-aware",
}

SLICES = {
    "injecagent": "InjecAgent: all positives vs benign-filler negatives",
    "agentdojo": "AgentDojo: all positives vs all distinct benign tool outputs",
    "agentdojo_paired": "AgentDojo: positives vs only the clean versions of injected outputs (format/length matched)",
    "hard_negatives": "Hand-written hard negatives only (benign, so only FPR is defined)",
    "attacks_vs_hard_negatives": "All positives (both sources) vs hard negatives",
}


def slice_mask(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "injecagent":
        return df.source == "injecagent"
    if name == "agentdojo":
        return df.source == "agentdojo"
    if name == "agentdojo_paired":
        return (df.source == "agentdojo") & ((df.label == "attack") | df.paired)
    if name == "hard_negatives":
        return df.source == "hard_negatives"
    if name == "attacks_vs_hard_negatives":
        return (df.label == "attack") | (df.source == "hard_negatives")
    raise KeyError(name)


def load_scores(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """All score CSVs (excluding --limit smoke tests) in long format, plus each system's meta JSON."""
    scores_dir = ROOT / cfg["paths"]["scores_dir"]
    frames, metas = [], {}
    for fname in sorted(os.listdir(scores_dir)):
        if not fname.endswith(".csv") or "_limit" in fname:
            continue
        frames.append(pd.read_csv(scores_dir / fname, dtype={"matched_patterns": str}))
        meta_path = scores_dir / fname.replace(".csv", "_meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            metas[meta["model"]] = meta
    if not frames:
        raise FileNotFoundError(f"No score CSVs in {scores_dir}; run score.py and baseline.py first")
    scores = pd.concat(frames, ignore_index=True)
    scores["system"] = scores["model"].map(lambda m: SHORT_NAMES.get(m, m))
    return scores, metas


def join(scores: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    missing = set(data.example_id) - set(scores.example_id)
    if missing:
        raise ValueError(f"{len(missing)} dataset examples have no scores; dataset was rebuilt after scoring? Rescore.")
    merged = scores.merge(data, on="example_id", how="inner", validate="many_to_one")
    merged["y"] = (merged.label == "attack").astype(int)
    # Baselines don't tokenize; fall back to the dataset's reference token count for length buckets.
    if "n_tokens" not in merged:
        merged["n_tokens"] = np.nan
    merged["n_tokens"] = merged["n_tokens"].fillna(merged["n_tokens_stats"])
    return merged


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def injection_offsets(data: pd.DataFrame) -> pd.Series:
    """Character offset where the injected text starts in each attack example, found by diffing against a benign
    counterpart (InjecAgent: a benign fill of the same template; AgentDojo: the paired clean output of the same call)."""
    benign = data[data.label == "benign"]
    ia_ref = benign[benign.source == "injecagent"].groupby("tool_name")["tool_output_text"].first()
    ad_paired = benign[(benign.source == "agentdojo") & benign.paired]
    ad_ref = ad_paired.groupby(["suite", "user_task_id", "tool_name", "notes"])["tool_output_text"].first()
    # Dedup keeps only the first user task's metadata for an output shared by several tasks, so also match on the call alone.
    ad_ref_by_call = ad_paired.groupby(["suite", "tool_name", "notes"])["tool_output_text"].first()
    offsets = {}
    for row in data[data.label == "attack"].itertuples():
        if row.source == "injecagent":
            ref = ia_ref.get(row.tool_name)
        else:
            ref = ad_ref.get((row.suite, row.user_task_id, row.tool_name, row.notes))
            if ref is None:
                ref = ad_ref_by_call.get((row.suite, row.tool_name, row.notes))
        offsets[row.example_id] = _common_prefix_len(row.tool_output_text, ref) if isinstance(ref, str) else np.nan
    return pd.Series(offsets, name="inj_char_start")


def injection_token_offsets(data: pd.DataFrame, char_offsets: pd.Series, tokenizer_name: str) -> pd.Series:
    """Token index (after the leading special token) where the injection starts, using the given tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    texts = data.set_index("example_id")["tool_output_text"]
    out = {}
    for eid, off in char_offsets.items():
        out[eid] = np.nan if np.isnan(off) else len(tok(texts[eid][: int(off)], add_special_tokens=False)["input_ids"])
    return pd.Series(out, name="inj_tok_start")


def excerpt(text: str, start: float, n_chars: int) -> str:
    """Excerpt around the injection start (or the beginning), single-lined for Markdown tables."""
    begin = 0 if (start is None or np.isnan(start)) else max(0, int(start) - 60)
    snippet = text[begin : begin + n_chars]
    snippet = ("…" if begin > 0 else "") + snippet + ("…" if begin + n_chars < len(text) else "")
    return " ".join(snippet.split()).replace("|", "\\|")
