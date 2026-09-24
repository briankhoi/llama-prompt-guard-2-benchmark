"""Score every dataset example with a sequence-classification injection detector, in truncation and chunking modes.

Usage:
  python score.py --model meta-llama/Llama-Prompt-Guard-2-86M
  python score.py --model protectai/deberta-v3-base-prompt-injection-v2 --limit 50
  python score.py --context            # user task + tool name + output, for scoring.context.models
Writes results/scores/<model_slug>.csv (one row per example x mode) and <model_slug>_meta.json.
With --context the system is named <model>+ctx, and the prefix-only control goes to results/context_prefix_only/.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pibench.common import ROOT, load_config, load_dataset, seed_everything
from pibench.context import build_prefix, context_stride

MODES = ("truncate", "chunk")


def model_slug(name: str) -> str:
    return name.replace("/", "__")


def resolve_malicious_index(model, name: str, cfg: dict, cli_index: int | None) -> tuple[int, str]:
    """Pick the output index meaning 'malicious' from an explicit override or from descriptive id2label names."""
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    if cli_index is not None:
        return cli_index, f"--malicious-index {cli_index} (id2label={id2label})"
    override = cfg["scoring"].get("malicious_index_overrides", {}).get(name)
    if override is not None:
        return int(override), f"config override (id2label={id2label})"
    wanted = {n.upper() for n in cfg["scoring"]["malicious_label_names"]}
    hits = [i for i, lab in id2label.items() if lab.upper() in wanted]
    if len(hits) == 1:
        return hits[0], f"id2label name '{id2label[hits[0]]}'"
    raise SystemExit(
        f"Cannot tell which output of {name} is malicious from id2label={id2label}. "
        "Check the model card, then pass --malicious-index N or add it to scoring.malicious_index_overrides in config.yaml."
    )


def pick_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def windows(content_ids: list[int], content_len: int, stride: int, mode: str) -> list[list[int]]:
    """Split content token ids (no special tokens) into windows of at most content_len tokens."""
    if mode == "truncate" or len(content_ids) <= content_len:
        return [content_ids[:content_len]]
    starts = list(range(0, len(content_ids) - content_len, stride)) + [len(content_ids) - content_len]
    return [content_ids[s : s + content_len] for s in starts]


def special_token_wrapper(tok) -> tuple[list[int], list[int]]:
    """Return the special tokens the tokenizer puts before/after a single sequence (e.g. [CLS] ... [SEP] for DeBERTa)."""
    empty = tok("", add_special_tokens=True)["input_ids"]
    if tok.cls_token_id is not None and tok.sep_token_id is not None and empty == [tok.cls_token_id, tok.sep_token_id]:
        return [tok.cls_token_id], [tok.sep_token_id]
    raise RuntimeError(f"Unrecognized special-token layout {empty} for {type(tok).__name__}; extend special_token_wrapper")


class Scorer:
    def __init__(self, name: str, cfg: dict, device: torch.device, malicious_index: int | None):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.malicious_index, self.malicious_source = resolve_malicious_index(self.model, name, cfg, malicious_index)
        self.max_length = cfg["scoring"]["max_length"]
        self.prefix_ids, self.suffix_ids = special_token_wrapper(self.tok)
        self.content_len = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        self.stride = cfg["scoring"]["chunk_stride"]
        if not 0 < self.stride <= self.content_len:
            raise ValueError(f"chunk_stride must be in (0, {self.content_len}]")
        self.ctx_overlap = cfg["scoring"]["context"]["chunk_overlap"]
        self.device = device
        self.model.to(device)

    def wrap(self, content_ids: list[int]) -> list[int]:
        return self.prefix_ids + content_ids + self.suffix_ids

    def _sync(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def score_windows(self, wins: list[list[int]]) -> np.ndarray:
        batch = [self.wrap(w) for w in wins]
        width = max(len(b) for b in batch)
        pad = self.tok.pad_token_id
        ids = torch.tensor([b + [pad] * (width - len(b)) for b in batch], device=self.device)
        mask = torch.tensor([[1] * len(b) + [0] * (width - len(b)) for b in batch], device=self.device)
        logits = self.model(input_ids=ids, attention_mask=mask).logits.float()
        return torch.softmax(logits, dim=-1)[:, self.malicious_index].cpu().numpy()

    def context_windows(self, context_ids: list[int], content: list[int], mode: str) -> tuple[list[list[int]], int]:
        """Windows over the output only, each preceded by the full context prefix; returns (windows, output room)."""
        room = self.content_len - len(context_ids)
        if room < 1:
            raise ValueError(f"Context prefix of {len(context_ids)} tokens leaves no room for output in a {self.max_length}-token window")
        return [context_ids + w for w in windows(content, room, context_stride(room, self.ctx_overlap), mode)], room

    def score_text(self, text: str, mode: str, context_prefix: str | None = None) -> dict:
        start = time.perf_counter()
        content = self.tok(text, add_special_tokens=False)["input_ids"]
        extra = {}
        if context_prefix is None:
            wins = windows(content, self.content_len, self.stride, mode)
            n_context = 0
        else:
            context_ids = self.tok(context_prefix, add_special_tokens=False)["input_ids"]
            wins, room = self.context_windows(context_ids, content, mode)
            n_context = len(context_ids)
            extra = {"prefix_tokens": n_context, "output_room": room}
        probs = self.score_windows(wins)
        self._sync()
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "score": float(probs.max()),
            "argmax_window": int(probs.argmax()),
            "n_windows": len(wins),
            "n_tokens": n_context + len(content) + len(self.prefix_ids) + len(self.suffix_ids),
            "latency_ms": latency_ms,
            **extra,
        }


def check_windowing_matches_tokenizer(scorer: Scorer, texts: list[str]) -> None:
    """Our manual window for a short text must equal the tokenizer's own truncated encoding."""
    for text in texts:
        content = scorer.tok(text, add_special_tokens=False)["input_ids"]
        manual = scorer.wrap(windows(content, scorer.content_len, scorer.stride, "truncate")[0])
        native = scorer.tok(text, truncation=True, max_length=scorer.max_length)["input_ids"]
        if manual != native:
            raise RuntimeError(f"Manual windowing disagrees with tokenizer truncation for {scorer.name}")


def check_context_windowing(scorer: Scorer, texts: list[str], prefixes: list[str]) -> None:
    """With context, the first window must equal the tokenizer's own truncated encoding of prefix + output."""
    for text, prefix in zip(texts, prefixes):
        content = scorer.tok(text, add_special_tokens=False)["input_ids"]
        context_ids = scorer.tok(prefix, add_special_tokens=False)["input_ids"]
        manual = scorer.wrap(scorer.context_windows(context_ids, content, "truncate")[0][0])
        native = scorer.tok(prefix + text, truncation=True, max_length=scorer.max_length)["input_ids"]
        if manual != native:
            raise RuntimeError(f"Context windowing disagrees with tokenizer truncation of prefix + output for {scorer.name}")


def run(scorer: Scorer, df: pd.DataFrame, warmup: int, prefixes: list[str | None], label: str) -> pd.DataFrame:
    for text, prefix in list(zip(df["tool_output_text"], prefixes))[:warmup]:
        scorer.score_text(text, "chunk", prefix)
    rows = []
    for i, (eid, text, prefix) in enumerate(zip(df["example_id"], df["tool_output_text"], prefixes)):
        for mode in MODES:
            # Untimed first call so MPS's one-off kernel compilation for a new input shape isn't counted as latency.
            scorer.score_text(text, mode, prefix)
            rows.append({"example_id": eid, "model": label, "mode": mode, **scorer.score_text(text, mode, prefix)})
        if (i + 1) % 500 == 0:
            print(f"  {label}: {i + 1}/{len(df)}", flush=True)
    return pd.DataFrame(rows)


def latency_summary(scores: pd.DataFrame) -> dict:
    out = {}
    for mode, g in scores.groupby("mode"):
        lat = g["latency_ms"].to_numpy()
        out[mode] = {
            "mean_ms": float(lat.mean()),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "max_ms": float(lat.max()),
            "mean_windows": float(g["n_windows"].mean()),
        }
    return out


def context_prefixes(df: pd.DataFrame, cfg: dict) -> list[str]:
    template = cfg["scoring"]["context"]["template"]
    return [build_prefix(template, ut, tn) for ut, tn in zip(df["user_task"], df["tool_name"])]


def score_prefix_only(scorer: Scorer, df: pd.DataFrame, cfg: dict, label: str) -> pd.DataFrame:
    """Control: the serialized context with an empty output, once per distinct (source, user_task, tool_name)."""
    pairs = df[["source", "user_task", "tool_name"]].drop_duplicates().reset_index(drop=True)
    rows = []
    for r in pairs.itertuples(index=False):
        prefix = build_prefix(cfg["scoring"]["context"]["template"], r.user_task, r.tool_name)
        res = scorer.score_text(prefix, "truncate")
        rows.append({"source": r.source, "user_task": r.user_task, "tool_name": r.tool_name, "model": label,
                     "score": res["score"], "n_tokens": res["n_tokens"]})
    return pd.DataFrame(rows)


def room_summary(scores: pd.DataFrame, df: pd.DataFrame, low_room: int) -> dict:
    """How much of each window the context prefix uses, overall and per source."""
    s = scores[scores["mode"] == "chunk"].merge(df[["example_id", "source"]], on="example_id")
    out = {"low_room_tokens": low_room}
    for src, g in [("all", s), *s.groupby("source")]:
        out[src] = {
            "prefix_tokens_median": float(g.prefix_tokens.median()),
            "prefix_tokens_max": int(g.prefix_tokens.max()),
            "output_room_min": int(g.output_room.min()),
            "n_below_low_room": int((g.output_room < low_room).sum()),
            "n": int(len(g)),
        }
    return out


def score_model(name: str, cfg: dict, args) -> None:
    seed_everything(cfg["seed"])
    df = load_dataset(cfg)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=cfg["seed"])
    label = name + cfg["scoring"]["context"]["system_suffix"] if args.context else name
    prefixes = context_prefixes(df, cfg) if args.context else [None] * len(df)
    device = pick_device(args.device or cfg["scoring"]["device"])

    def attempt(dev):
        scorer = Scorer(name, cfg, dev, args.malicious_index)
        if args.context:
            check_context_windowing(scorer, df["tool_output_text"].head(20).tolist(), prefixes[:20])
        else:
            check_windowing_matches_tokenizer(scorer, df["tool_output_text"].head(20).tolist())
        return scorer, run(scorer, df, cfg["scoring"]["warmup_examples"], prefixes, label)

    try:
        scorer, scores = attempt(device)
    except RuntimeError as exc:
        if device.type == "cpu":
            raise
        print(f"[warn] {device} failed ({exc}); falling back to CPU", flush=True)
        device = torch.device("cpu")
        scorer, scores = attempt(device)
    scores["device"] = device.type

    out_dir = ROOT / cfg["paths"]["scores_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    csv_path = out_dir / f"{model_slug(label)}{suffix}.csv"
    scores.to_csv(csv_path, index=False)
    meta = {
        "model": label,
        "base_model": name,
        "malicious_index": scorer.malicious_index,
        "malicious_index_source": scorer.malicious_source,
        "device": device.type,
        "max_length": scorer.max_length,
        "window_content_tokens": scorer.content_len,
        "chunk_stride": scorer.stride,
        "n_examples": int(len(df)),
        "latency": latency_summary(scores),
        "latency_note": "per example, one batch holding that example's windows, includes tokenization and device sync; timed on a second call per (example, mode) so first-seen-shape compilation on MPS is excluded",
        "torch": torch.__version__,
        "machine": platform.platform(),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.context:
        ctx = cfg["scoring"]["context"]
        meta["context"] = {"template": ctx["template"], "chunk_overlap": ctx["chunk_overlap"],
                           "stride_rule": "output room - overlap, floored at half the output room",
                           "room": room_summary(scores, df, ctx["low_room_tokens"])}
        control = score_prefix_only(scorer, df, cfg, label)
        control_dir = ROOT / "results" / "context_prefix_only"
        control_dir.mkdir(parents=True, exist_ok=True)
        control.to_csv(control_dir / f"{model_slug(label)}{suffix}.csv", index=False)
    (out_dir / f"{model_slug(label)}{suffix}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {csv_path}\n{json.dumps(meta['latency'], indent=2)}")
    if args.context:
        print(json.dumps(meta["context"]["room"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", help="HF model id (repeatable); default: scoring.models in config")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], help="override scoring.device")
    parser.add_argument("--malicious-index", type=int, help="output index meaning malicious, for models with generic labels")
    parser.add_argument("--limit", type=int, help="score a random subset (for smoke tests; output file gets a suffix)")
    parser.add_argument("--context", action="store_true", help="prepend the user task and tool name (scoring.context.template)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    default_models = cfg["scoring"]["context"]["models"] if args.context else cfg["scoring"]["models"]
    for name in args.model or default_models:
        print(f"Scoring {name}", flush=True)
        score_model(name, cfg, args)


if __name__ == "__main__":
    main()
