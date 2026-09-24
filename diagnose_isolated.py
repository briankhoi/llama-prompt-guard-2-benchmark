"""Diagnostic: score each unique injected string on its own, outside any tool output.

Separates "the detector doesn't consider this text malicious" from "the surrounding tool-output data dilutes it".
Writes results/isolated_injections.csv. Usage: python diagnose_isolated.py [--model ...]
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from pibench.common import ROOT, load_config, seed_everything
from pibench.injecagent import ENHANCED_PREFIX
from score import Scorer, pick_device


def injecagent_strings(cfg: dict) -> list[dict]:
    src = cfg["sources"]["injecagent"]
    repo = ROOT / cfg["paths"]["external_dir"] / "InjecAgent"
    out = {}
    for spec in src["files"]:
        for case in json.loads((repo / spec["path"]).read_text()):
            text = (ENHANCED_PREFIX if spec["setting"] == "enhanced" else "") + case["Attacker Instruction"]
            out[text] = {"source": "injecagent", "variant": f"injecagent_{spec['setting']}", "attack_category": case["Attack Type"], "text": text}
    return list(out.values())


def agentdojo_strings(cfg: dict) -> list[dict]:
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.task_suite.load_suites import get_suites

    from pibench.agentdojo_extract import _NamedPipeline

    src = cfg["sources"]["agentdojo"]
    suites = get_suites(src["benchmark_version"])
    pipeline = _NamedPipeline(src["pipeline_name_for_attacks"])
    out = {}
    for suite_name in src["suites"]:
        suite = suites[suite_name]
        user_task = next(iter(suite.user_tasks.values()))  # these templates don't depend on the user task
        for attack_name in src["attacks"]:
            attack = load_attack(attack_name, suite, pipeline)
            for it_id, it in suite.injection_tasks.items():
                for text in set(attack.attack(user_task, it).values()):
                    out[text] = {"source": "agentdojo", "variant": attack_name, "attack_category": attack_name, "suite": suite_name, "text": text}
    return list(out.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--model", action="append")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], help="override scoring.device")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    strings = pd.DataFrame(injecagent_strings(cfg) + agentdojo_strings(cfg))
    device = pick_device(args.device or cfg["scoring"]["device"])
    rows = []
    for name in args.model or cfg["scoring"]["models"]:
        scorer = Scorer(name, cfg, device, None)
        for r in strings.itertuples():
            res = scorer.score_text(r.text.strip(), "chunk")
            rows.append({**r._asdict(), "model": name, "score": res["score"], "n_tokens": res["n_tokens"]})
        del scorer
        if device.type == "mps":
            torch.mps.empty_cache()
    out = pd.DataFrame(rows).drop(columns=["Index"])
    path = ROOT / "results" / "isolated_injections.csv"
    out.to_csv(path, index=False)
    print(out.groupby(["model", "variant"])["score"].agg(["count", "median", lambda s: (s >= cfg["scoring"]["threshold"]).mean()]).round(3))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
