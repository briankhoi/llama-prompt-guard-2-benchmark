"""Extract AgentDojo tool outputs without an LLM by replaying each user task's ground-truth tool calls.

For each user task we replay the ground truth once on the default environment (benign) and once per
(attack, injection task) on the environment with that attack's injections applied. This is the same
mechanism AgentDojo's own `BaseAttack.get_injection_candidates` uses (GroundTruthPipeline).
"""

from __future__ import annotations

import json
import re
from collections import Counter

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.types import get_text_content_as_str

_NORMALIZE_RE = re.compile(r"[\s'\"\\]+")
# Write tools (send_email, create_file, ...) stamp datetime.now() with microseconds; the suites' own data never has
# microseconds, so this only matches wall-clock values. Replace them with a constant so outputs are reproducible.
_WALLCLOCK_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\.\d{6}")
FROZEN_TIMESTAMP = "2024-05-20 12:00:00"


class _NamedPipeline(BasePipelineElement):
    """Placeholder target pipeline: attacks only read its `name` to fill the {model} slot, it is never queried."""

    def __init__(self, name: str) -> None:
        self.name = name

    def query(self, *args, **kwargs):
        raise RuntimeError("This placeholder pipeline must never be queried, no LLM calls are made")


def _normalize(text: str) -> str:
    """Drop whitespace, quotes and backslashes so YAML line folding/quoting doesn't hide an injected string."""
    return _NORMALIZE_RE.sub("", text).lower()


def _replay(suite, user_task, injections: dict[str, str]) -> list[tuple[str, dict, str]]:
    env = suite.load_and_inject_default_environment(injections)
    runtime = FunctionsRuntime(suite.tools)
    _, _, _, messages, _ = GroundTruthPipeline(user_task).query(user_task.PROMPT, runtime, env)
    return [
        (m["tool_call"].function, dict(m["tool_call"].args), _WALLCLOCK_RE.sub(FROZEN_TIMESTAMP, get_text_content_as_str(m["content"])))
        for m in messages
        if m["role"] == "tool"
    ]


def _row(suite_name, user_task, fn, args, text, label, **extra) -> dict:
    return {
        "source": "agentdojo",
        "slice": "agentdojo",
        "tool_name": fn,
        "user_task": user_task.PROMPT,
        "tool_output_text": text,
        "label": label,
        "suite": suite_name,
        "user_task_id": user_task.ID,
        "notes": "args=" + json.dumps(args, default=str)[:300],
        **extra,
    }


def _benign_rows(suite_name, suite, stats) -> tuple[dict, dict]:
    """Returns benign rows keyed by (user_task_id, step) so injected steps can be marked as paired later."""
    rows, benign_runs = {}, {}
    for ut_id, ut in suite.user_tasks.items():
        try:
            outputs = _replay(suite, ut, {})
        except Exception as exc:  # a broken ground truth should not stop the whole extraction
            stats["benign_replay_errors"].append(f"{suite_name}/{ut_id}: {type(exc).__name__}: {exc}")
            continue
        benign_runs[ut_id] = outputs
        for step, (fn, args, text) in enumerate(outputs):
            rows[(ut_id, step)] = _row(suite_name, ut, fn, args, text, "benign", attack_category="none", paired=False)
    return rows, benign_runs


def _label_attacked_output(step, fn, text, benign_outputs, goal_marker, stats) -> str:
    """Attack iff the output contains the injection goal; cross-check against the benign replay at the same step."""
    contains_goal = goal_marker in _normalize(text)
    same_step = step < len(benign_outputs) and benign_outputs[step][0] == fn
    differs = (not same_step) or benign_outputs[step][2] != text
    if contains_goal != differs:
        stats["label_disagreements"] += 1
    return "attack" if contains_goal else "benign"


def _attack_rows(suite_name, suite, attack_name, pipeline, benign_runs, stats) -> tuple[list[dict], set]:
    """Returns attacked-run rows and the (user_task_id, step) keys whose output carried an injection."""
    attack = load_attack(attack_name, suite, pipeline)
    rows, injected_steps = [], set()
    for ut_id, ut in suite.user_tasks.items():
        if ut_id not in benign_runs:
            continue
        for it_id, it in suite.injection_tasks.items():
            try:
                injections = attack.attack(ut, it)
                outputs = _replay(suite, ut, injections)
            except Exception as exc:
                stats["attack_replay_errors"].append(f"{suite_name}/{attack_name}/{ut_id}/{it_id}: {type(exc).__name__}: {exc}")
                continue
            goal_marker = _normalize(it.GOAL)[:60]
            for step, (fn, args, text) in enumerate(outputs):
                label = _label_attacked_output(step, fn, text, benign_runs[ut_id], goal_marker, stats)
                category = attack_name if label == "attack" else "none"
                if label == "attack":
                    injected_steps.add((ut_id, step))
                rows.append(
                    _row(
                        suite_name, ut, fn, args, text, label,
                        attack_category=category,
                        attack_name=attack_name if label == "attack" else None,
                        injection_task_id=it_id if label == "attack" else None,
                        paired=label == "attack",
                    )
                )
            stats["runs"][attack_name] += 1
    return rows, injected_steps


def extract(cfg: dict) -> tuple[list[dict], dict]:
    suites = get_suites(cfg["benchmark_version"])
    pipeline = _NamedPipeline(cfg["pipeline_name_for_attacks"])
    stats = {
        "benign_replay_errors": [],
        "attack_replay_errors": [],
        "label_disagreements": 0,
        "runs": Counter(),
    }
    rows: list[dict] = []
    for suite_name in cfg["suites"]:
        suite = suites[suite_name]
        benign, benign_runs = _benign_rows(suite_name, suite, stats)
        injected_steps = set()
        for attack_name in cfg["attacks"]:
            attack_rows, steps = _attack_rows(suite_name, suite, attack_name, pipeline, benign_runs, stats)
            rows.extend(attack_rows)
            injected_steps |= steps
        # `paired` marks the clean version of an output that carries an injection under attack (a length/format-matched negative).
        rows.extend({**row, "paired": key in injected_steps} for key, row in benign.items())
        print(f"[agentdojo] {suite_name}: {len(rows)} raw rows so far", flush=True)
    stats["runs"] = dict(stats["runs"])
    return rows, stats
