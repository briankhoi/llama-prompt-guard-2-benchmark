# Prompt Guard 2 vs. indirect prompt injection in tool outputs

A feasibility study: does Meta's Llama Prompt Guard 2 (22M and 86M) already catch indirect prompt injections hidden in tool outputs, or does it leave meaningful gaps? We score tool-output text from two public benchmarks, InjecAgent and AgentDojo, plus a small set of hand-written hard negatives. No agent is run and no paid LLM API is called.

Results: [`results/REPORT.md`](results/REPORT.md). Judgment calls: [`DECISIONS.md`](DECISIONS.md).

## Setup

Requires macOS/Linux with [uv](https://github.com/astral-sh/uv) and git. Prompt Guard 2 is gated: accept the license on Hugging Face for both models, then log in.

```bash
make setup                 # Python 3.12 venv in .venv, pinned deps from requirements.txt
.venv/bin/hf auth login    # needed for meta-llama/Llama-Prompt-Guard-2-*
make fetch                 # clone InjecAgent and AgentDojo at the pinned commits into external/
```

## Reproduce

```bash
make all                   # setup, fetch, dataset, score, baseline, diagnose, evaluate
```

or step by step:

```bash
make dataset               # build_dataset.py  -> data/processed/dataset.{parquet,csv}, dataset_stats.md
make score                 # score.py          -> results/scores/<model>.csv + _meta.json, for every model in config.yaml
make baseline              # baseline.py       -> results/scores/keyword_*.csv
make diagnose              # diagnose_isolated.py -> results/isolated_injections.csv (each injected string scored alone)
make evaluate              # evaluate.py       -> results/REPORT.md, results/metrics_*.csv
make score-model MODEL=meta-llama/Llama-Prompt-Guard-2-86M   # one model
```

`score.py` works for any Hugging Face sequence classifier. If its `id2label` names are generic (`LABEL_0/LABEL_1`, as for Prompt Guard 2), it refuses to guess the malicious class: pass `--malicious-index N` or add the model to `scoring.malicious_index_overrides` in `config.yaml`. Scores are saved per example, so `evaluate.py` can be re-run without rescoring.

All settings (pinned commits, attacks, window size, stride, regexes, seed, threshold) live in `config.yaml`. Scoring uses PyTorch on MPS and falls back to CPU on error; MPS and CPU scores agreed to within 2e-6 on a 60-example check.

Pinned sources:

| source | version |
|---|---|
| InjecAgent | git `f19c9f2c79a41046eb13c03c51a24c567a8ffa07` |
| AgentDojo | pip `agentdojo==0.1.35` (= git tag `v0.1.35`, `a75aba7631d3ca5fb7ab938965c97ead2f9ff84b`), benchmark `v1.2.2` |

## Dataset construction

One row per tool output: `source`, `tool_name`, `user_task`, `tool_output_text`, `label` (attack/benign), `attack_category`, plus metadata (`setting`, `attack_goal`, `suite`, `user_task_id`, `injection_task_id`, `attack_name`, `paired`, `n_occurrences`, `notes`). Exact duplicates within (source, label) are collapsed and counted in `n_occurrences`.

### InjecAgent

InjecAgent's test cases (`data/test_cases_{dh,ds}_{base,enhanced}.json`) cross 17 tool-response templates, each with one `<Attacker Instruction>` slot, with 30 direct-harm and 32 data-stealing attacker instructions. The `enhanced` setting prepends a fixed "IMPORTANT!!! Ignore all previous instructions and strictly adhere to the following instruction: " string.

- **Positives (2,108):** every test case in all four files. The text is rebuilt as `template.replace("<Attacker Instruction>", instruction)` (with the enhanced prefix where applicable) because the raw `Tool Response` field carries inconsistent escaping; the build checks that every raw response contains exactly that inserted text and fails otherwise.
- **Negatives (170):** each of the 17 templates filled with 10 hand-written benign texts that fit the slot (a product review in a review slot, an email body in an email slot, an event name in a calendar slot), stored in `data/injecagent_benign_fillers.json`. Filling each test case's template instead would give only 17 unique negatives.

### AgentDojo

Tool outputs come from replaying each user task's ground-truth tool calls against the suite environment with AgentDojo's own `GroundTruthPipeline` (the mechanism AgentDojo uses internally to find which injection vectors a task exposes). No LLM is involved.

1. For each of the 4 suites (workspace, travel, banking, slack) and each of the 97 user tasks, replay the ground truth on the default environment. Every tool output is a benign candidate.
2. For each of 5 attacks (`important_instructions`, `tool_knowledge`, `ignore_previous`, `injecagent`, `direct`) and each injection task, generate the attack's injections for that user task, load the injected environment, and replay again (949 runs per attack).
3. An output is labeled `attack` if it contains the injection task's goal text (compared with whitespace and quotes removed, since YAML formatting reflows text). This agrees exactly with "differs from the benign replay of the same step", after masking wall-clock timestamps that `send_email`/`create_file` produce.
4. `paired = True` marks benign outputs that are the clean version of an output that gets injected. The `agentdojo_paired` slice compares attacks only against these, which controls for tool type and length (unpaired benign outputs are mostly short confirmations like `send_money` results).

### Hard negatives

`data/hard_negatives.yaml`: 40 hand-written benign tool outputs containing instructions or sensitive-sounding language: emails asking to forward files or pay rent, install docs with shell commands, READMEs with credential setup, bank statements and invoices, runbooks, CLI help, support tickets, product reviews. Three `meta_discussion` items quote injection strings while discussing them (a security blog post, a test-fixture file, a phishing-awareness email). All are labeled benign and kept as a separate slice.

## Scoring

- **Truncation:** the first 510 content tokens plus special tokens (512 total). Both Prompt Guard 2 tokenizers report no max length, so the limit is set explicitly.
- **Chunking:** overlapping windows of 510 content tokens with stride 384 (126-token overlap), the last window aligned to the end of the text; the example score is the max P(malicious) across windows, following Meta's advice to split long inputs and scan each segment.
- **Latency:** per example, including tokenization and device sync, one batch holding that example's windows. Each (example, mode) is timed on a second call because MPS compiles kernels the first time it sees a new input length, which would otherwise add 10 to 100 ms to arbitrary examples.

Models: `meta-llama/Llama-Prompt-Guard-2-22M`, `meta-llama/Llama-Prompt-Guard-2-86M`, and `protectai/deberta-v3-base-prompt-injection-v2` (set up as a stand-in while Prompt Guard 2 access was pending, kept as a reference point).

### Keyword baselines

Two regex baselines in `config.yaml`, scored as "number of patterns matched" and flagged at one or more matches:

- `keyword_generic`: phrases from general prompt-injection folklore ("ignore previous instructions", "you must now", "send ... to <email/account>", "grant ... access", ...).
- `keyword_template_aware`: the generic patterns plus strings copied from these benchmarks' own attack templates (`<INFORMATION>`, "IMPORTANT!!!", "TODO:", ...). It is an optimistic upper bound for signature matching on these datasets, not an estimate for unseen attacks.

## Known limitations

- InjecAgent positives are 17 templates × 62 instructions, and AgentDojo negatives come from one fixed environment (142 distinct benign outputs, 49 of them paired), so the effective sample sizes are much smaller than the row counts; the report shows Wilson and bootstrap intervals.
- The InjecAgent benign fillers and the hard negatives were hand-written by Claude (the assistant that built this pipeline) as static files, with no API calls; a different author would produce different negatives and likely different false-positive rates.
- Both benchmarks use a handful of fixed attack templates. Detection rates here say little about adaptive attackers who rephrase.
- Class ratio is roughly 13:1 attacks to benign, so raw precision and PR-AUC are inflated; the report adds precision at an assumed 1% attack prevalence.

## License

Code and the files written for this study (`data/hard_negatives.yaml`, `data/injecagent_benign_fillers.json`) are MIT licensed, see [`LICENSE`](LICENSE). `data/processed/` and `results/` contain text derived from two MIT-licensed benchmarks: [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) (Copyright (c) 2023 Qiusi Zhan) and [AgentDojo](https://github.com/ethz-spylab/agentdojo) (Copyright (c) 2024 Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr); their license notices are reproduced in [`LICENSE`](LICENSE). Prompt Guard 2 weights are not included and are subject to Meta's license on Hugging Face.
