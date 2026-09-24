# Decisions

Judgment calls made while running unattended, each with a one-line rationale.

## Environment and models

- **Python 3.12 venv via uv, deps pinned with `uv pip freeze` into `requirements.txt`.** 3.14 (system default) is too new for a reliable torch/agentdojo stack.
- **Prompt Guard 2 is the primary model; `protectai/deberta-v3-base-prompt-injection-v2` is kept as an extra reference.** Access was initially denied (401 with no token, then 403 while the request was pending), so the stand-in was set up first; access was granted mid-run, and the stand-in costs little to keep as a comparison point.
- **Malicious class for Prompt Guard 2 is index 1, set explicitly in `config.yaml` (`malicious_index_overrides`).** The shipped `config.json` has generic `id2label = {0: LABEL_0, 1: LABEL_1}` while the model card's example prints `MALICIOUS` via `id2label[argmax]` for "Ignore your previous instructions.", which is index 1 on the downloaded weights; both 22M and 86M also give p(index 1) > 0.99 on three clear injections and < 0.002 on three benign sentences. `score.py` refuses to guess on generic labels without an override.
- **protectai's malicious class is resolved from `id2label` (`INJECTION` = index 1).** Its labels are descriptive, so no override is needed.
- **Both scoring tokenizers report no max length (`model_max_length` ≈ 1e30), so `score.py` always sets `max_length=512` explicitly.** Otherwise "truncation" would silently not truncate.

## InjecAgent

- **Pinned to commit `f19c9f2c`; all four test-case files (dh/ds × base/enhanced) are used, 2,108 positives.** They are the complete released test set.
- **Positive text is rebuilt as `template.replace("<Attacker Instruction>", inserted_text)` rather than taken from the `Tool Response` field.** That field is `'"' + template + '"'` with the template's own double quotes backslash-escaped but the attacker text left raw, so it is neither plain text nor valid JSON (68 rows fail `json.loads`, 124 have `\"` artifacts); the build verifies that the slot recovered from every raw response equals the attacker instruction (plus the enhanced prefix) and fails otherwise. 0 mismatches.
- **Negatives: 10 hand-written benign slot fillers per template (17 templates → 170), in `data/injecagent_benign_fillers.json`.** InjecAgent has only 17 distinct templates, so filling each test case would produce 17 unique texts repeated ~124 times; hand-writing context-appropriate fillers (a product review for review slots, an email body for email slots) gives more distinct negatives without any LLM API.
- **Filler lengths deliberately span the attacker-instruction range (28–294 chars) but end up slightly shorter on average (90 vs 112 tokens for the full output).** Part of the gap is the 20-token enhanced prefix; flagged in the report as a possible length cue.
- **`attack_category` for InjecAgent = its `Attack Type` field; `setting` = base/enhanced; `attack_goal` = direct_harm/data_stealing.** These are the paper's own axes.

## AgentDojo

- **pip `agentdojo==0.1.35` (= git tag `v0.1.35`, commit `a75aba76`), benchmark `v1.2.2`, all 4 suites.** Latest release and latest benchmark version at the time of the run.
- **Tool outputs come from replaying each user task's ground-truth calls with AgentDojo's own `GroundTruthPipeline`, no LLM.** This is the mechanism AgentDojo uses internally to find which injection vectors a task exposes, so the outputs are exactly what an agent following the correct plan would see.
- **Five attacks: `important_instructions`, `tool_knowledge`, `ignore_previous`, `injecagent`, `direct`.** These are all the non-DoS, pure-template attacks; `direct` ("TODO: {goal}") is a useful low-signal case with no jailbreak wording. The `important_instructions_*` name variants were skipped as near duplicates.
- **Attacks that address the model by name get `GPT-4` (from a placeholder pipeline named `gpt-4o-2024-05-13`).** Only fills the `{model}` slot in the template; the placeholder pipeline raises if anything tries to query it.
- **Label = attack iff the output contains the injection task's goal (compared with whitespace/quotes stripped, first 60 chars), cross-checked against a diff with the benign replay at the same step.** The first build had 1,155 disagreements; all were benign write-tool outputs whose `datetime.now()` timestamps differed between runs.
- **Wall-clock timestamps with microseconds are replaced by a constant `2024-05-20 12:00:00`.** The suites' own data never has microseconds, so only `now()` values change; this makes outputs reproducible and removed all label disagreements and ~timestamp-only duplicate negatives.
- **`attack_category` for AgentDojo = attack name; `suite` and `injection_task_id` kept as metadata.** The attack template is the property most relevant to a text classifier.
- **Negatives are all distinct benign tool outputs from benign replays (142 after dedup), with a `paired` flag marking the 49 that are the clean version of an output that gets injected.** Unpaired negatives are mostly short outputs (median 33 tokens, e.g. `send_money` confirmations) that never carry injections, a length/format confound, so the report also evaluates an `agentdojo_paired` slice (attacks vs paired negatives only).

## Dataset

- **Exact-text dedup within (source, label); `n_occurrences` records how many raw rows collapsed.** Many user tasks call the same tool with the same arguments, so without dedup a few outputs would dominate the metrics.
- **Hard negatives authored as YAML (`data/hard_negatives.yaml`, 40 items) and kept as their own slice.** Block scalars keep multi-line tool outputs readable; 3 `meta_discussion` items quote injection strings while discussing them (security blog, test fixture, phishing newsletter) and are reported separately since a reasonable person could argue those should be flagged.
- **Token-length stats in `build_dataset.py` use the protectai tokenizer; `score.py` records each model's own token count per example, and the ≤512 / >512 length breakdown uses the scored model's count.** Token counts differ across tokenizers.

## Baselines and evaluation

- **Two keyword baselines instead of one: `keyword_generic` and `keyword_template_aware`.** Several strings in the first draft (`<INFORMATION>`, "before you can solve the task", the `iunstructions` typo) were only known from reading these benchmarks' attack templates, so a single list would have overstated what a keyword filter does on unseen attacks.
- **Fixed a bug in the generic "send ... to <email/account>" regex after the first run.** It excluded `.` between "to" and `@`, so it never matched addresses like `amy.watson@gmail.com`; this was a regex bug rather than tuning toward the data, but it was found by looking at results, so noting it.
- **Precision is also reported at an assumed 1% attack prevalence.** The dataset is ~93% attacks, which makes raw precision near 1.0 for almost any detector.
- **Latency is timed on a second call per (example, mode).** MPS compiles kernels for each new input length, adding 10 to 100 ms the first time; the warm number is closer to steady-state serving cost.
- **Added `diagnose_isolated.py` (not in the original brief).** Scoring each injected string alone separates "the detector doesn't see this text as an attack" from "the surrounding data dilutes it", which is the main question behind "where does it fail".
- **Example lists use PG2-86M chunk mode.** It is the stronger PG2 model; 22M flags almost nothing, so its missed-attack list would be uninformative.
- **Injection start offsets for AgentDojo are found by diffing against the paired clean output, falling back to matching on (suite, tool, args)** because dedup keeps only one user task's id for outputs shared by several tasks. All 4,558 offsets were verified (goal text after the offset, absent before it).
