# End-to-end: `make all` (setup, fetch pinned sources, build dataset, score, baseline, evaluate).
PY := .venv/bin/python
INJECAGENT_COMMIT := f19c9f2c79a41046eb13c03c51a24c567a8ffa07
AGENTDOJO_COMMIT := a75aba7631d3ca5fb7ab938965c97ead2f9ff84b
MODEL ?=

.PHONY: all setup fetch dataset score score-model baseline diagnose evaluate

all: setup fetch dataset score baseline diagnose evaluate

setup:
	[ -d .venv ] || uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt

# AgentDojo data is read from the pip package (pinned in requirements.txt); the clone is only for reading source at the same tag.
fetch:
	mkdir -p external
	[ -d external/InjecAgent ] || git clone https://github.com/uiuc-kang-lab/InjecAgent external/InjecAgent
	git -C external/InjecAgent checkout -q $(INJECAGENT_COMMIT)
	[ -d external/agentdojo ] || git clone https://github.com/ethz-spylab/agentdojo external/agentdojo
	git -C external/agentdojo checkout -q $(AGENTDOJO_COMMIT)

dataset:
	$(PY) build_dataset.py

# Scores every model listed under scoring.models in config.yaml.
score:
	$(PY) score.py

# One model: make score-model MODEL=meta-llama/Llama-Prompt-Guard-2-86M
score-model:
	@test -n "$(MODEL)" || (echo "usage: make score-model MODEL=<hf id>" && exit 1)
	$(PY) score.py --model $(MODEL)

baseline:
	$(PY) baseline.py

# Scores each unique injected string on its own, for the isolated-vs-embedded table.
diagnose:
	$(PY) diagnose_isolated.py

evaluate:
	$(PY) evaluate.py
