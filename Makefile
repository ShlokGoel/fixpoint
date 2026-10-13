PY ?= python
RUN ?= sub30
SUBSET ?= 30
SEED ?= 0
WORKERS ?= 2
# Three small, well-known Lite tasks for the smoke run; swap in any instance ids you like.
SMOKE_IDS ?= django__django-11099 psf__requests-2317 pytest-dev__pytest-5227

.PHONY: install test lint demo demo-docker smoke subset ablations full eval report compare

install:
	$(PY) -m pip install -e ".[eval,dev]"

test:
	$(PY) -m pytest

lint:
	ruff check . && ruff format --check .

demo:
	$(PY) scripts/demo_toy.py --scripted

demo-docker:
	docker build -f examples/Dockerfile.toy -t fixpoint-toy examples/
	$(PY) scripts/demo_toy.py --docker fixpoint-toy

smoke:
	$(PY) -m harness.run_agent --run-name smoke --instance-ids $(SMOKE_IDS)

subset:
	$(PY) -m harness.run_agent --run-name sub$(SUBSET) --subset $(SUBSET) --seed $(SEED) --workers $(WORKERS)

ablations:
	$(PY) -m harness.run_agent --run-name sub$(SUBSET)-noloc --subset $(SUBSET) --seed $(SEED) --workers $(WORKERS) \
		--set scaffold.localization=false
	$(PY) -m harness.run_agent --run-name sub$(SUBSET)-noverify --subset $(SUBSET) --seed $(SEED) --workers $(WORKERS) \
		--set scaffold.self_verification=false

full:
	$(PY) -m harness.run_agent --run-name full --workers $(WORKERS)

eval:
	$(PY) -m harness.evaluate --run-dir runs/$(RUN)

report:
	$(PY) -m analysis.report summarize runs/$(RUN)

compare:
	$(PY) -m analysis.report compare runs/$(A) runs/$(B)
