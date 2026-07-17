.PHONY: help install-core install-dev test-fast test-eval-core test-ux docs-check lint ruff-check format-check shell-check data-check runtime-registry-check compile-eval cli-check precommit precommit-install preflight

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
PRE_COMMIT ?= $(PYTHON) -m pre_commit
PYTHONPATH ?= .
WORLDFOUNDRY_EVAL ?= $(PYTHON) -m worldfoundry.cli
PREFLIGHT_PROFILE ?= all
PREFLIGHT_OUTPUT ?= tmp/preflight
CLI_CHECK_OUTPUT ?= tmp/ci-cli-check
RELEASE_HFD_ROOT ?= $(if $(WORLDFOUNDRY_HFD_ROOT),$(WORLDFOUNDRY_HFD_ROOT),$(HOME)/.cache/worldfoundry/checkpoints/hfd)
EVAL_CORE_CHECK_TESTS ?= \
	test/eval_core/test_api_contracts.py \
	test/eval_core/test_metric_registry.py \
	test/eval_core/test_task_yaml.py \
	test/eval_core/test_run_manifest.py \
	test/eval_core/test_public_namespace.py \
	test/eval_core/test_scorecard_snapshot.py \
	test/eval_core/test_contract_stability.py

help:
	@printf '%s\n' \
		'WorldFoundry development targets:' \
		'  make install-core      Install the editable core package.' \
		'  make install-dev       Install lightweight development dependencies.' \
		'  make test-fast         Run fast evaluation and CLI checks.' \
		'  make docs-check        Validate documented CLI entrypoints.' \
		'  make lint              Run lightweight source and catalog checks.' \
		'  make preflight         Run the public runtime preflight.'

install-core:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .
	$(PIP) install build pre-commit pytest PyYAML ruff

test-fast: test-eval-core test-ux docs-check

test-eval-core:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) -m fast_eval_core $(EVAL_CORE_CHECK_TESTS)

test-ux:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) -q \
		test/eval_core/test_cli_ux.py \
		test/eval_core/test_catalog_discovery_output.py \
		test/eval_core/test_docs_quickstart.py

docs-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli --help >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo benchmarks --json >/dev/null

lint: format-check shell-check data-check runtime-registry-check

ruff-check:
	@if $(PYTHON) -c 'import ruff' >/dev/null 2>&1; then \
		$(PYTHON) -m ruff check worldfoundry/cli worldfoundry/evaluation worldfoundry/mcp worldfoundry/runtime scripts/benchmark_zoo scripts/model_zoo test/eval_core; \
	else \
		printf '%s\n' 'ruff-check: skipped because ruff is not installed; run `make install-dev` first'; \
	fi

format-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q worldfoundry/evaluation scripts test/eval_core

shell-check:
	find scripts/setup -type f -name '*.sh' -exec bash -n {} +

data-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo models --json >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo benchmarks --json >/dev/null

runtime-registry-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c 'from worldfoundry.evaluation.models.runtime.validate import validate_runtime_registry; errors = [issue for issue in validate_runtime_registry() if issue.severity == "error"]; assert not errors, "\\n".join(f"[{issue.code}] {issue.message}" for issue in errors)'

compile-eval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q worldfoundry/evaluation scripts

cli-check:
	rm -rf $(CLI_CHECK_OUTPUT)
	mkdir -p $(CLI_CHECK_OUTPUT)/input
	printf '%s\n' '{"sample_id":"ci-0001","status":"success","artifacts":{"video":{"uri":"$(CLI_CHECK_OUTPUT)/input/demo.mp4","kind":"video"}}}' > $(CLI_CHECK_OUTPUT)/input/results.jsonl
	: > $(CLI_CHECK_OUTPUT)/input/demo.mp4
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli evaluate \
		--mode existing-results \
		--results-path $(CLI_CHECK_OUTPUT)/input/results.jsonl \
		--output-dir $(CLI_CHECK_OUTPUT)/run \
		--benchmark-id ci-existing-results \
		--model-id ci-package-check \
		--metric artifact_count \
		--required-artifact video \
		--json

precommit:
	$(PRE_COMMIT) run -a

precommit-install:
	$(PRE_COMMIT) install

preflight:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli preflight runtime \
		--profile $(PREFLIGHT_PROFILE) \
		--output-dir $(PREFLIGHT_OUTPUT) \
		--json
