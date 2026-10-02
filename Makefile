.PHONY: help install-core install-dev test test-infer test-eval-core docs-check docs-dev-fast docs-dev-ssd docs-dev-local docs-build-fast cli-entrypoint-check lint ruff-check format-check shell-check data-check runtime-registry-check workspace-registry-check check-cuda-constraints packaging-check compile-eval cli-check precommit precommit-install preflight

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PRE_COMMIT ?= $(PYTHON) -m pre_commit
PYTHONPATH ?= .
WORLDFOUNDRY_EVAL ?= $(PYTHON) -m worldfoundry.cli
PREFLIGHT_PROFILE ?= all
PREFLIGHT_OUTPUT ?= tmp/preflight
CLI_CHECK_OUTPUT ?= tmp/ci-cli-check
TEST_ARGS ?=
RELEASE_HFD_ROOT ?= $(if $(WORLDFOUNDRY_HFD_ROOT),$(WORLDFOUNDRY_HFD_ROOT),$(HOME)/.cache/worldfoundry/checkpoints/hfd)
CANONICAL_DIFFUSION_SOURCES ?= \
	worldfoundry/base_models/diffusion_model/*.py \
	worldfoundry/base_models/diffusion_model/extensions \
	worldfoundry/base_models/diffusion_model/loaders \
	worldfoundry/base_models/diffusion_model/models \
	worldfoundry/base_models/diffusion_model/optimizations \
	worldfoundry/base_models/diffusion_model/recipes \
	worldfoundry/base_models/diffusion_model/runners \
	worldfoundry/base_models/diffusion_model/schedulers
RUFF_SOURCES ?= \
	worldfoundry/cli \
	worldfoundry/evaluation/api \
	worldfoundry/evaluation/models/runtime \
	worldfoundry/evaluation/tasks/catalog \
	worldfoundry/evaluation/tasks/execution/orchestration \
	worldfoundry/mcp \
	worldfoundry/runtime \
	scripts/model_zoo

help:
	@printf '%s\n' \
		'WorldFoundry development targets:' \
		'  make install-core      Install the editable core package.' \
		'  make install-dev       Install lightweight development dependencies.' \
		'  make test              Run the public CPU inference and packaging gate.' \
		'  make test-infer        Alias for the public CPU gate.' \
		'  make test-eval-core    Run the extended evaluation contract suite.' \
		'  make docs-check        Verify checked-in generated documentation.' \
		'  make docs-dev-fast     Start docs using existing generated output.' \
		'  make docs-dev-ssd      Start docs with caches on local SSD.' \
		'  make docs-dev-local    Start docs from a local SSD mirror.' \
		'  make docs-build-fast   Build docs without the CI validation gates.' \
		'  make cli-entrypoint-check Validate documented CLI entrypoints.' \
		'  make lint              Run lightweight source and catalog checks.' \
		'  make preflight         Run the public runtime preflight.' \
		'  make check-cuda-constraints  Verify CUDA-tier torch constraint stubs.' \
		'  make packaging-check   Audit package discovery and license-gated wheel content.'

install-core:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q $(TEST_ARGS)

test-infer: test

test-eval-core:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests/eval_core $(TEST_ARGS)

docs-check:
	npm --prefix docs/fumadocs run api:check
	npm --prefix docs/fumadocs run models:check
	npm --prefix docs/fumadocs run models:homes:check
	npm --prefix docs/fumadocs run benchmarks:check
	npm --prefix docs/fumadocs run coverage:check

docs-dev-fast:
	npm --prefix docs/fumadocs run dev:fast

docs-dev-ssd:
	npm --prefix docs/fumadocs run dev:ssd

docs-dev-local:
	npm --prefix docs/fumadocs run dev:local

docs-build-fast:
	npm --prefix docs/fumadocs run build:fast

cli-entrypoint-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli --help >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo models --json >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo benchmarks --json >/dev/null

lint: ruff-check format-check shell-check data-check runtime-registry-check workspace-registry-check

ruff-check:
	$(PYTHON) -m ruff check $(RUFF_SOURCES)

format-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q $(CANONICAL_DIFFUSION_SOURCES) worldfoundry/evaluation scripts

# BeeGFS: enumerate the bounded tracked paths from Git's index, not a recursive filesystem walk.
shell-check:
	@set -eu; \
	git ls-files 'scripts/setup/*.sh' 'scripts/dev/*.sh' 'docs/fumadocs/scripts/*.sh' | \
	while IFS= read -r script; do \
		if [ -f "$$script" ]; then bash -n "$$script"; fi; \
	done

data-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo models --json >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo benchmarks --json >/dev/null

runtime-registry-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c 'from worldfoundry.evaluation.models.runtime.validate import validate_runtime_registry; errors = [issue for issue in validate_runtime_registry() if issue.severity == "error"]; assert not errors, "\\n".join(f"[{issue.code}] {issue.message}" for issue in errors)'

workspace-registry-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c 'from worldfoundry.evaluation.tasks.catalog.integrity import catalog_runner_table_issues, video_catalog_dispatch_issues; from worldfoundry.evaluation.tasks.catalog.workspace_registry import validate_workspace_registry; issues = [*validate_workspace_registry(), *video_catalog_dispatch_issues(), *catalog_runner_table_issues()]; assert not issues, "\n".join(issues); print("workspace registry OK")'

check-cuda-constraints:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/setup/check_cuda_torch_constraints.py

# Set WHEEL=... and/or SDIST=... to audit built distribution contents.
packaging-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/setup/check_packaging_license_gate.py $(if $(WHEEL),--wheel $(WHEEL),) $(if $(SDIST),--sdist $(SDIST),)

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
