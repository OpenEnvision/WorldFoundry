.PHONY: help install-core install-dev test test-infer test-infer-tensors test-infer-contracts test-infer-cuda-contracts test-geometry test-eval-core docs-check docs-dev-fast docs-dev-ssd docs-dev-local docs-build-fast cli-entrypoint-check lint ruff-check format-check shell-check data-check runtime-registry-check workspace-registry-check check-cuda-constraints packaging-check compile-eval cli-check precommit precommit-install preflight
.PHONY: test-infer-video-tensors plan-inference replay-inference coverage-inference verify-inference

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PRE_COMMIT ?= $(PYTHON) -m pre_commit
PYTHONPATH ?= .
WORLDFOUNDRY_EVAL ?= $(PYTHON) -m worldfoundry.cli
PREFLIGHT_PROFILE ?= all
PREFLIGHT_OUTPUT ?= tmp/preflight
CLI_CHECK_OUTPUT ?= tmp/ci-cli-check
TEST_ARGS ?=
VIDEO_TENSOR_CONTRACTS = tests/core/test_video_tensor_regression.py tests/core/test_causal_video_cache.py tests/base_models/diffusion_model/optimizations/test_static_cross_kv.py
FLASHDREAMS_CPU_CONTRACTS = tests/base_models/test_flashdreams_sana_wm_streaming.py tests/base_models/test_flashdreams_fastvideo_causal_wan.py tests/base_models/diffusion_model/test_causal_attention_padding.py tests/runtime/test_inference_benchmark_correctness.py
INFER_TENSOR_CONTRACTS = tests/core/execution tests/core/model_loading/test_checkpoint_roundtrip.py tests/core/geometry/test_geometry_conventions.py tests/runtime/test_geometry_regression.py tests/synthesis/test_dreamx_world_backend_policy.py $(VIDEO_TENSOR_CONTRACTS) $(FLASHDREAMS_CPU_CONTRACTS)
# Select numerical CUDA nodes explicitly: older CUDA tests do not all carry the gpu marker.
INFER_CUDA_CONTRACTS = \
	tests/core/execution/test_cuda_frame_transfer.py::test_cuda_host_pixels_wait_for_producer_and_preserve_frame_order \
	tests/core/execution/test_cuda_frame_transfer.py::test_cuda_prefetch_failure_uses_the_correct_blocking_fallback \
	tests/core/execution/test_cuda_frame_transfer.py::test_failed_cuda_callback_drains_queued_work_before_reuse \
	tests/core/execution/test_inference_graph_cuda.py \
	tests/core/test_diffusion_mutating_kernel.py::test_bf16_mutating_kernel_preserves_alias_version_and_eager_bits \
	tests/core/attention/test_kv_arena.py::test_cuda_long_sequence_attention_and_stream_ownership \
	tests/core/attention/test_kv_arena.py::test_cuda_graph_replay_reads_updated_current_segment \
	tests/core/attention/test_gqa_storage.py::test_efficient_cuda_gqa_matches_math \
	tests/runtime/test_inference_benchmark_correctness.py::test_denoise_benchmark_checks_every_step_before_timing \
	tests/runtime/test_inference_benchmark_correctness.py::test_full_stack_actual_fp8_cache_or_graph_execution
INFER_CUDA_REPORT ?= tmp/inference-cuda-contracts.xml
GEOMETRY_MATRIX ?= tests/manual/geometry_regression_cases.json
GEOMETRY_REFERENCE ?=
GEOMETRY_CANDIDATE ?=
GEOMETRY_REPORT ?= tmp/3d-regression-gate.json
GEOMETRY_BASE ?=
GEOMETRY_DEPENDENCIES ?= tests/manual/geometry_regression_dependencies.json
GEOMETRY_PLAN ?= tmp/3d-regression-plan.json
GEOMETRY_PROFILE ?=
INFERENCE_BASE ?= $(GEOMETRY_BASE)
INFERENCE_PLAN ?= tmp/inference-regression-plan.json
INFERENCE_PROFILE ?= $(GEOMETRY_PROFILE)
INFERENCE_REPORT ?=
INFERENCE_COVERAGE ?= tmp/inference-regression-coverage.json
INFERENCE_GATE_OUTPUT ?= tmp/inference-regression-gate.json
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
		'  make test-infer-tensors  Test small checkpoint, geometry and execution tensors with CPU Torch and safetensors.' \
		'  make test-infer-video-tensors  Check real small video operators, sampler math and resident request isolation with CPU Torch and einops.' \
		'  make test-infer-contracts  Test checkpoint, geometry, serializer and execution contracts in a model environment.' \
		'  make test-infer-cuda-contracts  Run real CUDA graph, cache, attention, kernel and transfer checks; rejects empty/skipped suites and requires CUDA Torch and Triton.' \
		'  make test-geometry     Audit real 3D replays against accepted references; requires GEOMETRY_REFERENCE and GEOMETRY_CANDIDATE.' \
		'  make plan-geometry     Select affected short 3D cases from GEOMETRY_BASE to HEAD without GPU/weights.' \
		'  make replay-geometry   Replay selected or all cases using a private GEOMETRY_PROFILE; optional GEOMETRY_REPLAY_PLAN.' \
		'  make plan-inference    Select affected 3D/video/world cases using INFERENCE_BASE.' \
		'  make replay-inference  Replay affected cases using INFERENCE_PROFILE and INFERENCE_PLAN.' \
		'  make coverage-inference  List short-case definitions and missing video/world variants.' \
		'  make verify-inference  Verify exact committed GPU evidence using INFERENCE_PLAN, INFERENCE_REPORT and INFERENCE_PROFILE.' \
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

# This suite needs CPU Torch, safetensors and einops, with no weights or renderer.
test-infer-tensors:
	CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q -m 'not gpu' $(INFER_TENSOR_CONTRACTS) $(TEST_ARGS)

test-infer-video-tensors:
	CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q -m 'not gpu' $(VIDEO_TENSOR_CONTRACTS) $(TEST_ARGS)

plan-inference:
	@test -n "$(INFERENCE_BASE)" || (echo 'Set INFERENCE_BASE to the comparison commit.'; exit 2)
	$(PYTHON) tests/manual/geometry_regression_impact.py --matrix "$(GEOMETRY_MATRIX)" --dependencies "$(GEOMETRY_DEPENDENCIES)" --base "$(INFERENCE_BASE)" --output "$(INFERENCE_PLAN)"

replay-inference:
	@test -n "$(INFERENCE_PROFILE)" || (echo 'Set INFERENCE_PROFILE to a private prepared model-host profile.'; exit 2)
	$(PYTHON) tests/manual/geometry_regression_suite.py --profile "$(INFERENCE_PROFILE)" --plan "$(INFERENCE_PLAN)"

coverage-inference:
	$(PYTHON) tests/manual/inference_regression_coverage.py --matrix "$(GEOMETRY_MATRIX)" --output "$(INFERENCE_COVERAGE)"

verify-inference:
	@test -n "$(INFERENCE_PROFILE)" -a -n "$(INFERENCE_REPORT)" || (echo 'Set INFERENCE_PROFILE and INFERENCE_REPORT.'; exit 2)
	$(PYTHON) tests/manual/inference_regression_gate.py --plan "$(INFERENCE_PLAN)" --report "$(INFERENCE_REPORT)" --profile "$(INFERENCE_PROFILE)" --output "$(INFERENCE_GATE_OUTPUT)"

# Run inside the 3D model environment; tensor and serializer tests need its dependencies.
test-infer-contracts:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q -m 'not gpu' $(INFER_TENSOR_CONTRACTS) tests/pipelines/test_geometry_result_exports.py $(TEST_ARGS)

# Run manually in a prepared CUDA/Triton environment; hosted CI has no GPU runner.
# An unavailable GPU, empty selection or skipped contract must fail this gate.
test-infer-cuda-contracts:
	$(PYTHON) -c 'import sys, torch; torch.cuda.is_available() or sys.exit("CUDA contract tests require an available GPU")'
	mkdir -p "$(dir $(INFER_CUDA_REPORT))"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q --strict-markers --junitxml="$(INFER_CUDA_REPORT)" $(INFER_CUDA_CONTRACTS) $(TEST_ARGS)
	$(PYTHON) -c 'import sys, xml.etree.ElementTree as ET; cases = list(ET.parse(sys.argv[1]).getroot().iter("testcase")); cases or sys.exit("CUDA gate executed no tests"); rejected = [case.get("name", "unknown") for case in cases if any(case.find(tag) is not None for tag in ("skipped", "failure", "error"))]; rejected and sys.exit("CUDA gate requires every selected test to pass: " + ", ".join(rejected)); print(f"CUDA gate: {len(cases)} tests passed without skips")' "$(INFER_CUDA_REPORT)"

test-geometry:
	@test -n "$(GEOMETRY_REFERENCE)" -a -n "$(GEOMETRY_CANDIDATE)" || { printf '%s\n' 'Set GEOMETRY_REFERENCE and GEOMETRY_CANDIDATE to completed real-checkpoint run directories.' >&2; exit 2; }
	$(PYTHON) tests/manual/geometry_regression.py audit --matrix "$(GEOMETRY_MATRIX)" --reference "$(GEOMETRY_REFERENCE)" --candidate "$(GEOMETRY_CANDIDATE)" --source-root "$(CURDIR)" --report "$(GEOMETRY_REPORT)"

.PHONY: plan-geometry replay-geometry
plan-geometry:
	@test -n "$(GEOMETRY_BASE)" || { printf '%s\n' 'Set GEOMETRY_BASE to the comparison commit or ref.' >&2; exit 2; }
	$(PYTHON) tests/manual/geometry_regression_impact.py --matrix "$(GEOMETRY_MATRIX)" --dependencies "$(GEOMETRY_DEPENDENCIES)" --source-root "$(CURDIR)" --base "$(GEOMETRY_BASE)" --output "$(GEOMETRY_PLAN)"

replay-geometry:
	@test -n "$(GEOMETRY_PROFILE)" || { printf '%s\n' 'Set GEOMETRY_PROFILE to the private host profile.' >&2; exit 2; }
	$(PYTHON) tests/manual/geometry_regression_suite.py --profile "$(GEOMETRY_PROFILE)" $(if $(GEOMETRY_REPLAY_PLAN),--plan "$(GEOMETRY_REPLAY_PLAN)",)

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
