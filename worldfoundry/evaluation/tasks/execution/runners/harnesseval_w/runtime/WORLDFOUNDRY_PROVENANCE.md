# HarnessEval-W evaluator provenance

- Source: https://github.com/MirroS-Lab/HarnessEval-W
- Revision: `ed4ccc6486b8271723ee8baea60d89b32d0a7518`
- Upstream location: `src/harnesseval/`
- Upstream README declares Apache License 2.0. No standalone LICENSE file is present at this revision.

Only the reachable evaluation pipeline is retained: inventory, skill registry,
eleven skill evaluators/backends, metric preprocessing, evidence caching, scoring,
report aggregation and completion validation. Generation adapters, CLI/shell
launchers, dependency installers, training, demos, videos, outputs, reference-only
formula implementations and automatic LLM plan generation are excluded. Uncalled
legacy aggregate functions and the standalone video-quality evaluator are also
omitted; the skill backend directly caches the same metric instances. Official
skill plans are required inputs; the two manifest readers from `pipeline/planner.py`
are retained. `report.py` omits CSV/HTML publishing; `validation.py` omits the
reference-document parser. The selected-core and observation scoring formulas,
normalization constants, prompts, frame sampling and common-case aggregation are
preserved.

WorldFoundry adaptations:

- `skills/common.py` requires an explicit cache instead of scanning upstream runs.
- `pipeline/inventory.py` ignores rollouts belonging to models outside the requested cohort.
- `metrics/weight_utils.py` uses managed external weight directories and registered
  base-model assets. AMT, RAFT, CLIP and aesthetic checkpoints must be staged.
- CLIP, RAFT and AMT import their canonical `base_models` implementations. RAFT
  failure raises instead of falling back to a different optical-flow algorithm.
  Motion backend config digests distinguish this required-RAFT policy, preventing
  reuse of older inference caches that may have used the fallback.
- HPSv3 uses `base_models.perception_core.video_quality.hpsv3.load_inferencer` and
  its existing `reward` API; CPU preparation keeps image files, and inference
  retains the same blank-prompt per-frame mean. Checkpoint parity needs a GPU run.
- MegaSAM's resident orchestration is implemented in
  `worldfoundry/base_models/three_dimensions/slam/megasam_resident.py`. Frame
  extraction and stride selection reuse `megasam.py`; mono-depth uses canonical
  Depth Anything v1 and DINOv2. Scene tracking outputs go to temporary storage.
  UniDepth defaults to the canonical `depth.unidepth.UniDepth2Model` loader and its
  registered `unidepth_v2_vitl14` assets. The canonical UniDepth source comes from
  upstream revision `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`; its compatibility
  decoder (`UniDepthV2old` upstream) matches the pinned HF snapshot
  `1d0d3c52f60b5164629d279bb9a7546458e6dcc4` strictly. The newer decoder has
  incompatible parameters and must not be loaded with `strict=False` against this snapshot.
  DINOv2 is shared; only UniDepth's output contract and unused checkpoint register differ.
  MegaSAM's Python tracking function is adapted from its existing tracking driver;
  resident DROIDNet injection replaces `runpy` and global loader monkeypatching.
  WBench calls the same pipeline in an isolated Python worker. Full-suite benchmark
  parity remains unverified; checkpoint loading and individual UniDepth GPU inference
  are separate evidence from benchmark validation.
- Uncalled standalone PAVRM reporting/API helpers are removed; the metric retains
  official reward inference through Transformers and its selected-skill caller.

The runner invokes Python APIs and bounded Python skill workers. Drift stages run
in separate processes; no model networks are duplicated under this evaluator.
Synthetic fixtures are WorldFoundry-authored. They verify schema and aggregation,
not the published 100-case suite. A full GPU/API run and judge/checkpoint parity
have not been verified.

Validation on 2026-09-19: pinned UniDepth weights load strictly and produce finite
GPU depth/points/intrinsics/confidence through both native inference and the shared
geometry-prior adapter. Both MegaSAM CUDA extensions compile for H100 (GCC 11,
CUDA 12.6, PyTorch 2.5.1+cu121). A 12-frame synthetic video produces 12 finite 4x4
camera poses through the public MegaSAM worker. HarnessEval-W's LocalBackend also
processes that video twice with one resident network load and 12 poses each time.
These are model smoke checks, not the 100-case benchmark or accuracy/parity evidence.
