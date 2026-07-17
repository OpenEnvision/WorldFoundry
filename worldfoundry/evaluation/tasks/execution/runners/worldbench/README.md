# WorldBench in-tree evaluator

WorldFoundry evaluates the `IntuitivePhysics` track directly from generated
continuations and/or textual answers. The runtime is independent code under
`runtime/worldbench`; it does not execute or import a checkout of the upstream
repository. SAM2 inference is reused from
`worldfoundry.base_models.perception_core.segment.sam2`.

## Inputs

The dataset root must contain the public Hugging Face layout:

```text
<dataset>/
  scenes/<group>/<scenario>/<scene>/
    rgba_00000.png
    segmentation_00000.png
    ...
  textual_questions/*.json
```

Generated continuations can mirror the scene IDs:

```text
<generated>/<group>/<scenario>/<scene>.mp4
<generated>/<group>/<scenario>/<scene>/output.mp4
<generated>/<group>/<scenario>/<scene>/00000.png
```

For any other layout, pass `--video-manifest`. Each JSON/JSONL/CSV/TSV row must
contain `sample_id` and `video_path`. Text predictions are supplied with
`--answer-manifest`; use the stable `question_id` values `<question-file-stem>:<row>`
or include both `video_name` and `question` because some videos have two questions.

## Run

```bash
PYTHONPATH=. "$WORLDFOUNDRY_UNIFIED_PYTHON" -m \
  worldfoundry.evaluation.tasks.execution.runners.worldbench.run_worldbench_official_runner \
  --run-official \
  --dataset-root "$WORLDFOUNDRY_WORLDBENCH_DATASET_ROOT" \
  --generated-video-dir "$WORLDFOUNDRY_GENERATED_ARTIFACT_DIR" \
  --answer-manifest answers.json \
  --output-dir outputs/worldbench \
  --json
```

Raw video scoring defaults to `facebook/sam2.1-hiera-large`, matching the public
release. Set `WORLDFOUNDRY_WORLDBENCH_SAM2_CKPT` to a local checkpoint; otherwise
the standard Hugging Face cache resolver is used. `--predicted-mask-dir` bypasses
SAM2 for deterministic metric regression. Existing upstream result files remain
supported with `--official-results-path` and no `--run-official`.

The benchmark uses nine conditioning frames. By default, generated frame zero is
aligned to ground-truth frame nine and 24 continuation frames are scored. If a
model prepends the nine conditioning frames to its output, pass
`--generated-skip-frames 9`. These defaults live in
`worldfoundry/data/benchmarks/assets/worldbench/evaluator.yaml`.

## Environment

No dedicated environment is required. Use `worldfoundry-unified-cu128`, which
provides PyTorch, Hydra, decord/OpenCV, NumPy, Pillow, and the shared SAM2 runtime.
The canonical dependency list is
`worldfoundry/data/benchmarks/runtime_profiles/official/worldbench.yaml`.

## Provenance and metric caveat

The official distributed bundle corresponds to
`WorldBenchmark/worldbench_eval@4575a164225341b5a87de7235cd50216b4b204d8`
but contains no license file. It also imports a missing module, references an
undefined variable, and consumes a result shape different from the one it writes.
WorldFoundry therefore keeps a clean in-tree implementation and stores provenance
in the data YAML instead of a `SOURCE.json`.

The public implementation calls a Dice coefficient `mIoU`. WorldFoundry reports
conventional `foreground_miou` and the release-compatible `foreground_dice`
separately. A successful in-tree run is integration evidence, but is not marked
leaderboard-valid until full-dataset numerical parity is audited.
