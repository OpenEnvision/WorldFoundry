# `worldfoundry.studio`

Browser UI for WorldFoundry: the multi-model **Workspace** (job submission, gallery,
visualizers, evaluation jobs) and single-model **Studio** frontends. Studio is a presentation
shell over the same catalog and pipelines as the CLI; it adds no model logic of its own.

## Usage

Multi-model workspace (recommended):

```bash
bash scripts/workspace/run_workspace.sh --host 127.0.0.1 --port 7870
# open http://127.0.0.1:7870/
```

Single model with a fixed frontend:

```bash
wf-studio lingbot-world --frontend interactive-world --port 7868
wf-studio --help
```

Headless job, the same path the Workspace **Create Job** form submits:

```bash
python -m worldfoundry.studio.workspace_job infer \
  --model-id <model-id> --prompt "..." --output-dir tmp/studio_job --device cuda
```

Guide: [Studio](https://openenvision.github.io/WorldFoundry/docs/guides/studio) ·
internals: [Studio internals](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture/studio-internals).

## Layout

| Path | Purpose |
| --- | --- |
| `cli.py` | `wf-studio` entry point (single-model session) |
| `workspace_job.py` | `python -m worldfoundry.studio.workspace_job infer` headless job runner |
| `runtime_job.py` | Job execution inside a Workspace worker |
| `inference/` | Studio model catalog, listings, request building, dispatch, resident workers, request batching |
| `workspace/` | Workspace app (`app.py`) and job queue (`jobs.py`) |
| `serving/` | HTTP helpers, auth, network, realtime (WebRTC) transport, telemetry |
| `visualization/` | Launcher and launch config, artifact → viewer routing, viewer backends (Viser point clouds, Rerun, interactive world), run records |
| `assets/` | Logo, reference images, browser pages/scripts (`web/`), vendored frontend assets |

## Notes

- Start the Workspace from the unified conda environment; a UI-only environment cannot load models.
- Interactive-world sessions keep one pipeline resident and stream frames over WebRTC; manifests and MP4 export stay on the offline path.
