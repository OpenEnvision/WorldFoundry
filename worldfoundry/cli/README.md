# `worldfoundry.cli`

Command-line surface of WorldFoundry, installed as the `wf` console script;
`python -m worldfoundry` uses the same parser.
The CLI turns user intent into requests and dispatches them — it does not own model
inference or scoring.

## Usage

```bash
wf --help                       # top-level command list
wf zoo models                   # browse the model catalog
wf run <model-id> --help        # typed --pipeline.* / --runtime.* flags
wf run <model-id> --output-dir tmp/run
wf run --model <id> --benchmark <id> --mode official-run --output-dir tmp/bench
wf tui                          # interactive command builder
```

Full reference: [CLI](https://openenvision.github.io/WorldFoundry/docs/reference/cli) ·
[TUI](https://openenvision.github.io/WorldFoundry/docs/guides/tui).

## Layout

| File | Purpose |
| --- | --- |
| `main.py` | `main()` entry point, root help, and parser assembly from each command's `register_*` function |
| `run.py` | `wf run`: direct model inference, model x benchmark cells, suites, run-plan replay |
| `model_run.py` | Typed per-model flags for `run <model-id>` (`--pipeline.*`, `--pipeline.load.*`, `--runtime.*`) |
| `evaluate.py` | `wf evaluate` (alias `eval`) and `wf validate` |
| `embodied.py` | `wf embodied run / serve / merge / plan` |
| `zoo.py` | `zoo models / benchmarks / model-show / model-download / benchmark-run` |
| `models.py` | `models list / runtime-runners / assets / visualize` |
| `evaluation_intents.py` | `score`, `generate-score`, `reproduce` intents |
| `tasks.py`, `dataset.py`, `plan_metric.py`, `config.py` | `tasks` / `task`, `dataset`, `plan` / `metric`, `config` subcommands |
| `preflight.py`, `reporting.py`, `mcp.py` | `preflight runtime`; `index-runs`, `compare-runs`, `validate-artifact`; `mcp` |
| `catalog.py` | Model, benchmark, and runtime-profile catalog rows shared by the TUI and the MCP server |
| `inference_variants.py` | Inference variant registry and resolution; also the entry point behind `scripts/inference/run_infer.sh` |
| `command_builders.py` | Shell command builders for inference, model x benchmark runs, suites, and Studio |
| `observability.py` | Logging setup and per-run lifecycle events for commands that own an output tree |
| `tui/` | Terminal UI: `__init__.py` (`wf tui` parser), `launcher.py` (action dispatch), Textual `app.py` and `app.tcss`, form `controls.py`, brand/logo/icon helpers, `assets/` |
| `presentation.py`, `help.py`, `utils.py` | Tables, help rendering, JSON output and shared argparse helpers |

## Adding a command

Add a module with a `register_*` function that builds the subparser and binds its handler,
call it from `_build_parser` in `main.py` with a function-local import (help stays
import-light), and call into `worldfoundry.evals` or `worldfoundry.runtime` for the
actual work. Every command that
other programs consume should accept `--json` on the leaf subcommand.
