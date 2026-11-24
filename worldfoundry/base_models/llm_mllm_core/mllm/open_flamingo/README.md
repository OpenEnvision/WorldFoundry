# `mllm/open_flamingo`

OpenFlamingo source (vision encoder + cross-attention LM), exposed as the top-level
`open_flamingo` package. It is a policy backbone, not an evaluation model.

## Used by

| Consumer | Purpose |
| --- | --- |
| RoboFlamingo VLA (`worldfoundry/synthesis/action_generation/roboflamingo/`) | Language-conditioned manipulation policy backbone |

No benchmark metric uses it.

## Usage

```python
from worldfoundry.base_models.llm_mllm_core.mllm.open_flamingo import ensure_import_paths

ensure_import_paths()  # makes `import open_flamingo` resolve to this tree
```

RoboFlamingo's runtime config is `worldfoundry/data/models/runtime/configs/vla_va_wam/roboflamingo.yaml`.
