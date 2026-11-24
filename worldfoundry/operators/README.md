# `worldfoundry.operators`

Input-side partners of pipelines. An operator validates and normalizes what the user passes
in — loading images and videos, parsing camera trajectories and action sequences, shaping
keyboard/mouse interactions, and running perception preprocessing — so the pipeline receives
a consistent request. Operators never run the model.

## Layout

| Path | Purpose |
| --- | --- |
| `base_operator.py` | `BaseOperator`, the base class |
| `interactive_world_operator.py` | Shared handling for interactive (WASD/IJKL, camera) world models |
| `embodied_action_operator.py` | Shared handling for robot/VLA observations and instructions |
| `geometry_prior_operator.py` | Shared handling for depth/geometry prior models |
| `<model>_operator.py` | One operator per model family (e.g. `gen3c_operator.py`, `matrix_game_*`, `kling_api_operator.py`) |

## Adding an operator

Every runnable pipeline has an operator, even a thin one. Subclass `BaseOperator` (or one of
the shared operators above), keep input parsing here rather than in the pipeline, and wire it
from the pipeline constructor. See [Runtime assembly](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture/runtime-assembly).
