# Natural-25 prompt suite

This directory ships the **Natural-25** scene/event prompt grid used by WRBench.

| File | Description |
| --- | --- |
| `scene_events_25x4.csv` | 25 scene families × 4 event axes (none / spatial / state / full) |
| `families.jsonl` | Scene-family metadata and prompt variants |
| `variants.jsonl` | Pre-generated deterministic TI2V prompt variants: 25 families × 4 event tiers × 4 camera gaps |
| `variants.legacy_pronoun_20260620.jsonl` | Frozen legacy snapshot kept for released-result provenance checks; currently byte-identical to `variants.jsonl` |
| `t2v_layout_anchors.jsonl` | Layout facts for text-only Natural-25 generation: subject left, interactor right, open surface, and background anchors |
| `t2v_event_tails.jsonl` | Text-only event/action tails keyed by `variant_id`, kept separate from the TI2V prompt of record |
| `prompt_profiles/t2v_layout_anchor.json` | Prompt-profile policy for text-only models; bans T2I/TI2V style tokens and keeps only layout/event facts |
| `camera_scopes/t2v_rotation_stress_30_60.json` | T2V-only rotation-stress scope: `static`, `yaw30_LR`, `yaw30_RL`, `yaw60_LR`, `yaw60_RL` |
| `first_frames/` | Released Natural-25 PNG first frames, one per `family_id` |
| `first_frames_manifest.json` | First-frame paths and source `t2i_scene` prompts |

The bundled first frames and prompt variants are released with the WRBench
repository under the repository Apache-2.0 license, so the open-source
camera-compile examples can run without generating a T2I image first. You can
also regenerate first-frame PNGs per family via `wrbench firstframe` (optional
extra) or substitute images from your own T2I pipeline. Human annotation
verdicts are released separately from this repository.

`variants.jsonl` keeps `ti2v_prompt` as the first-frame-anchored prompt of
record. Text-only runs should explicitly materialize a prompt profile such as
`t2v_layout_anchor`, where the initial layout and event tails are maintained in
separate T2V files because there is no first frame to carry them. Prompt-only
T2V leaderboard entries also use their own rotation-stress scope rather than
the main TI2V/TV2V leaderboard surface.

`variants.legacy_pronoun_20260620.jsonl` remains only as a frozen provenance
artifact for released T2V result metadata and OSS verification. The generic
duplicate rotation-stress scope is not shipped; `t2v_rotation_stress_30_60.json`
is the single bundled rotation-stress scope.

## Hugging Face release

- WRBench release collection:
  <https://huggingface.co/collections/WRBench/wrbench-current-world-models-lack-a-persistent-state-core-6a365c717251293c9fc2cc26>
- Natural-25 dataset card and viewer:
  <https://huggingface.co/datasets/WRBench/wrbench-natural25>
- Benchmark videos generated from Natural-25:
  <https://huggingface.co/datasets/WRBench/wrbench-videos>
- WRBench leaderboard:
  <https://huggingface.co/spaces/WRBench/wrbench-leaderboard>
- Paper page:
  <https://huggingface.co/papers/2606.20545>
