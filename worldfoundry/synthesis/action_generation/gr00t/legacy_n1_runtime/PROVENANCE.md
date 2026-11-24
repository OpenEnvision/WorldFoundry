# GR00T-N1 inference runtime

`gr00t/` is adapted from NVIDIA/Isaac-GR00T, revision `755876a9afdb41ca6eb6383b36f4a0adb085c73f`
(n1-release; Apache License 2.0, see root `THIRD-PARTY-NOTICES`).

The directory holds the import closure of the original GR00T-N1 policy that `legacy_n1_worker.py`
runs: the GR1 data configurations and transforms, the GR00T-N1 model with its flow-matching action
head, and the Eagle2 backbone code under `gr00t/model/backbone/eagle2_hg_model/`. Its configuration
and tokenizer files are bundled at `worldfoundry/data/models/runtime/configs/gr00t-n1/eagle2_hg_model/`,
which `$GR00T_BACKBONE_PATH` in checkpoint configurations resolves to. Every Python file was modified
by WorldFoundry: imports are package-relative and the code is formatted to the repository style.

The Eagle2 configuration and model are built from their classes (`Eagle2ChatConfig.from_pretrained`
and `Eagle2ChatModel._from_config`, which `AutoConfig` and `AutoModel` reach through `auto_map`)
instead of being loaded as remote code, and the tokenizer is loaded without remote code.
`gr00t/data/dataset.py` keeps only `ModalityConfig`; the LeRobot dataset classes, their statistics
helper and `gr00t/utils/video.py`, which only they use, are omitted, as are the evaluation,
experiment-runner and training modules of the release. Two statements in
`find_closest_aspect_ratio_v2` that computed values without using them were removed.

Only definitions that inference reaches are kept. The flow-matching action head builds its DiT and
its embodiment-conditioned state encoder, action encoder and action decoder from
`worldfoundry/synthesis/action_generation/gr00t/modeling`, the GR00T N1.7 inference modules, which
construct the same parameters and compute the same outputs for N1 configurations; on sm121 GPUs, or
with `GR00T_DIT_SDPA_MODE=math`, their attention runs on the math SDPA backend.
`gr00t/model/action_head/cross_attention_dit.py` and the local copies of those layers are therefore
omitted. Training members are omitted as well: the loss forward passes of `GR00T_N1` and the action
head, flow-matching time sampling, the trainable-parameter switches, the Eagle2 LoRA wrappers
(released configurations set `use_backbone_lora` and `use_llm_lora` to 0), Eagle2 chat and generation
methods, the training-time video and state-action augmentations, the LeRobot schema classes and the
training data collator.
