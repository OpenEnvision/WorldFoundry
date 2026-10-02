import itertools
import json
import sys
from pathlib import Path

import pytest
import torch

from worldfoundry.core.attention.sparse.block_pattern import AttnMaskSpec, BlockPattern, build_mask_fn
from worldfoundry.synthesis.visual_generation.rcm import worldfoundry_runtime as rcm_runtime
from worldfoundry.synthesis.visual_generation import runtime_manifest


def _dense(mask_fn, q_len: int, kv_len: int) -> torch.Tensor:
    """Materialize a mask predicate over a full index grid."""
    q_idx = torch.arange(q_len).view(q_len, 1).expand(q_len, kv_len)
    kv_idx = torch.arange(kv_len).view(1, kv_len).expand(q_len, kv_len)
    return mask_fn(0, 0, q_idx, kv_idx)


def _stage_checkpoints(tmp_path: Path) -> dict[str, str]:
    """Create the three checkpoint files a rollout gates on."""
    root = tmp_path / "ckpt"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "dit_path": root / "test-distilled-dit.pt",
        "vae_path": root / "test-vae.pt",
        "text_encoder_path": root / "test-text-encoder.pt",
    }
    for path in paths.values():
        path.write_bytes(b"")
    return {key: str(value) for key, value in paths.items()}


# ── BlockPattern: the shared causal chunk schedule ───────────


def test_block_pattern_chunk_arithmetic() -> None:
    pattern = BlockPattern(frame_tokens=10, first_chunk_frames=1, chunk_frames=3)

    assert pattern.get_block_tokens(0) == 10
    assert pattern.get_block_tokens(1) == 30
    assert pattern.blocks_to_frames(3) == 7
    assert pattern.blocks_to_tokens(3) == 70
    assert pattern.block_size(0) == 1
    assert pattern.block_size(2) == 3
    assert pattern.spans(3, 0)[0] == [(0, 1), (1, 4), (4, 7)]
    # A nonzero offset means block 0 is behind us, so every block is chunk-sized.
    assert pattern.spans(3, 2)[0] == [(0, 3), (3, 6), (6, 9)]
    assert pattern.block_bounds(0, 70) == [0, 10, 40, 70]


def test_token_to_rel_block_respects_the_first_chunk() -> None:
    pattern = BlockPattern(frame_tokens=2, first_chunk_frames=3, chunk_frames=2)
    tokens = torch.arange(20)

    with_first = pattern.token_to_rel_block(tokens, use_first_block=True)
    without_first = pattern.token_to_rel_block(tokens, use_first_block=False)

    # First three frames (six tokens) collapse into block 0.
    assert with_first[:6].tolist() == [0] * 6
    assert with_first[6:10].tolist() == [1] * 4
    # Streaming queries past the first chunk use uniform chunk-sized blocks.
    assert without_first[:4].tolist() == [0] * 4


def test_block_causal_mask_is_lower_triangular_over_blocks() -> None:
    spec = AttnMaskSpec(mode="block_causal", pattern=BlockPattern(frame_tokens=1, first_chunk_frames=1, chunk_frames=2))
    mask_fn, _ = build_mask_fn(spec, q_real=7, kv_real=7)

    mask = _dense(mask_fn, 7, 7)

    # Block layout over 7 tokens: [0] [1 2] [3 4] [5 6]
    assert mask[0].tolist() == [True, False, False, False, False, False, False]
    # Inside a block attention is bidirectional, so token 1 sees token 2.
    assert mask[1].tolist() == [True, True, True, False, False, False, False]
    assert mask[6].tolist() == [True] * 7


def test_sliding_window_and_sink_blocks_compose() -> None:
    pattern = BlockPattern(frame_tokens=1, first_chunk_frames=1, chunk_frames=1)
    windowed, _ = build_mask_fn(
        AttnMaskSpec(mode="block_causal", pattern=pattern, local_attn_blocks=2), q_real=6, kv_real=6
    )
    with_sink, _ = build_mask_fn(
        AttnMaskSpec(mode="block_causal", pattern=pattern, local_attn_blocks=2, sink_blocks=1), q_real=6, kv_real=6
    )

    # A 2-block window keeps only the current and previous block.
    assert _dense(windowed, 6, 6)[5].tolist() == [False, False, False, False, True, True]
    # The sink block stays attendable no matter how far the window has moved.
    assert _dense(with_sink, 6, 6)[5].tolist() == [True, False, False, False, True, True]


def test_teacher_forcing_mask_separates_clean_and_noisy_halves() -> None:
    pattern = BlockPattern(frame_tokens=1, first_chunk_frames=1, chunk_frames=1)
    spec = AttnMaskSpec(mode="teacher_forcing", pattern=pattern, clean_blocks=3)
    mask_fn, _ = build_mask_fn(spec, q_real=6, kv_real=6)

    mask = _dense(mask_fn, 6, 6)

    # Clean queries (0-2) are causal within the clean half and never see noise.
    assert mask[2].tolist() == [True, True, True, False, False, False]
    # Noisy query for block 2 (index 5) sees strictly earlier clean blocks plus itself.
    assert mask[5].tolist() == [True, True, False, False, False, True]
    # The first noisy query has no earlier clean block, so only its diagonal is open.
    assert mask[3].tolist() == [False, False, False, True, False, False]


def test_build_mask_fn_rejects_unmasked_mode() -> None:
    with pytest.raises(ValueError, match="Unknown attention mask mode"):
        build_mask_fn(AttnMaskSpec(mode="none"), q_real=4, kv_real=4)


def test_teacher_forcing_requires_clean_blocks() -> None:
    spec = AttnMaskSpec(mode="teacher_forcing", pattern=BlockPattern(frame_tokens=1))
    with pytest.raises(ValueError, match="clean_blocks must be positive"):
        build_mask_fn(spec, q_real=4, kv_real=4)


def test_mask_signatures_distinguish_configurations() -> None:
    pattern = BlockPattern(frame_tokens=1, first_chunk_frames=1, chunk_frames=2)
    _, first = build_mask_fn(AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=0), q_real=8, kv_real=8)
    _, second = build_mask_fn(AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=2), q_real=8, kv_real=8)

    # Signatures key the compiled block-mask cache; equal keys would serve a
    # streaming chunk the wrong mask.
    assert first != second
    assert hash(first) != hash(second)


def test_vendored_runtime_reexports_the_core_primitive() -> None:
    """The vendored blockmask module must not fork its own copy."""
    blockmask = pytest.importorskip(
        "worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.blockmask",
        reason="vendored rCM attention stack needs the optional runtime dependencies",
    )

    assert blockmask.BlockPattern is BlockPattern
    assert blockmask.AttnMaskSpec is AttnMaskSpec


# ── Runtime manifest wiring ──────────────────────────────────


def test_runtime_spec_points_at_the_vendored_entrypoint() -> None:
    spec = runtime_manifest.runtime_spec("causal-rcm")

    assert spec.display_name == "Causal-rCM"
    # Source is vendored under Apache-2.0, so only checkpoints gate execution.
    assert spec.blocked_reason == ""

    root, entrypoint, blocked_reason = runtime_manifest.resolve_runtime_manifest(spec)
    assert root == rcm_runtime.RUNTIME_DIR
    assert entrypoint == rcm_runtime.INFERENCE_ENTRYPOINT
    assert entrypoint.is_file()
    assert blocked_reason == ""


def test_upstream_framework_package_is_not_vendored() -> None:
    """rCM's `imaginaire` framework is replaced by WorldFoundry equivalents."""
    import ast

    vendored = rcm_runtime.RUNTIME_DIR / "rcm_runtime"
    offenders = []

    for path in sorted(vendored.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root in {"imaginaire", "rcm"}:
                    offenders.append(f"{path.name}: {name}")

    assert offenders == [], f"vendored runtime still imports upstream packages: {offenders}"


def test_command_settings_merges_plan_extras_over_load_options(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"extra": {"num_steps": 2, "seed": 7}}), encoding="utf-8")

    settings = runtime_manifest.command_settings(
        {"options": {"num_steps": 4, "chunk_t": 3}, "plan_path": str(plan_path)}
    )

    assert settings == {"num_steps": 2, "chunk_t": 3, "seed": 7}


def test_command_settings_survives_a_missing_plan(tmp_path: Path) -> None:
    settings = runtime_manifest.command_settings({"options": {"seed": 1}, "plan_path": str(tmp_path / "absent.json")})

    assert settings == {"seed": 1}


# ── Requirement gating ───────────────────────────────────────


def test_missing_requirements_reports_absent_checkpoints(tmp_path: Path) -> None:
    missing = rcm_runtime.missing_requirements(
        options={"python_executable": "/opt/causal-rcm/bin/python"},
        runtime_root=rcm_runtime.RUNTIME_DIR,
        entrypoint=rcm_runtime.INFERENCE_ENTRYPOINT,
        profile=None,
    )

    reasons = {entry["path"] for entry in missing}
    assert str(rcm_runtime.DEFAULT_CHECKPOINT_DIR / rcm_runtime.DEFAULT_DIT_FILENAME) in reasons
    assert all(entry["kind"] == "checkpoint" for entry in missing)


def test_missing_requirements_is_empty_once_checkpoints_are_staged(tmp_path: Path) -> None:
    options = _stage_checkpoints(tmp_path)
    options["python_executable"] = "/opt/causal-rcm/bin/python"

    missing = rcm_runtime.missing_requirements(
        options=options,
        runtime_root=rcm_runtime.RUNTIME_DIR,
        entrypoint=rcm_runtime.INFERENCE_ENTRYPOINT,
        profile=None,
    )

    assert missing == []


def test_missing_requirements_rejects_a_partial_public_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    default_dit = checkpoint_dir / rcm_runtime.DEFAULT_DIT_FILENAME
    default_dit.write_bytes(b"partial")

    missing = rcm_runtime.missing_requirements(
        options={"checkpoint_dir": str(checkpoint_dir), "python_executable": "/opt/causal-rcm/bin/python"},
        runtime_root=rcm_runtime.RUNTIME_DIR,
        entrypoint=rcm_runtime.INFERENCE_ENTRYPOINT,
        profile=None,
    )

    dit_missing = next(item for item in missing if item["path"] == str(default_dit))
    assert "incomplete" in dit_missing["reason"]


def test_missing_requirements_probes_packages_only_for_this_interpreter(tmp_path: Path) -> None:
    """A configured interpreter owns its own environment and cannot be probed."""
    options = _stage_checkpoints(tmp_path)

    same_interpreter = rcm_runtime.missing_requirements(
        options={**options, "python_executable": sys.executable},
        runtime_root=rcm_runtime.RUNTIME_DIR,
        entrypoint=rcm_runtime.INFERENCE_ENTRYPOINT,
        profile=None,
    )
    other_interpreter = rcm_runtime.missing_requirements(
        options={**options, "python_executable": "/opt/causal-rcm/bin/python"},
        runtime_root=rcm_runtime.RUNTIME_DIR,
        entrypoint=rcm_runtime.INFERENCE_ENTRYPOINT,
        profile=None,
    )

    assert other_interpreter == []
    assert all(entry["kind"] == "python_module" for entry in same_interpreter)


def test_dedicated_environment_interpreter_is_selected_when_available(monkeypatch, tmp_path: Path) -> None:
    interpreter = tmp_path / rcm_runtime.RUNTIME_ENV_NAME / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    monkeypatch.setenv("WORLDFOUNDRY_CONDA_ENVS_ROOT", str(tmp_path))

    assert rcm_runtime._python_executable({}) == str(interpreter)


# ── Command construction ─────────────────────────────────────


def _context(tmp_path: Path, *, options: dict, extra: dict | None = None) -> dict:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "causal-rcm.json"
    plan_path.write_text(json.dumps({"extra": extra or {}}), encoding="utf-8")
    return {
        "python": "/unused/python",
        "runtime_root": str(rcm_runtime.RUNTIME_DIR),
        "entrypoint": str(rcm_runtime.INFERENCE_ENTRYPOINT),
        "output_path": str(output_dir / "causal-rcm.mp4"),
        "output_dir": str(output_dir),
        "prompt": "a snowy mountain at sunrise",
        "device": "cuda",
        "plan_path": str(plan_path),
        "options": options,
        "profile": None,
    }


def test_build_command_matches_the_distilled_recipe(tmp_path: Path) -> None:
    options = _stage_checkpoints(tmp_path)
    options["python_executable"] = "/opt/causal-rcm/bin/python"

    command = rcm_runtime.build_command(_context(tmp_path, options=options, extra={"first_chunk_t": 3, "chunk_t": 3}))

    assert command[0] == "/opt/causal-rcm/bin/python"
    assert command[1] == str(rcm_runtime.INFERENCE_ENTRYPOINT)
    assert command[command.index("--dit_path") + 1] == options["dit_path"]
    assert command[command.index("--first_chunk_t") + 1] == "3"
    assert command[command.index("--chunk_t") + 1] == "3"
    assert command[command.index("--prompt") + 1] == "a snowy mountain at sunrise"
    assert "--distilled" in command
    # Default 4-step midpoints from the Causal-rCM recipe.
    mid_t_at = command.index("--mid_t")
    assert command[mid_t_at + 1 : mid_t_at + 4] == ["15/16", "5/6", "5/8"]


def test_build_command_switches_to_the_causal_diffusion_teacher(tmp_path: Path) -> None:
    options = _stage_checkpoints(tmp_path)

    command = rcm_runtime.build_command(
        _context(tmp_path, options=options, extra={"distilled": False, "num_steps": 50, "guidance_scale": 3.0})
    )

    assert "--distilled" not in command
    assert "--mid_t" not in command
    assert command[command.index("--num_steps") + 1] == "50"
    assert command[command.index("--guidance_scale") + 1] == "3.0"


def test_build_command_accepts_per_chunk_step_schedules(tmp_path: Path) -> None:
    options = _stage_checkpoints(tmp_path)

    command = rcm_runtime.build_command(
        _context(
            tmp_path,
            options=options,
            extra={"steps_per_chunk": [4, 2], "mid_t_schedules": "15/16,5/6,5/8;5/6"},
        )
    )

    steps_at = command.index("--steps_per_chunk")
    assert command[steps_at + 1 : steps_at + 3] == ["4", "2"]
    assert command[command.index("--mid_t_schedules") + 1] == "15/16,5/6,5/8;5/6"


def test_build_command_enables_noisy_context(tmp_path: Path) -> None:
    options = _stage_checkpoints(tmp_path)

    command = rcm_runtime.build_command(
        _context(
            tmp_path,
            options=options,
            extra={"context_from_last_step": True, "context_from_last_step_start_chunk": 1},
        )
    )

    assert "--context_from_last_step" in command
    assert command[command.index("--context_from_last_step_start_chunk") + 1] == "1"


def test_build_command_forwards_the_bounded_cache_configuration(tmp_path: Path) -> None:
    options = _stage_checkpoints(tmp_path)

    command = rcm_runtime.build_command(
        _context(
            tmp_path,
            options=options,
            extra={
                "kv_cache_policy": "sliding_window",
                "kv_cache_window_blocks": 4,
                "kv_cache_sink_blocks": 1,
            },
        )
    )

    assert command[command.index("--kv_cache_policy") + 1] == "sliding_window"
    assert command[command.index("--kv_cache_window_blocks") + 1] == "4"
    assert command[command.index("--kv_cache_sink_blocks") + 1] == "1"


def test_default_command_keeps_the_upstream_cache(tmp_path: Path) -> None:
    command = rcm_runtime.build_command(_context(tmp_path, options=_stage_checkpoints(tmp_path)))

    assert "--kv_cache_policy" not in command


def test_build_command_rejects_rope_remapping_cache_policies(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_all or sliding_window"):
        rcm_runtime.build_command(
            _context(
                tmp_path,
                options=_stage_checkpoints(tmp_path),
                extra={"kv_cache_policy": "block_relative_rope"},
            )
        )


def test_build_command_adds_i2v_conditioning(tmp_path: Path) -> None:
    options = _stage_checkpoints(tmp_path)
    image = tmp_path / "seed.jpg"
    image.write_bytes(b"")

    command = rcm_runtime.build_command(
        _context(tmp_path, options=options, extra={"image_path": str(image), "adaptive_resolution": True})
    )

    assert command[command.index("--image_path") + 1] == str(image)
    assert "--adaptive_resolution" in command


def test_build_command_emits_benchmark_flags_on_request(tmp_path: Path) -> None:
    options = _stage_checkpoints(tmp_path)

    plain = rcm_runtime.build_command(_context(tmp_path, options=options))
    timed = rcm_runtime.build_command(_context(tmp_path, options=options, extra={"warmup_iters": 3, "num_runs": 3}))

    assert "--warmup_iters" not in plain
    assert timed[timed.index("--warmup_iters") + 1] == "3"
    assert timed[timed.index("--num_runs") + 1] == "3"


def test_build_command_uses_the_public_default_dit_checkpoint(tmp_path: Path) -> None:
    command = rcm_runtime.build_command(_context(tmp_path, options={}))

    assert command[command.index("--dit_path") + 1] == str(
        rcm_runtime.DEFAULT_CHECKPOINT_DIR / rcm_runtime.DEFAULT_DIT_FILENAME
    )


def test_mid_t_accepts_a_delimited_string(tmp_path: Path) -> None:
    options = _stage_checkpoints(tmp_path)

    command = rcm_runtime.build_command(_context(tmp_path, options=options, extra={"mid_t": "15/16, 5/6, 5/8"}))

    mid_t_at = command.index("--mid_t")
    assert command[mid_t_at + 1 : mid_t_at + 4] == ["15/16", "5/6", "5/8"]


def test_bounded_cache_plan_reserves_only_the_retained_prefix_and_current_chunk() -> None:
    """The policy must lower allocation, not merely evict after a full allocation."""
    from worldfoundry.synthesis.visual_generation.rcm.kv_cache_plan import make_kv_cache_plan

    pattern = BlockPattern(frame_tokens=10, first_chunk_frames=1, chunk_frames=3)
    default_policy, default_frames = make_kv_cache_plan(
        pattern,
        5,
        policy_name="keep_all",
        window_blocks=4,
        sink_blocks=1,
    )
    bounded_policy, bounded_frames = make_kv_cache_plan(
        pattern,
        5,
        policy_name="sliding_window",
        window_blocks=3,
        sink_blocks=1,
    )

    assert default_policy is None
    assert default_frames == 13
    # After the [1], [3], [3] window is committed, READONLY still needs one
    # more 3-frame in-flight block.  10 frames is the exact maximum buffer.
    assert bounded_policy is not None
    assert bounded_frames == 10


def test_bounded_cache_plan_rejects_rope_remapping_policies() -> None:
    from worldfoundry.synthesis.visual_generation.rcm.kv_cache_plan import make_kv_cache_plan

    with pytest.raises(ValueError, match="keep_all and sliding_window"):
        make_kv_cache_plan(
            BlockPattern(frame_tokens=10, first_chunk_frames=1, chunk_frames=1),
            5,
            policy_name="block_relative_rope",
            window_blocks=3,
            sink_blocks=0,
        )


# ── Pipeline call surface ────────────────────────────────────


def test_pipeline_scopes_gated_options_to_a_single_call(tmp_path: Path) -> None:
    from worldfoundry.pipelines.world_model import CausalRCMPipeline

    class _Synthesis:
        def __init__(self) -> None:
            self.options: dict = {}
            self.seen: list[dict] = []

    synthesis = _Synthesis()

    class _Recorder(CausalRCMPipeline):
        def _call_component_pipeline(self, *args, **kwargs):
            synthesis.seen.append(dict(synthesis.options))
            return {"artifact_path": ""}

    pipeline = _Recorder(model_id="causal-rcm", synthesis_model=synthesis, operators=object())
    pipeline(prompt="", dit_path="/staged/dit.pt", images=str(tmp_path / "seed.jpg"))

    seen = synthesis.seen[0]
    assert seen["dit_path"] == "/staged/dit.pt"
    # The shared checkpoint gate looks for `checkpoint_path`, so dit_path mirrors onto it.
    assert seen["checkpoint_path"] == "/staged/dit.pt"
    assert seen["image_path"] == str(tmp_path / "seed.jpg")
    assert synthesis.options == {}


def test_base_pipeline_scoping_is_opt_in() -> None:
    """Runtimes opt in only to the per-call assets they gate on."""
    from worldfoundry.pipelines.world_model import DIAMONDPipeline, WorldModelRuntimePipeline

    assert WorldModelRuntimePipeline.RUNTIME_GATED_OPTION_KEYS == ()
    assert "pretrained_dir" in DIAMONDPipeline.RUNTIME_GATED_OPTION_KEYS
    assert "headless_steps" in DIAMONDPipeline.RUNTIME_GATED_OPTION_KEYS


def test_diamond_promotes_workspace_frames_to_headless_rollout() -> None:
    from worldfoundry.pipelines.world_model import DIAMONDPipeline

    pipeline = object.__new__(DIAMONDPipeline)

    assert pipeline._promote_call_options({"frames": 9})["headless_steps"] == 9


def test_every_registered_mask_mode_round_trips() -> None:
    """Guard the block-causal predicate across the schedules the recipe ships."""
    for first, chunk, offset in itertools.product((1, 3), (1, 3, 4), (0, 2)):
        pattern = BlockPattern(frame_tokens=2, first_chunk_frames=first, chunk_frames=chunk)
        spec = AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=offset)
        mask_fn, _ = build_mask_fn(spec, q_real=12, kv_real=12)
        mask = _dense(mask_fn, 12, 12)

        # Every query must retain at least one key, or its softmax row is empty.
        for q in range(12):
            assert mask[q].any(), f"query {q} has an empty attention row"
