import json
from pathlib import Path

import pytest

from worldfoundry.synthesis.visual_generation.world_model import runtime_manifest
from worldfoundry.synthesis.visual_generation.world_model.open_dreamer import vpt_actions
from worldfoundry.synthesis.visual_generation.world_model.open_dreamer import (
    worldfoundry_runtime as open_dreamer_runtime,
)


def _stage_checkout(tmp_path: Path) -> Path:
    """Create the minimal shape the adapter recognizes as a staged checkout."""
    root = tmp_path / "open-dreamer-inference"
    (root / "pipeline").mkdir(parents=True)
    (root / "inference.py").write_text("", encoding="utf-8")
    return root


# ── Action bridge ────────────────────────────────────────────


def test_split_interaction_token_handles_compound_names() -> None:
    assert vpt_actions.split_interaction_token("forward") == ["forward"]
    assert vpt_actions.split_interaction_token("forward_left") == ["forward", "left"]
    assert vpt_actions.split_interaction_token("forward_camera_l") == ["forward", "camera_l"]
    assert vpt_actions.split_interaction_token("camera_left") == ["camera_l"]
    assert vpt_actions.split_interaction_token("noop") == []


def test_split_interaction_token_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown Open Dreamer interaction"):
        vpt_actions.split_interaction_token("teleport")


def test_interaction_to_action_emits_vpt_keys_and_camera_deltas() -> None:
    action = vpt_actions.interaction_to_action("forward_camera_l", camera_step_degrees=5.0)

    assert action["keyboard"]["keys"] == ["key.keyboard.w"]
    # camera_l looks left, so dx is negative in raw VPT units.
    assert action["mouse"]["dx"] == pytest.approx(-5.0 * vpt_actions.RAW_UNITS_PER_DEGREE)
    assert action["mouse"]["dy"] == pytest.approx(0.0)
    assert action["mouse"]["buttons"] == []


def test_interaction_to_action_passes_through_vpt_dictionaries() -> None:
    raw = {"mouse": {"dx": 12.0, "dy": -3.0, "buttons": [0]}, "keyboard": {"keys": ["key.keyboard.space"]}}

    action = vpt_actions.interaction_to_action(raw)

    assert action["mouse"] == {"dx": 12.0, "dy": -3.0, "buttons": [0], "dwheel": 0.0}
    assert action["keyboard"]["keys"] == ["key.keyboard.space"]


def test_build_action_dicts_pads_context_with_noops_and_spreads_the_horizon() -> None:
    actions = vpt_actions.build_action_dicts(["forward", "camera_r"], context_frames=4, horizon=6)

    assert len(actions) == 10
    assert all(entry == vpt_actions.noop_action() for entry in actions[:4])
    assert all(entry["keyboard"]["keys"] == ["key.keyboard.w"] for entry in actions[4:7])
    assert all(entry["mouse"]["dx"] > 0 for entry in actions[7:])


def test_build_action_dicts_honours_explicit_frame_counts() -> None:
    actions = vpt_actions.build_action_dicts(
        [{"keys": ["forward"], "frames": 2}, {"keys": ["jump"], "frames": 3}],
        context_frames=1,
        horizon=5,
    )

    assert len(actions) == 6
    assert [entry["keyboard"]["keys"] for entry in actions[1:]] == [
        ["key.keyboard.w"],
        ["key.keyboard.w"],
        ["key.keyboard.space"],
        ["key.keyboard.space"],
        ["key.keyboard.space"],
    ]


def test_write_and_count_action_jsonl_roundtrip(tmp_path: Path) -> None:
    actions = vpt_actions.build_action_dicts(["forward"], context_frames=2, horizon=3)
    path = vpt_actions.write_action_jsonl(tmp_path / "actions.jsonl", actions)

    assert vpt_actions.count_action_entries(path) == 5


# ── Runtime manifest wiring ──────────────────────────────────


def test_runtime_spec_is_registered_and_source_bound() -> None:
    spec = runtime_manifest.runtime_spec("open-dreamer")

    assert spec.display_name == "Open Dreamer"
    assert spec.entrypoint_relative == "inference.py"
    # An empty blocked reason lets a fully staged install execute; gating lives in
    # the adapter's missing_requirements instead.
    assert spec.blocked_reason == ""

    root, entrypoint, blocked_reason = runtime_manifest.resolve_runtime_manifest(spec)
    assert entrypoint == root / "inference.py"
    assert blocked_reason == ""


def test_no_upstream_source_is_vendored() -> None:
    """Open Dreamer is all-rights-reserved, so only WorldFoundry glue may live here."""
    package_dir = open_dreamer_runtime.RUNTIME_DIR
    modules = sorted(path.name for path in package_dir.glob("*.py"))

    assert modules == ["__init__.py", "vpt_actions.py", "worldfoundry_runtime.py"]


# ── Requirement gating ───────────────────────────────────────


def test_missing_requirements_reports_unstaged_checkout(tmp_path: Path) -> None:
    root = tmp_path / "not-staged"

    missing = open_dreamer_runtime.missing_requirements(
        options={},
        runtime_root=root,
        entrypoint=root / "inference.py",
        profile=None,
    )

    kinds = {entry["kind"] for entry in missing}
    assert "source_repo" in kinds
    assert "entrypoint" in kinds
    assert "checkpoint" in kinds
    assert any(entry["path"] == "input_mp4" for entry in missing)


def test_missing_requirements_is_empty_once_everything_is_staged(tmp_path: Path) -> None:
    root = _stage_checkout(tmp_path)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    clip = tmp_path / "context.mp4"
    clip.write_bytes(b"")

    missing = open_dreamer_runtime.missing_requirements(
        options={
            "checkpoint_path": str(checkpoint),
            "input_mp4": str(clip),
            # A non-WorldFoundry interpreter means JAX lives in the checkout's venv.
            "python_executable": str(root / ".venv" / "bin" / "python"),
        },
        runtime_root=root,
        entrypoint=root / "inference.py",
        profile=None,
    )

    assert missing == []


def test_missing_requirements_flags_a_missing_action_file(tmp_path: Path) -> None:
    root = _stage_checkout(tmp_path)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    clip = tmp_path / "context.mp4"
    clip.write_bytes(b"")

    missing = open_dreamer_runtime.missing_requirements(
        options={
            "checkpoint_path": str(checkpoint),
            "input_mp4": str(clip),
            "actions_path": str(tmp_path / "absent.jsonl"),
            "python_executable": str(root / ".venv" / "bin" / "python"),
        },
        runtime_root=root,
        entrypoint=root / "inference.py",
        profile=None,
    )

    assert [entry["kind"] for entry in missing] == ["asset"]


# ── Command construction ─────────────────────────────────────


def _context(tmp_path: Path, *, options: dict, plan: dict) -> dict:
    root = _stage_checkout(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True)
    plan_path = output_dir / "open-dreamer.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return {
        "python": "/unused/python",
        "runtime_root": str(root),
        "entrypoint": str(root / "inference.py"),
        "output_path": str(output_dir / "open-dreamer.mp4"),
        "output_dir": str(output_dir),
        "prompt": "",
        "device": "cuda",
        "plan_path": str(plan_path),
        "options": options,
        "profile": None,
    }


def test_build_command_matches_the_official_entrypoint_contract(tmp_path: Path) -> None:
    clip = tmp_path / "context.mp4"
    clip.write_bytes(b"")
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    context = _context(
        tmp_path,
        options={
            "checkpoint_path": str(checkpoint),
            "input_mp4": str(clip),
            "python_executable": "/opt/open-dreamer/bin/python",
        },
        plan={"interactions": ["forward", "camera_r"], "extra": {"horizon": 8, "context_frames": 2}},
    )

    command = open_dreamer_runtime.build_command(context)

    assert command[0] == "env"
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in command
    assert "/opt/open-dreamer/bin/python" in command
    assert command[command.index("--checkpoint_path") + 1] == str(checkpoint)
    assert command[command.index("--input_mp4") + 1] == str(clip)
    assert command[command.index("--horizon") + 1] == "8"
    assert command[command.index("--context_frames") + 1] == "2"
    assert command[command.index("--output_mp4") + 1] == context["output_path"]
    assert "--use_ema" in command

    actions_path = Path(command[command.index("--actions_path") + 1])
    assert actions_path.is_file()
    assert vpt_actions.count_action_entries(actions_path) == 10


def test_build_command_prefers_call_time_kwargs_over_load_options(tmp_path: Path) -> None:
    clip = tmp_path / "context.mp4"
    clip.write_bytes(b"")
    other_clip = tmp_path / "other.mp4"
    other_clip.write_bytes(b"")
    context = _context(
        tmp_path,
        options={"input_mp4": str(clip), "horizon": 64, "python_executable": "/opt/od/bin/python"},
        plan={"interactions": [], "extra": {"input_mp4": str(other_clip), "horizon": 4, "use_ema": False}},
    )

    command = open_dreamer_runtime.build_command(context)

    assert command[command.index("--input_mp4") + 1] == str(other_clip)
    assert command[command.index("--horizon") + 1] == "4"
    assert "--use_ema" not in command


def test_build_command_reuses_a_supplied_action_file(tmp_path: Path) -> None:
    clip = tmp_path / "context.mp4"
    clip.write_bytes(b"")
    supplied = vpt_actions.write_action_jsonl(
        tmp_path / "supplied.jsonl",
        vpt_actions.build_action_dicts(
            ["forward"],
            context_frames=open_dreamer_runtime.DEFAULT_CONTEXT_FRAMES,
            horizon=open_dreamer_runtime.DEFAULT_HORIZON,
        ),
    )
    context = _context(
        tmp_path,
        options={"input_mp4": str(clip), "actions_path": str(supplied), "python_executable": "/opt/od/bin/python"},
        plan={"interactions": ["forward"], "extra": {}},
    )

    command = open_dreamer_runtime.build_command(context)

    assert command[command.index("--actions_path") + 1] == str(supplied)


def test_build_command_rejects_a_short_action_file(tmp_path: Path) -> None:
    clip = tmp_path / "context.mp4"
    clip.write_bytes(b"")
    supplied = vpt_actions.write_action_jsonl(
        tmp_path / "short.jsonl",
        vpt_actions.build_action_dicts(["forward"], context_frames=1, horizon=1),
    )
    context = _context(
        tmp_path,
        options={"input_mp4": str(clip), "actions_path": str(supplied), "python_executable": "/opt/od/bin/python"},
        plan={"interactions": ["forward"], "extra": {}},
    )

    with pytest.raises(ValueError, match="needs 80 actions"):
        open_dreamer_runtime.build_command(context)


def test_build_command_requires_a_context_clip(tmp_path: Path) -> None:
    context = _context(tmp_path, options={}, plan={"interactions": [], "extra": {}})

    with pytest.raises(ValueError, match="requires input_mp4"):
        open_dreamer_runtime.build_command(context)


# ── Pipeline call surface ────────────────────────────────────


def test_pipeline_promotes_a_video_path_into_input_mp4(tmp_path: Path) -> None:
    """`video=` must survive into the rollout command, which only sees plan extras."""
    from worldfoundry.pipelines.world_model import OpenDreamerPipeline

    captured: dict = {}

    class _Recorder(OpenDreamerPipeline):
        def _call_component_pipeline(self, *args, **kwargs):
            captured.update(kwargs)
            return {"artifact_path": ""}

    pipeline = _Recorder(model_id="open-dreamer", synthesis_model=object(), operators=object())
    pipeline(video=str(tmp_path / "context.mp4"))

    assert captured["input_mp4"] == str(tmp_path / "context.mp4")


def test_pipeline_leaves_non_path_inputs_alone(tmp_path: Path) -> None:
    from worldfoundry.pipelines.world_model import OpenDreamerPipeline

    captured: dict = {}

    class _Recorder(OpenDreamerPipeline):
        def _call_component_pipeline(self, *args, **kwargs):
            captured.update(kwargs)
            return {"artifact_path": ""}

    pipeline = _Recorder(model_id="open-dreamer", synthesis_model=object(), operators=object())
    pipeline(images=object())

    assert "input_mp4" not in captured


def test_pipeline_scopes_gated_options_to_a_single_call(tmp_path: Path) -> None:
    """missing_requirements only sees load-time options, so the call must seed them."""
    from worldfoundry.pipelines.world_model import OpenDreamerPipeline

    class _Synthesis:
        def __init__(self) -> None:
            self.options = {"checkpoint_path": "/staged/ckpt"}
            self.seen: list[dict] = []

    synthesis = _Synthesis()

    class _Recorder(OpenDreamerPipeline):
        def _call_component_pipeline(self, *args, **kwargs):
            synthesis.seen.append(dict(synthesis.options))
            return {"artifact_path": ""}

    pipeline = _Recorder(model_id="open-dreamer", synthesis_model=synthesis, operators=object())
    pipeline(video=str(tmp_path / "context.mp4"))

    assert synthesis.seen[0]["input_mp4"] == str(tmp_path / "context.mp4")
    assert synthesis.seen[0]["checkpoint_path"] == "/staged/ckpt"
    # The clip must not leak into a later rollout that supplies its own input.
    assert synthesis.options == {"checkpoint_path": "/staged/ckpt"}
