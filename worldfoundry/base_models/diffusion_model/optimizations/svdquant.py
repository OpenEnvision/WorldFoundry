"""Checkpoint-bound SVDQuant adapter for explicitly calibrated native linears."""

from pathlib import Path

from worldfoundry.core.acceleration.guards import guard_fixed_inference
from worldfoundry.core.acceleration.plugins import AccelerationHandle, PreparedAcceleration
from worldfoundry.core.acceleration.quantization.calibration import load_calibration
from worldfoundry.core.acceleration.quantization.svdquant import PackedSVDQuantLinear, validate_svdquant_state

from .projection_selection import ProjectionSelection, select_native_projections


def prepare_svdquant(model, options, policy):
    if set(options) != {"artifact"} or not isinstance(options["artifact"], (str, Path)):
        raise ValueError("svdquant requires an explicit calibration artifact path")
    artifact = load_calibration(options["artifact"], kind="svdquant")
    states = artifact["states"]
    selection = ProjectionSelection(include=tuple(states), min_features=16)
    modules, selected = select_native_projections(model, selection, policy)
    if {path for path, _ in selected} != set(states):
        raise ValueError("SVDQuant artifact paths must exactly match eligible native projections")
    for path, source in selected:
        validate_svdquant_state(source, states[path])
    seams = {"diffusion.approximation", "diffusion.precision_cache"}
    seams.update(f"diffusion.projection.{path}" for path, _ in selected)
    if any(".cross_attn.k" in path or ".cross_attn.v" in path for path, _ in selected):
        seams.add("wan.cross_attention.projections")

    def activate():
        replacements = [(path, source, PackedSVDQuantLinear(source, states[path])) for path, source in selected]
        undo_guards = guard_fixed_inference(model, "svdquant", forbid_serialization=True)
        installed = []

        def undo():
            for parent, key, source in reversed(installed):
                setattr(parent, key, source)
            installed.clear()
            undo_guards()

        try:
            for path, source, replacement in replacements:
                parent, _, key = path.rpartition(".")
                setattr(modules[parent], key, replacement)
                installed.append((modules[parent], key, source))
        except BaseException:
            undo()
            raise
        return AccelerationHandle(
            "svdquant",
            {
                "approximate": True,
                "modules": list(states),
                "calibration": artifact["metadata"],
                "provider": "triton-packed-int4",
                "group_size": 64,
                "ranks": {path: state["rank"] for path, state in states.items()},
                "dense_storage": "retained-original-for-uninstall",
                "dense_fallback": False,
                "artifact_layout": "worldfoundry-svdquant-v1; not Nunchaku binary-compatible",
            },
            undo,
        )

    return PreparedAcceleration("svdquant", frozenset(seams), activate)
