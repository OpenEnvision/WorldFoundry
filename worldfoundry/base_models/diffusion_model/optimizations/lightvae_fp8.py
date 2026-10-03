"""Reversible calibrated encoder convolutions for the matched Wan21 student.

The canonical causal/cache implementation, attention and decoder remain owned
by the codec. Only explicit calibrated convolution calls use the FP8 provider.
"""

from worldfoundry.core.acceleration.guards import guard_fixed_inference
from worldfoundry.core.acceleration.plugins import AccelerationHandle, AccelerationRegistry, PreparedAcceleration
from worldfoundry.core.acceleration.quantization.calibration import load_calibration
from worldfoundry.core.acceleration.quantization.fp8_conv import (
    CalibratedFP8Convolution,
    validate_fp8_convolution_state,
)


def lightvae_encoder_convolutions(model):
    from torch import nn

    from ..models.autoencoders.wan.model import CausalConv3d
    from ..models.autoencoders.wan.variants.light_21 import Wan21LightVAE

    if type(model) is not Wan21LightVAE:
        raise ValueError("FP8 codec requires the checkpoint-matched Wan21LightVAE")
    return {
        name: module
        for name, module in model.named_modules()
        if (name.startswith("model.encoder.") or name == "model.conv1")
        and type(module) in {nn.Conv2d, nn.Conv3d, CausalConv3d}
    }


def _prepare(model, options, policy):
    del policy
    if set(options) != {"artifact"}:
        raise ValueError("lightvae_fp8 requires a calibration artifact")
    if any(module.training for module in model.modules()):
        raise ValueError("lightvae_fp8 requires model.eval()")
    artifact = load_calibration(options["artifact"], kind="lightvae_fp8")
    supported = lightvae_encoder_convolutions(model)
    states = artifact["states"]
    if not set(states) <= set(supported):
        raise ValueError("FP8 codec artifact names unsupported encoder convolutions")
    for name, state in states.items():
        module = supported[name]
        if "_conv_forward" in module.__dict__ or module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("FP8 codec requires original unhooked convolution calls")
        validate_fp8_convolution_state(module, state)

    def activate():
        executors = {name: CalibratedFP8Convolution(supported[name], state) for name, state in states.items()}
        undo_guards = guard_fixed_inference(model, "lightvae_fp8")
        installed = []

        def undo():
            for module in installed:
                module.__dict__.pop("_conv_forward", None)
            installed.clear()
            undo_guards()
            model.__dict__.pop("_worldfoundry_fp8_codec", None)

        try:
            for name, executor in executors.items():
                supported[name]._conv_forward = executor
                installed.append(supported[name])
            model._worldfoundry_fp8_codec = executors
        except BaseException:
            undo()
            raise
        return AccelerationHandle(
            "lightvae_fp8",
            {
                "approximate": True,
                "provider": "triton-fp8-implicit-convolution",
                "modules": list(states),
                "calibration": artifact["metadata"],
                "normalization_and_cache_dtype": "float32",
                "dense_fallback": False,
            },
            undo,
        )

    return PreparedAcceleration("lightvae_fp8", frozenset({"codec.encoder.convolutions"}), activate)


def install_lightvae_fp8(model, artifact):
    registry = AccelerationRegistry()
    registry.register("lightvae_fp8", _prepare)
    return registry.install(model, {"lightvae_fp8": {"artifact": artifact}})


def lightvae_fp8_report(model):
    states = getattr(model, "_worldfoundry_fp8_codec", {})
    layers = {name: state.report() for name, state in states.items()}
    return {
        "enabled": bool(states),
        "layers": layers,
        "kernel_calls": sum(value["kernel_calls"] for value in layers.values()),
        "clipped_input_operands": sum(value["clipped_input_operands"] for value in layers.values()),
        "dense_fallback_calls": 0,
    }
