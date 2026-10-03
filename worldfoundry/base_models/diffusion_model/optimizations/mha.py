"""Native Wan MHA schedules; canonical norms, RoPE and checkpoint keys survive."""

from dataclasses import asdict

import torch

from worldfoundry.core.acceleration.guards import guard_fixed_inference
from worldfoundry.core.acceleration.plugins import AccelerationHandle, PreparedAcceleration
from worldfoundry.core.attention.schedule import MHASchedule, ScheduledProjection, scheduled_sdpa

from .projection_selection import ProjectionSelection, select_native_projections


class _WanMHAProcessor:
    def __init__(self, attention, scope, schedule):
        self.scope, self.schedule = scope, schedule
        precision = schedule.projection_precision
        if scope == "self" and schedule.fusion == "qkv":
            self.projections = {"qkv": ScheduledProjection((attention.q, attention.k, attention.v), precision)}
        else:
            self.projections = {"q": ScheduledProjection((attention.q,), precision)}
            for suffix in ("", "_img") if getattr(attention, "has_image_input", False) else ("",):
                key, value = getattr(attention, "k" + suffix), getattr(attention, "v" + suffix)
                if schedule.fusion != "none":
                    self.projections["kv" + suffix] = ScheduledProjection((key, value), precision)
                else:
                    self.projections["k" + suffix] = ScheduledProjection((key,), precision)
                    self.projections["v" + suffix] = ScheduledProjection((value,), precision)
        self.projections["o"] = ScheduledProjection((attention.o,), schedule.output_precision)
        self.calls = self.sdpa_calls = 0

    def _kv(self, x, suffix=""):
        if "kv" + suffix in self.projections:
            return self.projections["kv" + suffix](x)
        return self.projections["k" + suffix](x)[0], self.projections["v" + suffix](x)[0]

    def _attend(self, attention, q, k, v):
        output = scheduled_sdpa(q, k, v, num_heads=attention.num_heads, schedule=self.schedule)
        self.sdpa_calls += 1
        return output

    def __call__(self, attention, x, condition, **kwargs):
        if torch.is_grad_enabled() or torch.compiler.is_compiling() or attention.training:
            raise RuntimeError("optimized_mha requires eager eval no-grad inference")
        if x.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("optimized_mha CUDA Graph capture is unvalidated")
        if self.scope == "self":
            from ..models.networks.wan.model import apply_wan_qk_norm_rope

            if "qkv" in self.projections:
                q, k, v = self.projections["qkv"](x)
            else:
                q = self.projections["q"](x)[0]
                k, v = self._kv(x)
            q, k = apply_wan_qk_norm_rope(
                attention,
                q,
                k,
                condition,
                fused_table=kwargs.get("_worldfoundry_rope_table"),
                fused_grid=kwargs.get("_worldfoundry_rope_grid"),
                precision=kwargs.get("_worldfoundry_rope_precision", "fp64"),
            )
            output = self._attend(attention, q, k, v)
        else:
            q = attention.norm_q(self.projections["q"](x)[0])
            text = condition[:, 257:] if attention.has_image_input else condition
            k, v = self._kv(text)
            output = self._attend(attention, q, attention.norm_k(k), v)
            if attention.has_image_input:
                k, v = self._kv(condition[:, :257], "_img")
                output = output + self._attend(attention, q, attention.norm_k_img(k), v)
        output = self.projections["o"](output)[0]
        self.calls += 1
        return output

    def reset_request_window(self):
        self.calls = self.sdpa_calls = 0
        for projection in self.projections.values():
            projection.reset_request_window()

    def report(self):
        return {
            "calls": self.calls,
            "sdpa_calls": self.sdpa_calls,
            "projections": {name: projection.report() for name, projection in self.projections.items()},
        }


def prepare_optimized_mha(model, options, policy):
    from ..models.networks.wan.model import WanModel

    if type(model) is not WanModel or not options or set(options) - {"self", "cross"}:
        raise ValueError("optimized_mha requires native WanModel and explicit self/cross schedules")
    schedules = {scope: MHASchedule(**values) for scope, values in options.items()}
    paths = [
        f"blocks.{index}.{scope}_attn.{projection}"
        for index, block in enumerate(model.blocks)
        for scope in schedules
        for projection in (
            ("q", "k", "v", "o", "k_img", "v_img")
            if scope == "cross" and block.cross_attn.has_image_input
            else ("q", "k", "v", "o")
        )
    ]
    _, selected = select_native_projections(model, ProjectionSelection(include=paths, min_features=16), policy)
    device, dtype = selected[0][1].weight.device, selected[0][1].weight.dtype
    if any(module.weight.device != device or module.weight.dtype != dtype for _, module in selected):
        raise ValueError("optimized_mha requires a uniform projection device and dtype")
    for schedule in schedules.values():
        if schedule.sdpa_backend != "torch" or schedule.approximate:
            if device.type != "cuda" or dtype not in {torch.float16, torch.bfloat16}:
                raise ValueError("accelerated MHA schedules require CUDA FP16/BF16 policy")
        if schedule.sdpa_backend == "fa2":
            for block in model.blocks:
                head_dim = block.self_attn.q.out_features // block.self_attn.num_heads
                if not 16 <= head_dim <= 256 or head_dim & (head_dim - 1):
                    raise ValueError("FA2 requires power-of-two head dimensions in [16, 256]")
            if torch.cuda.get_device_capability(device)[0] < 9:
                raise ValueError("FA2 schedules require NVIDIA SM90 or newer")
            if not schedule.use_tma:
                from worldfoundry.core.attention.backends.triton_fa2 import flash_attention_2

                del flash_attention_2  # Import validates optional provider availability before activation.
        if schedule.sdpa_backend == "cudnn":
            if not torch.backends.cudnn.is_available():
                raise ValueError("cuDNN schedule requires an available cuDNN provider")
            if schedule.quantized_sdpa:
                import importlib

                if torch.cuda.get_device_capability(device)[0] < 9:
                    raise ValueError("FP8 cuDNN schedules require NVIDIA SM90 or newer")
                for block in model.blocks:
                    if block.self_attn.q.out_features // block.self_attn.num_heads not in {64, 128}:
                        raise ValueError("FP8 cuDNN schedules require head dimensions 64 or 128")
                importlib.import_module("cudnn")
                importlib.import_module("cuda.bindings.runtime")
        if schedule.use_tma:
            from worldfoundry.core.attention.backends.triton_tma import _require_triton_tma_version

            _require_triton_tma_version()
    targets = [
        (f"blocks.{index}.{scope}_attn", getattr(block, scope + "_attn"), scope)
        for index, block in enumerate(model.blocks)
        for scope in schedules
    ]
    seams = {"diffusion.precision_cache", *(f"wan.{scope}_attention.backend" for scope in schedules)}
    seams.update(f"diffusion.projection.{path}" for path in paths)
    if "cross" in schedules:
        seams.add("wan.cross_attention.projections")
    if any(schedule.approximate for schedule in schedules.values()):
        seams.add("diffusion.approximation")

    def activate():
        processors = {path: _WanMHAProcessor(module, scope, schedules[scope]) for path, module, scope in targets}
        originals = [(module, module.processor) for _, module, _ in targets]
        undo_guards = guard_fixed_inference(model, "optimized_mha")

        def undo():
            for module, processor in originals:
                module.processor = processor
            processors.clear()
            undo_guards()

        def reset():
            for processor in processors.values():
                processor.reset_request_window()

        try:
            for path, module, _ in targets:
                module.processor = processors[path]
        except BaseException:
            undo()
            raise
        return AccelerationHandle(
            "optimized_mha",
            {
                "schedules": {scope: asdict(value) for scope, value in schedules.items()},
                "modules": list(processors),
                "projection_modules": {path: list(processor.projections) for path, processor in processors.items()},
                "approximate": any(s.approximate for s in schedules.values()),
                "canonical_parameters_preserved": True,
                "placement": "fixed-until-uninstall",
            },
            undo,
            runtime_report=lambda: {path: processor.report() for path, processor in processors.items()},
            reset_request_window=reset,
        )

    return PreparedAcceleration("optimized_mha", frozenset(seams), activate)
