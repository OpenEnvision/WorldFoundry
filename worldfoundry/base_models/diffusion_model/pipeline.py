"""Small public construction surface around a fully built native runner.

This module does not implement the sampling loop.  It only:

1. Builds a runner through ``from_pretrained`` / ``from_registry``.
2. Forwards :class:`DiffusionRequest` to ``runner.run`` from ``__call__``.
3. Exposes ``model_id`` / ``components`` / ``device`` / ``dtype`` for
   higher-level orchestration.

Do not put family-specific logic here.  Variants belong in a recipe, an
execution strategy, or an instance-local extension.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from time import perf_counter
from typing import cast

from .components import ComponentKey
from .contracts import DiffusionOutput, DiffusionRequest
from .extensions import DiffusionExtension
from .loaders import CheckpointSpec
from .optimizations import RuntimePolicy
from .registry import NativeDiffusionRegistry
from .runners import DiffusionExecutor


class NativeDiffusionPipeline:
    """Request boundary around one fully constructed :class:`DiffusionExecutor`.

    Callers should depend on this class plus :class:`DiffusionRequest` /
    :class:`DiffusionOutput`, not on the assembler or strategy registry.
    """

    def __init__(self, runner: DiffusionExecutor) -> None:
        self.runner = runner

    @property
    def model_id(self) -> str:
        """Canonical recipe id (not an alias)."""
        return self.runner.model_id

    @property
    def components(self):
        """Expose the assembled role bundle for training or visualization.

        Some execution strategies do not expose ``components``; fail loudly
        instead of returning ``None``.
        """

        try:
            return self.runner.components
        except AttributeError as error:
            raise AttributeError("this diffusion execution strategy does not expose a component bundle") from error

    @property
    def device(self):
        """Device that actually holds the runner weights."""
        return self.runner.device

    @property
    def dtype(self):
        """Inference compute dtype, usually taken from :class:`RuntimePolicy`."""
        return self.runner.dtype

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        registry: NativeDiffusionRegistry | None = None,
        policy: RuntimePolicy | None = None,
        checkpoint_overrides: Mapping[str, CheckpointSpec | str] | None = None,
        component_options: Mapping[str | ComponentKey, Mapping[str, object]] | None = None,
        extensions: Iterable[DiffusionExtension] = (),
    ) -> "NativeDiffusionPipeline":
        """Construct a built-in model id through the canonical assembler.

        When ``registry`` is omitted, the frozen
        :func:`default_native_diffusion_registry` is used.
        ``checkpoint_overrides`` may replace a Hub source with a local
        directory or a single file.  ``extensions`` attach to this runner
        instance only.
        """

        if registry is None:
            from .registry import default_native_diffusion_registry

            registry = default_native_diffusion_registry()
        return cls.from_registry(
            registry,
            model_id,
            policy=policy,
            checkpoint_overrides=checkpoint_overrides,
            component_options=component_options,
            extensions=extensions,
        )

    @classmethod
    def from_registry(
        cls,
        registry: NativeDiffusionRegistry,
        model_id: str,
        *,
        policy: RuntimePolicy | None = None,
        checkpoint_overrides: Mapping[str, CheckpointSpec | str] | None = None,
        component_options: Mapping[str | ComponentKey, Mapping[str, object]] | None = None,
        extensions: Iterable[DiffusionExtension] = (),
    ) -> "NativeDiffusionPipeline":
        """Resolve a recipe on ``registry`` and assemble its runner."""
        return cls(
            registry.build_runner(
                model_id,
                policy=policy,
                checkpoint_overrides=checkpoint_overrides,
                component_options=component_options,
                extensions=extensions,
            )
        )

    def warmup(
        self, requests: Iterable[DiffusionRequest | Sequence[DiffusionRequest]]
    ) -> tuple[Mapping[str, object], ...]:
        """Run representative singleton requests or explicit batches, retaining no outputs.

        Native standard, masked-latent and Wan dual-expert strategies own request
        cleanup. All sampling steps,
        CFG/expert branches and decoding execute with the supplied inputs. CUDA
        graphs capture during warmup and use eager execution for unseen serving
        signatures afterwards. Call between requests; custom runners/extensions
        require their own warmup lifecycle.
        """
        import torch

        from .extensions.frozen_mask import FrozenLatentMaskExtension
        from .runners.base import NativeDiffusionRunner, Wan22DualExpertGuidanceRunner

        cases = tuple(requests)
        if not cases:
            raise ValueError("warmup requires one or more DiffusionRequest values or batches")
        groups = []
        for case in cases:
            group = (case,) if isinstance(case, DiffusionRequest) else tuple(case) if isinstance(case, Sequence) else ()
            if not group or any(not isinstance(request, DiffusionRequest) for request in group):
                raise ValueError("warmup requires one or more DiffusionRequest values or batches")
            groups.append(group)
        runner = cast(NativeDiffusionRunner | Wan22DualExpertGuidanceRunner, self.runner)
        if type(runner) not in {NativeDiffusionRunner, Wan22DualExpertGuidanceRunner} or any(
            type(extension) is not FrozenLatentMaskExtension for extension in runner.extensions
        ):
            raise ValueError("warmup requires a native stateless runner without custom extensions")
        denoiser = self.components.denoiser
        prepare = getattr(denoiser, "execution_warmup", None)
        device = torch.device(self.device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        reports: list[Mapping[str, object]] = []
        with torch.random.fork_rng(devices=devices), (prepare() if callable(prepare) else nullcontext()):
            for case, group in zip(cases, groups):
                request = group[0]
                if devices:
                    torch.cuda.synchronize(device)
                started = perf_counter()
                output = self(request) if isinstance(case, DiffusionRequest) else self.run_batch(group)
                if devices:
                    torch.cuda.synchronize(device)
                elapsed = perf_counter() - started
                reporter = getattr(denoiser, "runtime_optimization_report", None)
                reports.append({
                    "height": request.height,
                    "width": request.width,
                    "num_frames": request.num_frames,
                    "batch_size": sum(item.batch_size for item in group),
                    "request_count": len(group),
                    "num_inference_steps": request.sampling.num_inference_steps,
                    "seconds": elapsed,
                    "denoiser": reporter() if callable(reporter) else None,
                })
                del output
        return tuple(reports)

    def run_batch(self, requests: Iterable[DiffusionRequest]) -> tuple[DiffusionOutput, ...]:
        """Execute compatible requests together and return independently owned outputs.

        Supported native Wan T2V/I2V runners retain per-request seeds and solver state.
        Calls on this pipeline must remain serialized, including warmup and
        singleton calls. Incompatible batches fail rather than retrying singles.
        """
        execute = getattr(self.runner, "run_batch", None)
        if not callable(execute):
            raise ValueError("this diffusion execution strategy does not support request batching")
        return execute(tuple(requests))

    def __call__(self, request: DiffusionRequest) -> DiffusionOutput:
        """Run one normalized request.  Bare dicts are rejected so contracts stay enforced."""
        if not isinstance(request, DiffusionRequest):
            raise TypeError(f"native diffusion pipeline expects DiffusionRequest, got {type(request).__name__}")
        return self.runner.run(request)


__all__ = ["NativeDiffusionPipeline"]
