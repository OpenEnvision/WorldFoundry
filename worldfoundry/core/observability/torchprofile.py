"""Optional TorchProfile telemetry for WorldFoundry model runs.

The collector is intentionally explicit: tracing a model is neither free nor
universally compatible with dynamic generation pipelines.  Runtime adapters
call :func:`profile_torch_module` once they have a real model and a
representative positional input tuple.  The resulting static-compute summary
is emitted through the regular structured logging pipeline, so it inherits
the current run/job/model context and can also be copied to a dedicated JSONL
artifact for later comparison.

``torchprofile`` counts multiply-accumulate operations (MACs) via
``torch.jit.trace``.  Its FLOPs value is therefore an explicit estimate using
the conventional ``2 FLOPs = 1 MAC`` conversion; it is not a latency or memory
measurement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from worldfoundry.core.observability.logging_setup import get_logger, write_jsonl_event

TORCHPROFILE_SCHEMA_VERSION = "worldfoundry.torchprofile.v1"

__all__ = [
    "TORCHPROFILE_SCHEMA_VERSION",
    "TorchProfileResult",
    "profile_torch_module",
]


@dataclass(frozen=True, slots=True)
class TorchProfileResult:
    """A compact static-compute result from one optional TorchProfile trace."""

    status: Literal["completed", "unavailable", "failed"]
    macs: int | None = None
    flops_estimate: int | None = None
    parameter_count: int | None = None
    torchprofile_version: str | None = None
    model_class: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation suitable for result payloads."""

        return {
            "schema_version": TORCHPROFILE_SCHEMA_VERSION,
            "status": self.status,
            "macs": self.macs,
            "flops_estimate": self.flops_estimate,
            "flops_convention": "two_flops_per_mac",
            "parameter_count": self.parameter_count,
            "torchprofile_version": self.torchprofile_version,
            "model_class": self.model_class,
            "error": self.error,
        }


def _torchprofile_version(module: Any) -> str | None:
    """Resolve the installed dependency version without making it mandatory."""

    value = getattr(module, "__version__", None)
    if value:
        return str(value)
    try:
        return metadata.version("torchprofile")
    except metadata.PackageNotFoundError:
        return None


def _parameter_count(model: Any) -> int | None:
    """Return parameter cardinality when *model* exposes the PyTorch protocol."""

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return sum(int(parameter.numel()) for parameter in parameters())
    except Exception:
        return None


def _preserve_training_mode(model: Any) -> tuple[tuple[Any, bool], ...]:
    """Switch an ``nn.Module`` tree to evaluation mode and retain every flag."""

    modules = getattr(model, "modules", None)
    set_eval = getattr(model, "eval", None)
    if not callable(modules) or not callable(set_eval):
        return ()
    try:
        states = tuple((module, bool(module.training)) for module in modules())
        set_eval()
        return states
    except Exception:
        return ()


def _restore_training_mode(states: tuple[tuple[Any, bool], ...]) -> None:
    """Restore module training flags captured by :func:`_preserve_training_mode`."""

    for module, training in states:
        try:
            module.training = training
        except Exception:
            continue


def _event_fields(
    result: TorchProfileResult,
    *,
    input_spec: Mapping[str, Any] | None,
    event_log_path: str | Path | None,
) -> dict[str, Any]:
    """Build the bounded event payload shared by the configured and manual sinks."""

    fields: dict[str, Any] = {
        "profile_schema_version": TORCHPROFILE_SCHEMA_VERSION,
        "source": "torchprofile",
        "model_class": result.model_class,
        "torchprofile_version": result.torchprofile_version,
        "macs": result.macs,
        "flops_estimate": result.flops_estimate,
        "flops_convention": "two_flops_per_mac",
        "parameter_count": result.parameter_count,
    }
    if input_spec is not None:
        # Callers supply dimensions/dtypes here, never tensor contents.  The
        # shared logging formatter will also redact sensitive field names.
        fields["input_spec"] = dict(input_spec)
    if event_log_path is not None:
        fields["event_log_path"] = str(event_log_path)
    return fields


def _emit_result(
    result: TorchProfileResult,
    *,
    model_id: str | None,
    input_spec: Mapping[str, Any] | None,
    event_log_path: str | Path | None,
    exception: BaseException | None = None,
) -> None:
    """Send a result to the configured logger and optional run-owned JSONL file."""

    if result.status == "completed":
        level, event, message = "INFO", "torchprofile.completed", "TorchProfile trace completed"
    elif result.status == "unavailable":
        level, event, message = "WARNING", "torchprofile.unavailable", "TorchProfile is unavailable"
    else:
        level, event, message = "WARNING", "torchprofile.failed", "TorchProfile trace failed"

    fields = _event_fields(result, input_spec=input_spec, event_log_path=event_log_path)
    logger = get_logger(__name__)
    logger.event(
        level,
        event,
        message,
        exc_info=exception if exception is not None else None,
        model_id=model_id,
        error=result.error,
        **fields,
    )
    if event_log_path is not None:
        write_jsonl_event(
            event_log_path,
            level=level,
            event=event,
            message=message,
            logger_name=__name__,
            exception=exception,
            model_id=model_id,
            error=result.error,
            **fields,
        )


def profile_torch_module(
    model: Any,
    inputs: Any = (),
    *,
    model_id: str | None = None,
    input_spec: Mapping[str, Any] | None = None,
    event_log_path: str | Path | None = None,
    set_eval: bool = True,
    strict: bool = False,
) -> TorchProfileResult:
    """Trace *model* with TorchProfile and log a static compute summary.

    Args:
        model: A loaded PyTorch module.  The caller retains ownership of the
            model and its device placement.
        inputs: One tensor or tuple of positional tensors representative of
            the inference workload.  ``torchprofile`` does not support
            keyword-argument tracing.
        model_id: Optional model identifier; promoted to a top-level field in
            WorldFoundry JSON logs.
        input_spec: Optional JSON-safe shape/dtype metadata for the event. Do
            not place actual tensor values or credentials in this mapping.
        event_log_path: Optional run-owned JSONL path that receives the same
            canonical structured event even when no logger sink is configured.
        set_eval: Temporarily call ``model.eval()`` for deterministic tracing
            and restore all module training flags afterwards.
        strict: Re-raise unavailable/profile failures after emitting the event.

    Returns:
        A :class:`TorchProfileResult`.  With ``strict=False`` (the default),
        missing optional dependencies and trace incompatibilities are reported
        as result statuses rather than breaking the underlying model run.
    """

    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    try:
        import torchprofile
    except Exception as exc:  # Optional dependency may be absent or broken.
        result = TorchProfileResult(status="unavailable", model_class=model_class, error=str(exc))
        _emit_result(
            result,
            model_id=model_id,
            input_spec=input_spec,
            event_log_path=event_log_path,
            exception=exc,
        )
        if strict:
            raise
        return result

    states = _preserve_training_mode(model) if set_eval else ()
    try:
        macs = int(torchprofile.profile_macs(model, inputs))
        result = TorchProfileResult(
            status="completed",
            macs=macs,
            flops_estimate=macs * 2,
            parameter_count=_parameter_count(model),
            torchprofile_version=_torchprofile_version(torchprofile),
            model_class=model_class,
        )
    except Exception as exc:  # Trace compatibility is model- and input-specific.
        result = TorchProfileResult(
            status="failed",
            parameter_count=_parameter_count(model),
            torchprofile_version=_torchprofile_version(torchprofile),
            model_class=model_class,
            error=str(exc),
        )
        _emit_result(
            result,
            model_id=model_id,
            input_spec=input_spec,
            event_log_path=event_log_path,
            exception=exc,
        )
        if strict:
            raise
        return result
    finally:
        _restore_training_mode(states)

    _emit_result(
        result,
        model_id=model_id,
        input_spec=input_spec,
        event_log_path=event_log_path,
    )
    return result
