"""Fail-closed numerical and execution gates, run before benchmark timing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch


@dataclass(frozen=True)
class NumericalBudget:
    atol: float
    rtol: float
    relative_l2: float

    def __post_init__(self) -> None:
        if any(not isfinite(value) or value < 0 for value in (self.atol, self.rtol, self.relative_l2)):
            raise ValueError("numerical budgets must be finite and nonnegative")


EXACT_BUDGET = NumericalBudget(atol=1e-5, rtol=1e-5, relative_l2=1e-5)
# Operator/trajectory acceptance only. This does not certify checkpoint video quality.
FP8_BUDGET = NumericalBudget(atol=0.15, rtol=0.03, relative_l2=0.03)


def validate_steps(steps: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")


def compare_step(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    step: int,
    budget: NumericalBudget = EXACT_BUDGET,
) -> dict[str, float | int]:
    """Require shape, finiteness, elementwise and trajectory-error parity."""
    prefix = f"correctness failed at step {step}"
    if reference.shape != actual.shape or reference.numel() == 0:
        raise AssertionError(f"{prefix}: nonempty matching shapes required ({reference.shape}, {actual.shape})")
    ref, got = reference.detach().float(), actual.detach().float()
    if not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(got).all()):
        raise AssertionError(f"{prefix}: NaN/Inf output")
    delta = got - ref
    error_norm = torch.linalg.vector_norm(delta.double())
    reference_norm = torch.linalg.vector_norm(ref.double())
    relative_l2 = (
        float(error_norm / reference_norm)
        if float(reference_norm) > 0
        else (0.0 if float(error_norm) == 0 else float("inf"))
    )
    metrics = {
        "step": step,
        "relative_l2": relative_l2,
        "rmse": float(delta.double().square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
    }
    torch.testing.assert_close(got, ref, atol=budget.atol, rtol=budget.rtol, msg=prefix)
    if relative_l2 > budget.relative_l2:
        raise AssertionError(f"{prefix}: relative L2 {relative_l2:.6g} > {budget.relative_l2:.6g}")
    return metrics


def require_graph_execution(report: dict[str, Any], *, steps: int) -> None:
    """A requested graph or lifetime counter cannot certify this execution."""
    validate_steps(steps)
    if (
        not report.get("enabled")
        or int(report.get("graphs", 0)) < 1
        or int(report.get("replay", 0)) < steps
        or int(report.get("eager", 0)) != 0
        or int(report.get("capture_failed", 0)) != 0
    ):
        raise AssertionError(f"CUDA Graph did not execute every correctness step: {report}")


__all__ = ["EXACT_BUDGET", "FP8_BUDGET", "NumericalBudget", "compare_step", "require_graph_execution", "validate_steps"]
