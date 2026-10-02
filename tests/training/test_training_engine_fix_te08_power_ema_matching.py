"""TE-08 regression: PowerEMA must not silently skip unmatched parameters.

``PowerEMA.forward``/``copy_to`` used to ``continue`` past any parameter
whose name was absent from the shadow map.  If the module was re-wrapped
after EMA construction (container module, PEFT injection, FSDP1 flattening)
every name gained a prefix and the EMA silently stopped updating -- the
exported "EMA" weights stayed near initialization with no error anywhere.
Both methods now require every tracked shadow to resolve to a live
parameter before mutating anything.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.ema import PowerEMA  # noqa: E402


def _model() -> torch.nn.Module:
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2, bias=True)
    return model


def test_power_ema_updates_and_exports_when_names_match() -> None:
    model = _model()
    ema = PowerEMA(model, rate=0.1)

    with torch.no_grad():
        model.weight.add_(1.0)
    ema(model)

    assert int(ema.num_updates.item()) == 1
    # Iteration 0 has beta == 0: shadows must equal the current parameters.
    shadow = getattr(ema, ema._shadow_names["weight"])
    torch.testing.assert_close(shadow, model.weight.detach(), rtol=0, atol=0)

    with torch.no_grad():
        model.weight.mul_(5.0)
    ema.copy_to(model)
    torch.testing.assert_close(model.weight.detach(), shadow, rtol=0, atol=0)


def test_power_ema_ignores_frozen_parameters_it_never_tracked() -> None:
    model = _model()
    model.bias.requires_grad_(False)
    ema = PowerEMA(model, rate=0.1)

    assert "bias" not in ema._shadow_names
    ema(model)
    assert int(ema.num_updates.item()) == 1


def test_power_ema_raises_when_module_is_rewrapped_after_construction() -> None:
    model = _model()
    ema = PowerEMA(model, rate=0.1)
    wrapped = torch.nn.Sequential(model)  # every name gains a "0." prefix

    with pytest.raises(RuntimeError, match="re-wrapped or renamed"):
        ema(wrapped)
    # Validation failed before mutation: no update was recorded.
    assert int(ema.num_updates.item()) == 0

    with pytest.raises(RuntimeError, match="re-wrapped or renamed"):
        ema.copy_to(wrapped)


def test_power_ema_raises_when_a_tracked_parameter_disappears() -> None:
    model = _model()
    ema = PowerEMA(model, rate=0.1)
    smaller = torch.nn.Linear(3, 2, bias=False)

    with pytest.raises(RuntimeError, match="1 of 2 tracked"):
        ema(smaller)
