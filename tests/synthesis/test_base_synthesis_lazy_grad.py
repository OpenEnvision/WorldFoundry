"""Runtime discovery stays light while prediction retains Torch's no-grad contract."""

from __future__ import annotations

from abc import ABCMeta

import pytest
import torch

from worldfoundry.synthesis.base_synthesis import BaseSynthesis, _lazy_no_grad


@pytest.mark.parametrize("enabled", [False, True])
def test_real_torch_prediction_disables_grad_and_restores_caller_state(enabled):
    @_lazy_no_grad
    def operation(value):
        assert not torch.is_grad_enabled()
        return value.square()

    assert operation.__name__ == "operation"
    with torch.set_grad_enabled(enabled):
        source = torch.tensor([2.0, 3.0], requires_grad=True)
        result = operation(source)
        torch.testing.assert_close(result, torch.tensor([4.0, 9.0]))
        assert not result.requires_grad
        assert torch.is_grad_enabled() is enabled


def test_base_predict_exception_occurs_under_no_grad_and_restores_caller():
    observations = []

    class ProbeMeta(ABCMeta):
        def __getattribute__(cls, name):
            if name == "__name__":
                observations.append(torch.is_grad_enabled())
            return super().__getattribute__(name)

    class Backend(BaseSynthesis, metaclass=ProbeMeta):
        pass

    with torch.enable_grad():
        with pytest.raises(NotImplementedError, match="Backend.predict"):
            Backend().predict()
        assert observations == [False]
        assert torch.is_grad_enabled()


def test_generator_retains_torch_no_grad_decorator_semantics():
    @_lazy_no_grad
    def predictions(source):
        for offset in range(2):
            assert not torch.is_grad_enabled()
            yield source + offset

    with torch.enable_grad():
        generator = predictions(torch.tensor(2.0, requires_grad=True))
        assert torch.is_grad_enabled()
        for expected in (2.0, 3.0):
            value = next(generator)
            assert float(value) == expected
            assert not value.requires_grad
            assert torch.is_grad_enabled()
        with pytest.raises(StopIteration):
            next(generator)
        assert torch.is_grad_enabled()
