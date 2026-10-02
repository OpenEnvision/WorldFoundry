from __future__ import annotations

from abc import ABC

import pytest

pytest.importorskip("torch")

from worldfoundry.core.memory.base import BaseMemory
from worldfoundry.representations.base_representation import BaseRepresentation
from worldfoundry.synthesis.base_synthesis import BaseSynthesis


def test_base_contracts_remain_instantiable() -> None:
    assert isinstance(BaseSynthesis(), BaseSynthesis)
    assert isinstance(BaseRepresentation(), BaseRepresentation)
    assert isinstance(BaseMemory(), BaseMemory)


def test_base_contracts_are_abc_without_abstract_methods() -> None:
    for contract in (BaseSynthesis, BaseRepresentation, BaseMemory):
        assert issubclass(contract, ABC)
        assert contract.__abstractmethods__ == frozenset()


def test_unimplemented_model_contract_methods_raise_clear_errors() -> None:
    with pytest.raises(NotImplementedError, match="BaseSynthesis.from_pretrained"):
        BaseSynthesis.from_pretrained("model", args={})
    with pytest.raises(NotImplementedError, match="BaseSynthesis.predict"):
        BaseSynthesis().predict()
    with pytest.raises(NotImplementedError, match="BaseRepresentation.get_representation"):
        BaseRepresentation().get_representation({})


def test_base_memory_contract_methods_raise_clear_errors() -> None:
    memory = BaseMemory(capacity=2)

    assert memory.capacity == 2
    assert memory.storage == []
    assert memory.check_template() == {
        "required_keys": ("content", "type", "timestamp", "metadata"),
        "supported_types": ("image", "video", "text", "audio", "action", "other"),
    }

    with pytest.raises(NotImplementedError, match="BaseMemory.record"):
        memory.record({})
    with pytest.raises(NotImplementedError, match="BaseMemory.select"):
        memory.select({})
    with pytest.raises(NotImplementedError, match="BaseMemory.compress"):
        memory.compress([])
    with pytest.raises(NotImplementedError, match="BaseMemory.process"):
        memory.process([])
    with pytest.raises(NotImplementedError, match="BaseMemory.manage"):
        memory.manage()
