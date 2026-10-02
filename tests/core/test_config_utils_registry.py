from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest

pytest.importorskip("hydra")
pytest.importorskip("tree")
pytest.importorskip("omegaconf")

from worldfoundry.core.configuration import hydra_utils as config_utils


def _unique_name(label: str) -> str:
    return f"RegistryTest_{label}_{uuid4().hex}"


def test_register_class_conflict_does_not_leave_partial_registration():
    conflict_name = _unique_name("conflict")
    free_alias = _unique_name("free_alias")
    candidate_name = _unique_name("candidate")
    occupant = type(_unique_name("occupant"), (), {})
    candidate = type(candidate_name, (), {})
    config_utils.register_callable(conflict_name, occupant)

    with pytest.raises(ValueError, match="already bound"):
        config_utils.register_class(alias=[free_alias, conflict_name])(candidate)

    assert candidate_name not in config_utils._CLASS_REGISTRY
    assert free_alias not in config_utils._CLASS_REGISTRY
    assert config_utils._CLASS_REGISTRY[conflict_name] is occupant


def test_concurrent_same_name_registration_has_one_winner():
    shared_name = _unique_name("shared")
    aliases = [_unique_name("first_alias"), _unique_name("second_alias")]
    candidates = [type(shared_name, (), {}), type(shared_name, (), {})]
    barrier = Barrier(len(candidates))

    def register(index):
        barrier.wait()
        try:
            config_utils.register_class(alias=[aliases[index]])(candidates[index])
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        results = list(executor.map(register, range(len(candidates))))

    assert results.count(True) == 1
    winner = results.index(True)
    loser = 1 - winner
    assert config_utils._CLASS_REGISTRY[shared_name] is candidates[winner]
    assert config_utils._CLASS_REGISTRY[aliases[winner]] is candidates[winner]
    assert aliases[loser] not in config_utils._CLASS_REGISTRY


def test_registering_same_class_and_aliases_is_idempotent():
    class_type = type(_unique_name("idempotent"), (), {})
    aliases = [_unique_name("alias_one"), _unique_name("alias_two")]

    config_utils.register_class(alias=aliases)(class_type)
    config_utils.register_class(alias=aliases)(class_type)

    assert config_utils._CLASS_REGISTRY[class_type.__name__] is class_type
    assert all(config_utils._CLASS_REGISTRY[alias] is class_type for alias in aliases)
