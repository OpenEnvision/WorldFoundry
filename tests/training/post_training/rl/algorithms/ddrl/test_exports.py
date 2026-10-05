from __future__ import annotations

from worldfoundry.training import post_training
from worldfoundry.training.post_training import rl
from worldfoundry.training.post_training.rl import algorithms
from worldfoundry.training.post_training.rl.algorithms import ddrl


def test_ddrl_lazy_facades_resolve_to_canonical_runtime() -> None:
    for name in ddrl.__all__:
        canonical = getattr(ddrl, name)
        assert getattr(algorithms, name) is canonical
        assert getattr(rl, name) is canonical
        assert getattr(post_training, name) is canonical
