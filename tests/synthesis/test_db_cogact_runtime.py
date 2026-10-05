from types import SimpleNamespace

import numpy as np

from worldfoundry.synthesis.action_generation.db_cogact.runtime import (
    _finalize_vision_tower,
    _policy_observation,
    _prepare_policy_observation,
)


def _prepare_for_policy(policy, observation):
    payload = _policy_observation(observation, None, "test instruction", {})
    return _prepare_policy_observation(policy, payload)


def test_unused_state_is_not_sent_to_official_cogact_policy() -> None:
    payload = _prepare_for_policy(
        SimpleNamespace(state_used=False),
        {"image/1": "front.png", "state": [0.0] * 8},
    )

    assert "state" not in payload


def test_single_state_vector_is_not_mistaken_for_a_batch() -> None:
    payload = _prepare_for_policy(
        SimpleNamespace(state_used=True),
        {"image/1": "front.png", "state": [0.0] * 8},
    )

    assert isinstance(payload["state"], np.ndarray)
    assert payload["state"].shape == (8,)


def test_transformers_meta_vision_tower_is_finalized() -> None:
    class VisionTower:
        _meta_initialized = True
        is_loaded = False

        def load_model(self):
            self.is_loaded = True

    tower = VisionTower()
    model = SimpleNamespace(model=SimpleNamespace(mm_vision_tower=tower))

    _finalize_vision_tower(model)

    assert tower.is_loaded
