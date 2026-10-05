"""XC-23: training sessions seed Python, NumPy, and torch via set_seed_everywhere.

Hand-rolled random.seed + torch.manual_seed blocks left NumPy's global RNG
unseeded; these tests pin the sessions to the canonical helper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.api import ObjectiveBatch, PreparedBatch, TrainingBatch  # noqa: E402
from worldfoundry.training.engine import (  # noqa: E402
    SingleDeviceTrainEngine,
    SingleDeviceTrainingSession,
    build_adamw,
)
from worldfoundry.training.engine.sessions import single_device as single_device_module  # noqa: E402
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)
from worldfoundry.training.recipes import TrainingRecipe  # noqa: E402


class _TinyAdapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes: tuple[type, ...] = ()

    def __init__(self) -> None:
        self.trainable_module = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        assert isinstance(batch.pixel_values, torch.Tensor)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=batch.pixel_values[:, :, 0],
        )

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        assert isinstance(batch.model_input, torch.Tensor)
        return self.trainable_module(batch.model_input)


def _recipe() -> TrainingRecipe:
    return TrainingRecipe.from_mapping(
        {
            "execution_owner": "worldfoundry-native",
            "run": {"id": "seed-everywhere", "output_dir": "unused"},
            "model": {"recipe": "sana-tiny", "checkpoint": "default"},
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "manifest.jsonl",
                "cache": "cache",
                "shuffle": False,
                "tail_policy": "uneven",
            },
            "objective": {
                "type": "flow_matching",
                "prediction_type": "flow_velocity",
                "timestep_sampler": "uniform",
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.05,
                "gradient_accumulation_steps": 1,
            },
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "checkpoint": {"save_every_steps": 0, "async": False},
        }
    )


def _batches() -> list[TrainingBatch]:
    return [
        TrainingBatch(
            sample_ids=(f"sample-{index}",),
            prompts=("cached",),
            pixel_values=torch.tensor([[[[[0.0, 0.5], [1.0, -0.5]]]]]) + index * 0.1,
        )
        for index in range(2)
    ]


def _session(output_dir: Path) -> SingleDeviceTrainingSession:
    adapter = _TinyAdapter()
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    optimizer = build_adamw(
        adapter.trainable_module.parameters(),
        learning_rate=0.05,
        fused=False,
    )
    engine = SingleDeviceTrainEngine(adapter, objective, optimizer)
    return SingleDeviceTrainingSession(
        recipe=_recipe(),
        engine=engine,
        dataloader=_batches(),
        output_dir=output_dir,
        data_identity={"samples": ["sample-0", "sample-1"]},
    )


def _record_set_seed_everywhere(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    recorded: list[int] = []
    real = single_device_module.set_seed_everywhere

    def _recording(seed: int) -> int | None:
        recorded.append(int(seed))
        return real(seed)

    monkeypatch.setattr(single_device_module, "set_seed_everywhere", _recording)
    return recorded


def test_fresh_run_seeds_all_backends_once_with_the_rank_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _record_set_seed_everywhere(monkeypatch)

    session = _session(tmp_path / "run")
    session.run(max_steps=1, seed=13)

    assert recorded == [13]


def test_fresh_run_seeds_the_numpy_global_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded: list[int] = []
    original = np.random.seed

    def _recording(value: int) -> None:
        seeded.append(int(value))
        original(value)

    monkeypatch.setattr(np.random, "seed", _recording)

    session = _session(tmp_path / "numpy-run")
    session.run(max_steps=1, seed=13)

    assert seeded == [13]


def test_fixed_corruption_reseeds_every_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _record_set_seed_everywhere(monkeypatch)

    session = _session(tmp_path / "fixed")
    session.run(max_steps=2, seed=17, fixed_batch=True, fixed_corruption=True)

    assert recorded == [17, 17, 17]


def test_rank_offset_is_preserved_in_the_derived_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _record_set_seed_everywhere(monkeypatch)

    session = _session(tmp_path / "ranked")
    session.rank = 3
    session.run(max_steps=1, seed=11)

    assert recorded == [14]


def _scm_ladd_module():
    # scm_ladd imports transformers at module scope; mirror the guard used by
    # tests/training/test_sana_scm_ladd_runtime.py.
    pytest.importorskip("transformers")
    from worldfoundry.training.engine.sana import scm_ladd

    return scm_ladd


def test_scm_ladd_seed_routes_the_resolved_seed_through_set_seed_everywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scm_ladd = _scm_ladd_module()
    recorded: list[int] = []
    monkeypatch.setattr(scm_ladd, "set_seed_everywhere", lambda seed: recorded.append(int(seed)))

    scm_ladd._seed(2**63 + 5)

    assert recorded == [(2**63 + 5) % (2**63 - 1)]
    assert recorded == [6]


def test_scm_ladd_seed_none_skips_seeding(monkeypatch: pytest.MonkeyPatch) -> None:
    scm_ladd = _scm_ladd_module()
    recorded: list[int] = []
    monkeypatch.setattr(scm_ladd, "set_seed_everywhere", lambda seed: recorded.append(int(seed)))

    scm_ladd._seed(None)

    assert recorded == []


def test_scm_ladd_seed_still_rejects_bool() -> None:
    scm_ladd = _scm_ladd_module()
    with pytest.raises(TypeError, match="initialization_seed"):
        scm_ladd._seed(True)


def test_scm_ladd_seed_seeds_the_numpy_global_rng() -> None:
    scm_ladd = _scm_ladd_module()

    scm_ladd._seed(123)
    observed = np.random.randint(0, 2**31)

    np.random.seed(123)
    assert observed == np.random.randint(0, 2**31)
