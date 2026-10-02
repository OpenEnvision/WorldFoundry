from __future__ import annotations

from worldfoundry.runtime.conda import RuntimeCondaEnvSpec
from worldfoundry.studio.inference.dispatch import _runtime_env
from worldfoundry.base_models.three_dimensions.three_d_four_d.runtime import (
    ThreeDFourDRuntimeSynthesis,
    three_d_four_d_runtime_spec,
)


OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def test_conda_child_preserves_explicit_offline_mode(tmp_path, monkeypatch) -> None:
    for name, value in OFFLINE_ENV.items():
        monkeypatch.setenv(name, value)
    spec = RuntimeCondaEnvSpec(model_id="fixture", env_name="fixture", env_root=tmp_path)

    env = _runtime_env(spec, "cuda:0")

    assert {name: env.get(name) for name in OFFLINE_ENV} == OFFLINE_ENV


def test_three_d_four_d_child_preserves_explicit_offline_mode(tmp_path, monkeypatch) -> None:
    for name, value in OFFLINE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))
    runtime = ThreeDFourDRuntimeSynthesis(
        spec=three_d_four_d_runtime_spec("mvdiffusion"),
        source_root=tmp_path,
        device="cpu",
    )

    env = runtime._subprocess_env()

    assert {name: env.get(name) for name in OFFLINE_ENV} == OFFLINE_ENV
