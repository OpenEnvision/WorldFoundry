"""CPU-only regression tests for core-foundation point-defect fixes.

Covers review findings fixed in the second batch:

* CF-12  TypedRegistry register race + did-you-mean errors
* CF-46  safety package imports stay light (no torch/imageio)
* CF-7   flags honor WORLDFOUNDRY_/COSMOS_ prefixes; dead constants removed
* CF-8/9 cosmos_config lazy megatron fallback; freeze covers EMAConfig,
         decorator idempotent
* CF-14  logging_setup configure race + _parse_bytes("mb")
* CF-18  model_loading.model works without transformers; vram_limit derived
* CF-27  sharded zstd shard decompression uses attached -T form
* CF-34  merge_video_audio failures propagate and clean the temp file
* CF-3/5/6 lazy config and Hydra avoid process-wide mutable state
* CF-20  LightX2V LoRA merging is in-place and keeps parent contracts
* CF-37/38 pickle requires opt-in; easy_io.exists propagates backend failures
* CF-43/44 async thread maps own pools; seed/CUDA graph compatibility converges
* CF-47  structures export table stays consistent with submodule __all__
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import types
from importlib import import_module
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_by_path(module_name: str, relative_path: str, stubs: dict[str, types.ModuleType] | None = None):
    """Load a module from source without triggering its package __init__."""
    for stub_name, stub in (stubs or {}).items():
        sys.modules[stub_name] = stub
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTypedRegistry:
    def test_concurrent_duplicate_registration_single_winner(self):
        from worldfoundry.core.registry import DuplicateRegistryKeyError, TypedRegistry

        registry = TypedRegistry()
        outcomes: list[str] = []

        def worker() -> None:
            try:
                registry.register("key", object())
                outcomes.append("ok")
            except DuplicateRegistryKeyError:
                outcomes.append("dup")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert outcomes.count("ok") == 1

    def test_unknown_key_suggests_close_match(self):
        from worldfoundry.core.registry import TypedRegistry, UnknownRegistryKeyError

        registry = TypedRegistry()
        registry.register("CosmosPredict", 1, aliases=("cosmos",))
        with pytest.raises(UnknownRegistryKeyError, match="did you mean"):
            registry.get("cosmos_predict")
        assert registry.get("cosmos") == 1


class TestSafetyImportsStayLight:
    def test_guardrail_import_avoids_heavy_modules(self):
        code = (
            "import sys\n"
            "from worldfoundry.core.safety import GuardrailRunner\n"
            "assert 'torch' not in sys.modules, 'torch imported'\n"
            "assert 'imageio' not in sys.modules, 'imageio imported'\n"
            "runner = GuardrailRunner()\n"
            "ok, message = runner.run_safety_check('hello')\n"
            "assert ok is True\n"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )


class TestFlags:
    def test_prefix_fallback_and_dead_constants_removed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WORLDFOUNDRY_VERBOSE", "1")
        monkeypatch.setenv("COSMOS_INTERNAL", "true")
        monkeypatch.delenv("WORLDFOUNDRY_INTERNAL", raising=False)
        monkeypatch.delenv("COSMOS_VALIDATION", raising=False)
        monkeypatch.delenv("WORLDFOUNDRY_VALIDATION", raising=False)
        flags = _load_by_path("wf_test_flags", "worldfoundry/core/configuration/flags.py")
        assert flags.VERBOSE is True
        assert flags.INTERNAL is True
        assert flags.VALIDATION is False
        assert not hasattr(flags, "TRAINING")
        assert not hasattr(flags, "SMOKE")

    def test_worldfoundry_prefix_wins_over_cosmos(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WORLDFOUNDRY_VERBOSE", "0")
        monkeypatch.setenv("COSMOS_VERBOSE", "1")
        flags = _load_by_path("wf_test_flags_priority", "worldfoundry/core/configuration/flags.py")
        assert flags.VERBOSE is False


class TestConfigurationIsolation:
    def test_lazy_config_does_not_patch_process_globals(self, tmp_path: Path):
        import builtins

        from omegaconf import OmegaConf

        original_import = builtins.__import__
        original_to_object = OmegaConf.to_object
        from worldfoundry.core.configuration.lazy_config import LazyConfig

        shared = tmp_path / "shared.py"
        root = tmp_path / "root.py"
        shared.write_text('shared = {"value": 7}\n', encoding="utf-8")
        root.write_text(
            'from .shared import shared\nconfig = {"nested": shared}\n',
            encoding="utf-8",
        )
        loaded = LazyConfig.load(str(root), keys="config")
        assert loaded.nested.value == 7
        assert builtins.__import__ is original_import
        assert OmegaConf.to_object is original_to_object

    def test_hydra_override_removes_temporary_config_store_node(self):
        from hydra.core.config_store import ConfigStore

        from worldfoundry.core.configuration.cosmos_config import Config
        from worldfoundry.core.configuration.hydra import override

        before = set(ConfigStore.instance().repo)
        result = override(Config(model=None), ["--", "job.name=isolation-test"])
        assert result.job.name == "isolation-test"
        assert set(ConfigStore.instance().repo) == before

    def test_model_config_aliases_are_unambiguous(self):
        from worldfoundry.core import DiffusionModelConfig, LoaderModelConfig, ModelConfig
        from worldfoundry.core.configuration import ModelConfig as LegacyDiffusionModelConfig
        from worldfoundry.core.model_loading import ModelConfig as LegacyLoaderModelConfig

        assert DiffusionModelConfig is LegacyDiffusionModelConfig
        assert LoaderModelConfig is LegacyLoaderModelConfig
        assert ModelConfig is LoaderModelConfig


@pytest.fixture()
def cosmos_config_module():
    stub = types.ModuleType("worldfoundry.core.configuration.lazy_config")

    class LazyDict(dict):
        pass

    stub.LazyDict = LazyDict
    previous = sys.modules.get("worldfoundry.core.configuration.lazy_config")
    module = _load_by_path(
        "wf_test_cosmos_config",
        "worldfoundry/core/configuration/cosmos_config.py",
        stubs={"worldfoundry.core.configuration.lazy_config": stub},
    )
    yield module
    if previous is not None:
        sys.modules["worldfoundry.core.configuration.lazy_config"] = previous
    else:
        sys.modules.pop("worldfoundry.core.configuration.lazy_config", None)


class TestCosmosConfig:
    def test_model_parallel_falls_back_without_megatron(self, cosmos_config_module):
        config = cosmos_config_module.Config(model=None)
        assert isinstance(config.model_parallel, cosmos_config_module._FallbackModelParallelConfig)
        assert config.model_parallel.context_parallel_size == 1

    def test_freeze_recurses_and_covers_ema(self, cosmos_config_module):
        config = cosmos_config_module.Config(model=None)
        config.freeze()
        with pytest.raises(AttributeError):
            config.job.project = "x"
        ema = cosmos_config_module.EMAConfig()
        ema.freeze()
        with pytest.raises(AttributeError):
            ema.rate = 0.5

    def test_make_freezable_idempotent(self, cosmos_config_module):
        cls = cosmos_config_module.JobConfig
        before = cls.__setattr__
        cosmos_config_module.make_freezable(cls)
        assert cls.__setattr__ is before


class TestLoggingSetup:
    def test_parse_bytes_bare_unit_returns_default(self):
        from worldfoundry.core.observability.logging_setup import _parse_bytes

        assert _parse_bytes("mb", 999) == 999
        assert _parse_bytes("10 mb", 0) == 10 * 1024**2
        assert _parse_bytes("", 7) == 7
        assert _parse_bytes(2048, 0) == 2048

    def test_concurrent_configure_is_serialized(self):
        from worldfoundry.core.observability import logging_setup

        errors: list[BaseException] = []

        def worker() -> None:
            try:
                logging_setup.configure_logging(level="INFO")
            except BaseException as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert logging_setup.is_configured()


class TestModelLoadingModel:
    def test_import_and_load_without_transformers(self):
        code = (
            "import sys\n"
            "from worldfoundry.core.model_loading.model import load_model\n"
            "assert 'transformers' not in sys.modules, 'transformers imported eagerly'\n"
            "import torch\n"
            "model = load_model(\n"
            "    torch.nn.Linear, path=None,\n"
            "    config={'in_features': 4, 'out_features': 2},\n"
            "    torch_dtype=torch.float32, device='cpu',\n"
            "    state_dict={'weight': torch.ones(2, 4), 'bias': torch.zeros(2)},\n"
            ")\n"
            "assert not model.training\n"
            "assert torch.equal(model.weight, torch.ones(2, 4))\n"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )

    def test_disk_offload_vram_limit_cpu_fallback(self):
        from worldfoundry.core.model_loading.model import _disk_offload_vram_limit

        assert _disk_offload_vram_limit("cpu") == 80.0


class TestLightX2VLoRA:
    def test_parent_mapping_contract_and_in_place_merge(self, monkeypatch: pytest.MonkeyPatch):
        import torch

        from worldfoundry.core.model_loading.lora import LightX2VLoRALoader

        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        checkpoint = {
            "lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "lora_B.weight": torch.tensor([[3.0], [4.0]]),
            "diff_b": torch.tensor([0.5, -0.5]),
        }
        loader = LightX2VLoRALoader(torch_dtype=torch.float32)
        assert isinstance(loader.get_name_dict(checkpoint), dict)
        pairs, diffs = loader.get_lightx2v_mappings(checkpoint)
        assert pairs["weight"] == ("lora_A.weight", "lora_B.weight")
        assert diffs["bias"] == "diff_b"

        monkeypatch.setattr(
            model,
            "state_dict",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full state_dict copied")),
        )
        monkeypatch.setattr(
            model,
            "load_state_dict",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full state_dict loaded")),
        )
        assert loader.load(model, checkpoint) == 2
        assert torch.equal(model.weight, torch.tensor([[3.0, 6.0], [4.0, 8.0]]))
        assert torch.equal(model.bias, torch.tensor([0.5, -0.5]))


class TestShardedZstd:
    @pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd binary unavailable")
    def test_load_shard_with_threads_roundtrip(self, tmp_path: Path):
        import torch
        from safetensors.torch import save_file

        from worldfoundry.core.model_loading.checkpoints.sharded_safetensors import _load_shard

        shard = tmp_path / "model.safetensors"
        save_file({"w": torch.arange(6, dtype=torch.float32)}, shard)
        subprocess.run(["zstd", "-q", str(shard), "-o", str(shard) + ".zst"], check=True)
        shard.unlink()

        # Pre-fix, ["-T", "2"] was parsed by zstd as a file operand and failed.
        loaded = _load_shard(str(shard), ["w"], num_threads=2)
        assert torch.equal(loaded["w"], torch.arange(6, dtype=torch.float32))

    def test_missing_zstd_binary_raises_actionable_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from worldfoundry.core.model_loading.checkpoints import sharded_safetensors

        (tmp_path / "model.safetensors.zst").write_bytes(b"anything")

        def raising_popen(*args, **kwargs):
            raise FileNotFoundError("zstd")

        monkeypatch.setattr(sharded_safetensors.subprocess, "Popen", raising_popen)
        with pytest.raises(RuntimeError, match="zstd binary not found"):
            sharded_safetensors._load_shard(str(tmp_path / "model.safetensors"), ["w"], num_threads=2)


class TestMergeVideoAudio:
    def test_missing_inputs_raise(self, tmp_path: Path):
        from worldfoundry.core.io.inputs.video_data import merge_video_audio

        audio = tmp_path / "a.aac"
        audio.write_bytes(b"x")
        with pytest.raises(FileNotFoundError):
            merge_video_audio(str(tmp_path / "missing.mp4"), str(audio))

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
    def test_ffmpeg_failure_propagates_and_cleans_temp(self, tmp_path: Path):
        from worldfoundry.core.io.inputs.video_data import merge_video_audio

        video = tmp_path / "v.mp4"
        audio = tmp_path / "a.aac"
        video.write_bytes(b"not a real video")
        audio.write_bytes(b"not real audio")
        with pytest.raises(RuntimeError, match="FFmpeg execute failed"):
            merge_video_audio(str(video), str(audio))
        assert not (tmp_path / "v_temp.mp4").exists()
        assert video.exists()


class TestEasyIOExists:
    def test_missing_path_is_false_and_present_path_is_true(self, tmp_path: Path):
        from worldfoundry.core.io.assets.easy_io import easy_io

        assert easy_io.exists(str(tmp_path / "nope" / "missing.bin")) is False
        target = tmp_path / "real.bin"
        target.write_bytes(b"data")
        assert easy_io.exists(str(target)) is True

    def test_unexpected_backend_errors_propagate_unless_suppressed(self, monkeypatch: pytest.MonkeyPatch):
        from worldfoundry.core.io.assets import easy_io as easy_io_module

        def fail_resolution(path):
            raise RuntimeError(f"backend unavailable for {path}")

        monkeypatch.setattr(easy_io_module, "resolve_checkpoint_path", fail_resolution)
        with pytest.raises(RuntimeError, match="backend unavailable"):
            easy_io_module.easy_io.exists("hf://private/model.bin")
        assert easy_io_module.easy_io.exists("hf://private/model.bin", suppress_errors=True) is False


class TestSafeSerialization:
    def test_pickle_and_gzip_pickle_require_explicit_opt_in(self, tmp_path: Path):
        from worldfoundry.core.io.formats.serialization import dump_serialized, load_serialized

        for suffix in ("pkl", "gz"):
            path = tmp_path / f"payload.{suffix}"
            dump_serialized({"safe-source": True}, path)
            with pytest.raises(ValueError, match="allow_pickle=True"):
                load_serialized(path)
            assert load_serialized(path, allow_pickle=True) == {"safe-source": True}

    def test_legacy_load_pickle_also_requires_opt_in(self, tmp_path: Path):
        from worldfoundry.core.io.filesystem.file_utils import dump_pickle, load_pickle

        path = tmp_path / "legacy.pkl"
        dump_pickle({"trusted": 1}, path)
        with pytest.raises(ValueError, match="allow_pickle=True"):
            load_pickle(path)
        assert load_pickle(path, allow_pickle=True) == {"trusted": 1}

    def test_generic_video_default_is_standard_24_fps(self):
        import inspect

        from worldfoundry.core.io.formats.serialization import _dump_video

        assert inspect.signature(_dump_video).parameters["fps"].default == 24


class TestParallelExecution:
    def test_results_and_error_propagation(self):
        from worldfoundry.core.execution.parallel_execution import parallel_execution

        assert parallel_execution([1, 2, 3], action=lambda x: x * 2, num_processes=2) == [2, 4, 6]

        def boom(value):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            parallel_execution([1], action=boom, num_processes=2)

    def test_async_return_hands_over_live_pool(self):
        from worldfoundry.core.execution.parallel_execution import parallel_execution

        pending = parallel_execution(
            [1, 2],
            action=lambda x: x * 3,
            num_workers=2,
            async_return=True,
        )
        assert pending.get() == [3, 6]


class TestSeedAndCudaGraphCompatibility:
    def test_legacy_seed_helpers_forward_to_canonical(self, monkeypatch: pytest.MonkeyPatch):
        from worldfoundry.core.utils.tensors import torch as torch_utils

        seeds: list[int] = []
        monkeypatch.setattr(torch_utils, "set_seed_everywhere", lambda seed: seeds.append(seed) or seed)
        assert torch_utils.set_random_seed(17) == 17
        assert torch_utils.fix_random_seeds(23) == 23
        assert seeds == [17, 23]

    def test_cuda_graph_prefers_public_pool_api_and_guards_pytree(self):
        source = (REPO_ROOT / "worldfoundry/core/execution/graphs/cuda_graph.py").read_text(encoding="utf-8")
        assert "from torch._C import _graph_pool_handle" not in source
        assert 'getattr(torch.cuda, "graph_pool_handle"' in source
        assert "if _torch_pytree is None" in source


class TestUtilsExports:
    def test_export_table_matches_submodule_all(self):
        import worldfoundry.core.utils as utils

        for name, module_name in utils._EXPORT_MODULES.items():
            module = import_module(module_name)
            assert hasattr(module, name), f"{module_name} lost export {name}"
            declared = getattr(module, "__all__", None)
            if declared is not None:
                assert name in declared, f"{name} not in {module_name}.__all__"

    def test_validator_all_fully_exported(self):
        import worldfoundry.core.utils as utils
        from worldfoundry.core.configuration import validators as validator

        exported = {name for name, module in utils._EXPORT_MODULES.items() if module == validator.__name__}
        # Private names (e.g. the _UNSET sentinel) intentionally stay module-local.
        public = {name for name in validator.__all__ if not name.startswith("_")}
        assert public == exported
