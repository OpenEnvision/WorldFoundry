from __future__ import annotations

import json
from importlib.metadata import version as package_version
from pathlib import Path

import pytest
import torch
from packaging.version import Version
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.optimizations import (
    AttentionBackend,
    OffloadMode,
    OffloadPolicy,
    RuntimePolicy,
)
from worldfoundry.core.attention.chunk_partition import TemporalChunkPartition
from worldfoundry.training.models import anyflow as model_boundary
from worldfoundry.training.models.anyflow import (
    ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT,
    ANYFLOW_FAR_WAN_SMALL_CHECKPOINT,
    NativeAnyFlowModelMaterializer,
)
from worldfoundry.training.post_training.shared.role_checkpoints import (
    ResolvedRoleCheckpoint,
)


def _partition() -> TemporalChunkPartition:
    return TemporalChunkPartition(
        chunks=(1, 1),
        full_chunk_limit=1,
        patch_size=(1, 1, 1),
        compressed_patch_size=(1, 2, 2),
    )


def _config(*, far: bool) -> dict[str, object]:
    values: dict[str, object] = {
        "_class_name": (
            "AnyFlowFARTransformer3DModel" if far else "AnyFlowTransformer3DModel"
        ),
        "_diffusers_version": "0.35.1",
        "patch_size": [1, 1, 1],
        "num_attention_heads": 1,
        "attention_head_dim": 12,
        "in_channels": 2,
        "out_channels": 2,
        "text_dim": 8,
        "freq_dim": 8,
        "ffn_dim": 24,
        "num_layers": 1,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "image_dim": None,
        "rope_max_seq_len": 16,
        "gate_value": 0.25,
        "deltatime_type": "r",
    }
    if far:
        values.update(
            {
                "compressed_patch_size": [1, 2, 2],
                "full_chunk_limit": 1,
                "chunk_partition": [1, 1],
            }
        )
    return values


def _checkpoint(root: Path, *, far: bool) -> CheckpointSpec:
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    config = _config(far=far)
    (transformer / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    model_class = model_boundary._native_model_class()
    constructor = {name: value for name, value in config.items() if not name.startswith("_")}
    module = model_class(**constructor)
    save_file(
        {
            name: tensor.detach().contiguous()
            for name, tensor in module.state_dict().items()
        },
        transformer / "diffusion_pytorch_model.safetensors",
    )
    return CheckpointSpec(
        source=root,
        files=("transformer/diffusion_pytorch_model.safetensors",),
    )


def _policy() -> RuntimePolicy:
    return RuntimePolicy(
        device="cpu",
        dtype=torch.float32,
        attention=AttentionBackend.TORCH,
    )


def _cuda_policy() -> RuntimePolicy:
    return RuntimePolicy(
        device="cuda",
        dtype=torch.float32,
        attention=AttentionBackend.TORCH,
    )


def test_released_small_checkpoint_defaults_are_immutable_and_component_only() -> None:
    assert ANYFLOW_FAR_WAN_SMALL_CHECKPOINT.revision == (
        "915af337434035df8545797ecc910d79fa78cf29"
    )
    assert ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT.revision == (
        "4c2ec05c7fa4dbafbca131ad32430905c7ff2974"
    )
    for checkpoint in (
        ANYFLOW_FAR_WAN_SMALL_CHECKPOINT,
        ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT,
    ):
        assert checkpoint.files == (
            "transformer/diffusion_pytorch_model.safetensors",
        )
        assert all(pattern.startswith("transformer/") for pattern in checkpoint.allow_patterns)


def test_model_boundary_import_is_lazy_and_dependency_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = Version(package_version("diffusers"))
    assert Version("0.35.1") <= installed < Version("0.40")

    def unavailable(name: str) -> object:
        raise ImportError("missing", name="diffusers")

    monkeypatch.setattr(model_boundary, "import_module", unavailable)
    with pytest.raises(RuntimeError, match=r"diffusers>=0\.35\.1,<0\.40"):
        model_boundary._native_model_class()


def test_copied_graph_keeps_exact_apache_source_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "worldfoundry/base_models/diffusion_model/models/networks/wan/variants/anyflow.py"
    ).read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: Apache-2.0" in source
    assert "d2acf7373a45173082ec47eb16553a373b10f856" in source
    assert "from far." not in source


def test_official_single_time_clean_context_bug_is_fixed_and_differentiable() -> None:
    native = model_boundary.import_module(
        "worldfoundry.base_models.diffusion_model.models.networks.wan.variants.anyflow"
    )
    embedding = native.WanTimeTextImageEmbedding(
        dim=12,
        time_freq_dim=8,
        time_proj_dim=72,
        text_embed_dim=8,
    )
    context = torch.randn(1, 2, 8)
    temporal, projected, encoded, image = embedding(
        torch.tensor([[900.0, 600.0]]),
        torch.zeros(1, 2),
        context,
        far_cfg={
            "num_full_frames": 1,
            "full_token_per_frame": 2,
            "compressed_token_per_frame": 1,
        },
        clean_timestep=torch.zeros(1, 1),
        is_causal=True,
    )
    assert temporal.shape == (1, 5, 12)
    assert projected.shape == (1, 5, 72)
    assert encoded.shape == (1, 2, 12)
    assert image is None
    (temporal.square().mean() + projected.square().mean()).backward()
    assert embedding.time_embedder.linear_1.weight.grad is not None


def test_copied_graph_dispatch_consumes_the_explicit_causal_flag() -> None:
    native = model_boundary.import_module(
        "worldfoundry.base_models.diffusion_model.models.networks.wan.variants.anyflow"
    )

    class DispatchProbe:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def _forward_train(self, *args: object, **kwargs: object) -> str:
            self.calls.append(("train", dict(kwargs)))
            return "train"

        def _forward_bidirection(self, *args: object, **kwargs: object) -> str:
            self.calls.append(("bidirectional", dict(kwargs)))
            return "bidirectional"

    probe = DispatchProbe()
    forward = native.AnyFlowWanTransformer3DModel.forward

    assert forward(probe, object(), is_causal=True, timestep="t") == "train"
    assert probe.calls[-1] == ("train", {"timestep": "t"})
    assert forward(probe, object(), is_causal=False, timestep="t") == (
        "bidirectional"
    )
    assert probe.calls[-1] == (
        "bidirectional",
        {"is_causal": False, "timestep": "t"},
    )


def test_materializer_loads_real_far_graph_and_audits_partition(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "far", far=True)
    adapter = NativeAnyFlowModelMaterializer().far_student(
        checkpoint,
        checkpoint_identity="far-local",
        partition=_partition(),
        policy=_policy(),
        gradient_checkpointing=True,
    )

    assert adapter.checkpoint_identity == "far-local"
    assert adapter.module.training
    assert adapter.module.gradient_checkpointing
    assert all(parameter.requires_grad for parameter in adapter.module.parameters())
    assert tuple(adapter.module.config.chunk_partition) == (1, 1)
    assert isinstance(adapter.module.far_patch_embedding, torch.nn.Module)
    state = adapter.create_rollout_state(
        partition=_partition(),
        reference=torch.zeros(1, 2, 2, 2, 2),
    )
    assert state.flags == {"num_cached_chunks": 0, "is_cache_step": False}
    assert set(state.model_cache) == {0}


def test_materializer_loads_independent_bidirectional_student_and_score_roles(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "bidirectional", far=False)
    materializer = NativeAnyFlowModelMaterializer()
    student = materializer.bidirectional_student(
        checkpoint,
        checkpoint_identity="student-local",
        policy=_policy(),
    )
    real_score = materializer.real_score(
        checkpoint,
        checkpoint_identity="real-local",
        policy=_policy(),
    )
    fake_score = materializer.fake_score(
        checkpoint,
        checkpoint_identity="fake-local",
        policy=_policy(),
    )

    assert len({id(student.module), id(real_score.module), id(fake_score.module)}) == 3
    assert student.module.training and fake_score.module.training
    assert not real_score.module.training
    assert all(parameter.requires_grad for parameter in student.module.parameters())
    assert all(parameter.requires_grad for parameter in fake_score.module.parameters())
    assert not any(parameter.requires_grad for parameter in real_score.module.parameters())

    latents = torch.randn(1, 2, 2, 2, 2)
    source = torch.full((1, 2), 800.0)
    destination = torch.full((1, 2), 200.0)
    prediction = student.predict_flow_map(
        latents,
        source,
        destination,
        sample_ids=("sample",),
        conditioning={"context": torch.randn(1, 2, 8)},
        training=True,
    )
    assert prediction.shape == latents.shape
    prediction.square().mean().backward()
    assert any(parameter.grad is not None for parameter in student.module.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bidirectional_materializer_executes_a_real_cuda_optimizer_update(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "cuda-bidirectional", far=False)
    student = NativeAnyFlowModelMaterializer().bidirectional_student(
        checkpoint,
        checkpoint_identity="cuda-student",
        policy=_cuda_policy(),
    )
    optimizer = torch.optim.AdamW(student.module.parameters(), lr=1.0e-4)
    latents = torch.randn(1, 2, 2, 2, 2, device="cuda")
    prediction = student.predict_flow_map(
        latents,
        torch.full((1, 2), 800.0, device="cuda"),
        torch.full((1, 2), 200.0, device="cuda"),
        sample_ids=("cuda-sample",),
        conditioning={"context": torch.randn(1, 2, 8, device="cuda")},
        training=True,
    )
    loss = prediction.float().square().mean()
    assert bool(torch.isfinite(loss))
    before = next(student.module.parameters()).detach().clone()
    loss.backward()
    optimizer.step()
    after = next(student.module.parameters()).detach()
    assert not torch.equal(before, after)


def test_materializer_binds_resolved_checkpoint_identity(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "resolved", far=False)
    resolved = ResolvedRoleCheckpoint(
        role="student",
        requested_reference="default",
        checkpoint=checkpoint,
        source_kind="local",
    )
    adapter = NativeAnyFlowModelMaterializer().bidirectional_student(
        resolved,
        policy=_policy(),
    )
    assert adapter.checkpoint_identity == "default"

    with pytest.raises(ValueError, match="differs from the resolved role"):
        NativeAnyFlowModelMaterializer().bidirectional_student(
            resolved,
            checkpoint_identity="other",
            policy=_policy(),
        )


def test_materializer_rejects_checkpoint_semantic_drift_before_loading(
    tmp_path: Path,
) -> None:
    root = tmp_path / "drift"
    checkpoint = _checkpoint(root, far=True)
    config_path = root / "transformer/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["gate_value"] = 0.5
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="gate_value=0.25"):
        NativeAnyFlowModelMaterializer().far_student(
            checkpoint,
            checkpoint_identity="drifted",
            partition=_partition(),
            policy=_policy(),
        )


def test_materializer_rejects_inference_only_runtime_policies(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "offload", far=False)
    policy = RuntimePolicy(
        offload=OffloadPolicy(mode=OffloadMode.BLOCK),
    )
    with pytest.raises(ValueError, match="inference-time weight offload"):
        NativeAnyFlowModelMaterializer().bidirectional_student(
            checkpoint,
            checkpoint_identity="student-local",
            policy=policy,
        )
