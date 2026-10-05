from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.model_zoo.foundation_inventory import (
    ast_duplicate_groups,
    build_foundation_layer_inventory,
    file_duplicate_groups,
    iter_foundation_layer_files,
)
from worldfoundry.core import (
    DuplicateRegistryKeyError,
    PatchGridSpec,
    TypedRegistry,
    UnknownRegistryKeyError,
    attention_backend_from_env,
    attention_backend_report,
    normalize_attention_backend,
    resolve_attention_backend,
)
from worldfoundry.core.io import (
    MediaKind,
    artifact_root_path,
    cache_root_path,
    checkpoint_root_path,
    conda_envs_root_path,
    conda_root_path,
    guess_mime_type,
    hfd_root_path,
    infer_media_kind,
    is_media_path,
    local_data_root_path,
    local_model_root_path,
    model_source_root_path,
    official_runtime_repo_path,
    package_root,
    project_root,
    repo_relative_path,
    resolve_data_path,
    resolve_package_path,
    resolve_worldfoundry_path,
    suffix_for_uri,
    worldfoundry_path_tokens,
)
from worldfoundry.core.nn import (
    TransformerShapeSpec,
    apply_rotary_embedding,
    attention_backend_info,
    attention_head_dim,
    causal_attention_mask,
    layer_scale,
    merge_attention_heads,
    mlp_hidden_size,
    patchify_image,
    rms_norm,
    rotary_frequencies,
    scaled_dot_product_attention,
    split_attention_heads,
    transformer_shape_spec,
    unpatchify_image,
)


def test_typed_registry_resolves_keys_and_aliases_case_insensitively() -> None:
    registry: TypedRegistry[int] = TypedRegistry()
    registry.register("Matrix-Game-1", 7, aliases=("matrix-game", "MG1"), metadata={"domain": "world"})

    assert registry.get("matrix-game-1") == 7
    assert registry.get("MATRIX-GAME") == 7
    assert registry.get_item("mg1").metadata == {"domain": "world"}
    assert registry.keys() == ("Matrix-Game-1",)
    assert "mg1" in registry


def test_typed_registry_rejects_duplicate_keys_and_aliases() -> None:
    registry: TypedRegistry[str] = TypedRegistry()
    registry.register("alpha", "a", aliases=("shared",))

    with pytest.raises(DuplicateRegistryKeyError):
        registry.register("ALPHA", "duplicate")
    with pytest.raises(DuplicateRegistryKeyError):
        registry.register("beta", "b", aliases=("shared",))
    with pytest.raises(DuplicateRegistryKeyError):
        registry.register("shared", "conflicts-with-alias")


def test_typed_registry_reports_unknown_keys() -> None:
    registry: TypedRegistry[str] = TypedRegistry()

    with pytest.raises(UnknownRegistryKeyError):
        registry.get("missing")


def test_core_path_helpers_resolve_package_repo_and_data_roots() -> None:
    repo = project_root()
    package = package_root()

    assert (repo / "pyproject.toml").is_file()
    assert package.name == "worldfoundry"
    assert resolve_package_path("__init__.py").is_file()
    assert resolve_data_path("models").is_dir()
    assert repo_relative_path(resolve_data_path("models"), root=repo) == "worldfoundry/data/models"


def test_core_path_tokens_are_generic_and_env_overridable(tmp_path: Path) -> None:
    tokens = worldfoundry_path_tokens({"WORLDFOUNDRY_HOME": str(tmp_path / "home")})

    assert tokens["WORLDFOUNDRY_HOME"] == str(tmp_path / "home")
    assert tokens["WORLDFOUNDRY_DATA_ROOT"].endswith("worldfoundry/data")
    assert tokens["WORLDFOUNDRY_CACHE_DIR"] == str(tmp_path / "home" / "cache")
    assert tokens["WORLDFOUNDRY_DATA_DIR"] == str(tmp_path / "home" / "data")
    assert tokens["WORLDFOUNDRY_ARTIFACT_DIR"] == str(tmp_path / "home" / "artifacts")
    assert tokens["WORLDFOUNDRY_MODEL_DIR"] == str(tmp_path / "home" / "models")
    assert tokens["WORLDFOUNDRY_MODEL_SOURCE_DIR"] == str(tmp_path / "home" / "cache" / "official_runtime_repos")
    assert tokens["WORLDFOUNDRY_CKPT_DIR"] == str(tmp_path / "home" / "checkpoints")
    assert tokens["WORLDFOUNDRY_HFD_ROOT"] == str(tmp_path / "home" / "checkpoints" / "hfd")
    assert tokens["WORLDFOUNDRY_CONDA_ENVS_ROOT"] == str(tmp_path / "home" / "conda_envs")


def test_core_model_runtime_path_helpers_are_env_driven(tmp_path: Path) -> None:
    env = {
        "WORLDFOUNDRY_MODEL_SOURCE_DIR": str(tmp_path / "repos"),
        "WORLDFOUNDRY_CACHE_DIR": str(tmp_path / "cache"),
        "WORLDFOUNDRY_DATA_DIR": str(tmp_path / "data"),
        "WORLDFOUNDRY_ARTIFACT_DIR": str(tmp_path / "artifacts"),
        "WORLDFOUNDRY_MODEL_DIR": str(tmp_path / "models"),
        "WORLDFOUNDRY_CKPT_DIR": str(tmp_path / "ckpt"),
        "WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd"),
        "WORLDFOUNDRY_CONDA_ROOT": str(tmp_path / "conda"),
        "WORLDFOUNDRY_CONDA_ENVS_ROOT": str(tmp_path / "envs"),
    }

    assert cache_root_path(env) == tmp_path / "cache"
    assert local_data_root_path(env) == tmp_path / "data"
    assert artifact_root_path(env) == tmp_path / "artifacts"
    assert local_model_root_path(env) == tmp_path / "models"
    assert model_source_root_path(env) == tmp_path / "repos"
    assert official_runtime_repo_path("Echo-Infinity", env=env) == tmp_path / "repos" / "Echo-Infinity"
    assert checkpoint_root_path("echo-infinity", env=env) == tmp_path / "ckpt" / "echo-infinity"
    assert hfd_root_path("org--model", env=env) == tmp_path / "hfd" / "org--model"
    assert conda_root_path(env) == tmp_path / "conda"
    assert conda_envs_root_path(env) == tmp_path / "envs"
    assert resolve_worldfoundry_path("${WORLDFOUNDRY_HFD_ROOT}/mirror", {"WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd")}) == (
        tmp_path / "hfd" / "mirror"
    )


def test_core_media_inference_handles_paths_urls_and_queries() -> None:
    assert infer_media_kind("frame.PNG") is MediaKind.IMAGE
    assert infer_media_kind("https://example.test/render.mp4?token=abc") is MediaKind.VIDEO
    assert infer_media_kind("s3://bucket/scene.ply") is MediaKind.GEOMETRY
    assert infer_media_kind("metrics.jsonl") is MediaKind.JSON
    assert infer_media_kind("weights.safetensors") is MediaKind.BINARY
    assert infer_media_kind("unknown.worldfoundry") is MediaKind.UNKNOWN


def test_core_media_suffix_and_mime_helpers() -> None:
    assert suffix_for_uri("https://example.test/archive.tar.gz?download=1") == ".tar.gz"
    assert infer_media_kind("archive.tar.gz") is MediaKind.ARCHIVE
    assert guess_mime_type("preview.webp") == "image/webp"
    assert is_media_path("clip.mov", MediaKind.VIDEO)
    assert is_media_path("scene.spz", "geometry")
    assert not is_media_path("notes.unknown", MediaKind.TEXT)


def test_core_artifact_visualization_helpers_are_canonical() -> None:
    from worldfoundry.core.io import (
        colorize_depth_map,
        colorize_normal_map,
        depth_to_uint8,
        depths_to_pil_images,
        flow_to_image,
        parse_game_control_config,
        process_game_control_video,
        render_point_cloud,
        save_openloop_action_comparison,
        visualize_tensor_bcthw,
        visualize_wasd_and_rotation_ui,
    )
    depth = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)
    depth_uint8 = depth_to_uint8(depth)
    assert depth_uint8 is not None
    assert depth_uint8.dtype == np.uint8
    assert len(depths_to_pil_images(depth, mode="grayscale")) == 1
    image = render_point_cloud(
        points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        colors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        height=8,
        width=8,
        splat_radius=1,
    )
    assert image.size == (8, 8)
    assert render_point_cloud.__module__ == "worldfoundry.core.media.artifacts"

    key_data, mouse_data = parse_game_control_config(
        (
            np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32),
            np.array([[0.0, 0.0], [0.1, -0.2]], dtype=np.float32),
        )
    )
    assert key_data[0]["W"] is True
    assert key_data[1]["D"] is True
    assert mouse_data[0] == (320, 176)

    frames = np.zeros((2, 180, 240, 3), dtype=np.float32)
    overlaid = visualize_wasd_and_rotation_ui(
        frames,
        wasd_actions=np.array([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32),
        ijkl_actions=np.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32),
    )
    assert overlaid.shape == frames.shape
    assert overlaid.dtype == np.float32

    flow_vis = flow_to_image(np.zeros((4, 5, 2), dtype=np.float32))
    assert flow_vis.shape == (4, 5, 3)
    assert flow_vis.dtype == np.uint8

    colored_depth = colorize_depth_map(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    colored_normal = colorize_normal_map(np.zeros((2, 2, 3), dtype=np.float32))
    assert colored_depth.shape == (2, 2, 3)
    assert colored_normal.shape == (2, 2, 3)
    assert process_game_control_video.__module__ == "worldfoundry.core.media.artifacts"
    assert save_openloop_action_comparison.__module__ == "worldfoundry.core.media.artifacts"
    assert visualize_tensor_bcthw.__module__ == "worldfoundry.core.media.artifacts"


def test_disk_preflight_helpers_use_worldfoundry_env(tmp_path: Path, monkeypatch) -> None:
    from worldfoundry.core.io.filesystem.disk import (
        CACHE_MIN_FREE_ENV,
        bytes_from_gib,
        cache_min_free_bytes,
        default_worldfoundry_cache_dir,
        ensure_free_disk,
    )

    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv(CACHE_MIN_FREE_ENV, "0")

    ensure_free_disk(tmp_path / "new-cache", required_bytes=0, label="test cache")

    assert bytes_from_gib(1.5) == 1610612736
    assert cache_min_free_bytes() == 0
    assert default_worldfoundry_cache_dir() == tmp_path / "cache"
    assert (tmp_path / "new-cache").is_dir()


def test_download_to_cache_publishes_validated_file(tmp_path: Path, monkeypatch) -> None:
    from worldfoundry.core.io.assets.download import download_to_cache
    from worldfoundry.core.io.filesystem.disk import CACHE_MIN_FREE_ENV

    source = tmp_path / "source.txt"
    source.write_text("worldfoundry", encoding="utf-8")
    monkeypatch.setenv(CACHE_MIN_FREE_ENV, "0")

    cached = download_to_cache(
        source.as_uri(),
        cache_dir=tmp_path / "cache",
        validator=lambda path: path.read_text(encoding="utf-8") == "worldfoundry",
    )

    assert cached.read_text(encoding="utf-8") == "worldfoundry"
    assert cached == tmp_path / "cache" / "source.txt"


def test_s3_uri_validator_is_available_without_credentials() -> None:
    from worldfoundry.core.io.filesystem.s3_filesystem import S3FileSystem

    assert S3FileSystem.validate_checkpoint_id("s3://bucket/path/checkpoint")
    assert not S3FileSystem.validate_checkpoint_id("/local/checkpoint")


def test_checkpoint_remap_and_lazy_facade() -> None:
    torch = pytest.importorskip("torch")
    from worldfoundry.core.model_loading.checkpoints import load_checkpoint, remap_checkpoint_keys

    state = {
        "blocks.0.attn.to_q.weight": torch.tensor([1.0]),
        "unchanged": torch.tensor([2.0]),
    }

    remapped = remap_checkpoint_keys(state, {r"^blocks\.(\d+)\.attn\.(.*)$": r"layers.\1.\2"})

    assert callable(load_checkpoint)
    assert "layers.0.to_q.weight" in remapped
    assert remapped["unchanged"] is state["unchanged"]


def test_model_loading_reads_safetensors_index(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from worldfoundry.core.io import dump_serialized
    from worldfoundry.core.model_loading import load_state_dict

    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    safetensors.save_file({"a": torch.tensor([1.0])}, str(shard_a))
    safetensors.save_file({"b": torch.tensor([2.0])}, str(shard_b))
    dump_serialized(
        {"metadata": {}, "weight_map": {"a": shard_a.name, "b": shard_b.name}},
        tmp_path / "model.safetensors.index.json",
    )

    state = load_state_dict(tmp_path / "model.safetensors.index.json")

    assert torch.equal(state["a"], torch.tensor([1.0]))
    assert torch.equal(state["b"], torch.tensor([2.0]))


def test_uri_storage_and_serialization_helpers_round_trip_common_formats(tmp_path: Path) -> None:
    from worldfoundry.core.io import (
        dump_serialized,
        exists_uri,
        load_serialized,
        parse_uri_scheme,
        read_binary_uri,
        write_binary_uri,
    )

    json_path = tmp_path / "item.json"
    jsonl_path = tmp_path / "items.jsonl"
    yaml_path = tmp_path / "item.yaml"
    binary_path = tmp_path / "blob.bin"

    dump_serialized({"value": 1}, json_path)
    dump_serialized([{"a": 1}, {"b": 2}], jsonl_path)
    dump_serialized({"name": "worldfoundry"}, yaml_path)
    write_binary_uri(binary_path, b"worldfoundry")

    assert parse_uri_scheme(json_path) == "file"
    assert exists_uri(json_path)
    assert load_serialized(json_path) == {"value": 1}
    assert load_serialized(jsonl_path) == [{"a": 1}, {"b": 2}]
    assert load_serialized(yaml_path) == {"name": "worldfoundry"}
    assert read_binary_uri(binary_path) == b"worldfoundry"


def test_core_json_and_text_writers_create_parent_dirs_and_normalize_payloads(tmp_path: Path) -> None:
    from dataclasses import dataclass

    from worldfoundry.core.io import (
        append_jsonl,
        read_json_object,
        read_jsonl_objects,
        write_json,
        write_jsonl,
        write_text_file,
    )

    @dataclass
    class Payload:
        path: Path
        tags: tuple[str, ...]

    json_path = tmp_path / "nested" / "payload.json"
    jsonl_path = tmp_path / "nested" / "rows.jsonl"
    text_path = tmp_path / "nested" / "notes" / "item.txt"

    write_json(json_path, {"payload": Payload(path=tmp_path / "asset.bin", tags=("core", "io"))})
    write_jsonl(jsonl_path, [{"row": 1}])
    append_jsonl(jsonl_path, {"row": 2, "path": tmp_path / "asset.bin"})
    write_text_file(text_path, "worldfoundry\n")

    assert read_json_object(json_path)["payload"] == {
        "path": str(tmp_path / "asset.bin"),
        "tags": ["core", "io"],
    }
    assert read_jsonl_objects(jsonl_path) == [
        {"row": 1},
        {"path": str(tmp_path / "asset.bin"), "row": 2},
    ]
    assert text_path.read_text(encoding="utf-8") == "worldfoundry\n"


def test_core_json_writers_reject_non_finite_numbers_without_publishing(tmp_path: Path) -> None:
    from worldfoundry.core.io import write_json, write_jsonl

    json_path = tmp_path / "non-finite.json"
    jsonl_path = tmp_path / "non-finite.jsonl"

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json(json_path, {"score": float("nan")})
    with pytest.raises(ValueError, match="Out of range float values"):
        write_jsonl(jsonl_path, [{"score": float("inf")}])

    assert not json_path.exists()
    assert not jsonl_path.exists()


def test_patchify_image_round_trips_nchw_arrays() -> None:
    image = np.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6)

    patches, spec = patchify_image(image, (2, 3), layout="nchw")
    restored = unpatchify_image(patches, spec)

    assert isinstance(spec, PatchGridSpec)
    assert spec.grid_shape == (2, 2)
    assert spec.patch_count == 4
    assert spec.patch_vector_size == 18
    assert patches.shape == (2, 4, 18)
    assert np.array_equal(restored, image)


def test_patchify_image_round_trips_nhwc_arrays() -> None:
    image = np.arange(2 * 4 * 6 * 3).reshape(2, 4, 6, 3)

    patches, spec = patchify_image(image, 2, layout="nhwc")
    restored = unpatchify_image(patches, spec)

    assert spec.grid_shape == (2, 3)
    assert patches.shape == (2, 6, 12)
    assert np.array_equal(restored, image)


def test_patchify_image_validates_spatial_divisibility() -> None:
    image = np.zeros((1, 3, 5, 6))

    with pytest.raises(ValueError, match="not divisible"):
        patchify_image(image, (2, 3), layout="nchw")


def test_unpatchify_image_validates_patch_shape() -> None:
    image = np.zeros((1, 3, 4, 4))
    patches, spec = patchify_image(image, 2)

    with pytest.raises(ValueError, match="does not match expected"):
        unpatchify_image(patches[:, :-1, :], spec)


def test_rotary_embedding_helpers_preserve_shape_and_rotate_numpy_arrays() -> None:
    value = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    cos = np.ones((1, 1, 4))
    sin = np.zeros((1, 1, 4))
    freqs_cos, freqs_sin = rotary_frequencies(seq_len=3, dim=4)

    assert np.array_equal(apply_rotary_embedding(value, cos, sin), value)
    assert freqs_cos.shape == (3, 4)
    assert freqs_sin.shape == (3, 4)

    with pytest.raises(ValueError, match="positive even"):
        apply_rotary_embedding(value, cos, sin, rotary_dim=3)


def test_normalization_helpers_work_on_numpy_arrays() -> None:
    value = np.array([[3.0, 4.0]])
    normalized = rms_norm(value, eps=0.0)
    scaled = layer_scale(value, np.array([2.0, 3.0]), bias=np.array([1.0, -1.0]))

    assert normalized.shape == value.shape
    assert np.mean(normalized * normalized) == pytest.approx(1.0)
    assert np.array_equal(scaled, np.array([[7.0, 11.0]]))


def test_transformer_shape_helpers_split_heads_and_build_masks() -> None:
    value = np.arange(2 * 3 * 12).reshape(2, 3, 12)

    heads = split_attention_heads(value, 4)
    restored = merge_attention_heads(heads)
    spec = transformer_shape_spec(12, 4, mlp_ratio=2.5, multiple_of=8)
    mask = causal_attention_mask(2, 4)
    strict_mask = causal_attention_mask(2, 4, include_self=False)

    assert isinstance(spec, TransformerShapeSpec)
    assert attention_head_dim(12, 4) == 3
    assert mlp_hidden_size(12, multiplier=2.5, multiple_of=8) == 32
    assert spec.head_dim == 3
    assert spec.mlp_hidden_size == 32
    assert heads.shape == (2, 4, 3, 3)
    assert np.array_equal(restored, value)
    assert mask.tolist() == [[True, True, True, False], [True, True, True, True]]
    assert strict_mask.tolist() == [[True, True, False, False], [True, True, True, False]]


def test_transformer_shape_helpers_validate_dimensions() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        attention_head_dim(10, 4)
    with pytest.raises(ValueError, match="at least sequence"):
        split_attention_heads(np.zeros((12,)), 4)
    with pytest.raises(ValueError, match="query_len"):
        causal_attention_mask(0)


def test_attention_helper_matches_torch_sdpa_shape_if_torch_available() -> None:
    torch = pytest.importorskip("torch")

    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 5)
    output = scaled_dot_product_attention(query, key, value)
    backend = attention_backend_info()

    assert output.shape == (1, 2, 3, 5)
    assert backend.backend


def test_attention_backend_probe_normalizes_sparse_attention_names() -> None:
    assert normalize_attention_backend("FLASH_ATTN") == "flash_attention"
    assert normalize_attention_backend("SAGE_ATTN_THREE") == "sage_attention_3"
    assert attention_backend_from_env({"WORLDFOUNDRY_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN"}) == "video_sparse_attention"
    assert resolve_attention_backend("math") == "torch"
    assert any(capability.name == "torch" and capability.usable for capability in attention_backend_report())

    with pytest.raises(ValueError, match="Unknown attention backend"):
        normalize_attention_backend("unknown-kernel")


def test_attention_forward_uses_core_backend_probe_and_cpu_fallback() -> None:
    torch = pytest.importorskip("torch")
    from worldfoundry.core.attention import attention_forward

    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 5)

    output = attention_forward(query, key, value, compatibility_mode=True)

    assert output.shape == (1, 2, 3, 5)


def test_compile_module_helper_is_opt_in(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from worldfoundry.core import compile_module_if_enabled

    calls: list[dict[str, object]] = []
    module = torch.nn.Linear(2, 2)

    def fake_compile(target, **kwargs):
        calls.append(kwargs)
        return target

    monkeypatch.setattr(torch, "compile", fake_compile, raising=False)

    assert compile_module_if_enabled(module) is module
    assert calls == []
    assert compile_module_if_enabled(module, enabled=True, backend="eager", mode="reduce-overhead") is module
    assert calls == [{"backend": "eager", "mode": "reduce-overhead"}]


def test_layerwise_cpu_offload_helper_is_safe_without_cuda(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from worldfoundry.core import enable_layerwise_cpu_offload, layerwise_offload_mutation_scope

    model = torch.nn.Sequential(torch.nn.ModuleList([torch.nn.Linear(2, 2)]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    handle = enable_layerwise_cpu_offload(model)

    assert handle.enabled is False
    assert handle.layer_count == 0
    assert "CUDA" in handle.reason
    with layerwise_offload_mutation_scope(model):
        pass


def test_native_attention_runs_with_cpu_math_backend() -> None:
    torch = pytest.importorskip("torch")
    from worldfoundry.core.attention import NativeAttention

    attention = NativeAttention(qkv_format="bshd", backend="math")
    query = torch.randn(2, 3, 4, 5)
    key = torch.randn(2, 3, 4, 5)
    value = torch.randn(2, 3, 4, 7)

    output = attention(query, key, value)

    assert output.shape == (2, 3, 4, 7)


def test_block_kv_cache_rolls_local_window_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    from worldfoundry.core.attention import BlockKVCache

    cache = BlockKVCache(
        k_shape=(1, 4, 2),
        v_shape=(1, 4, 3),
        seq_dim=1,
        chunk_size=2,
        window_size=4,
        device="cpu",
        dtype=torch.float32,
    )
    chunks = [torch.full((1, 2, 2), float(index)) for index in range(3)]
    values = [torch.full((1, 2, 3), float(index)) for index in range(3)]

    for index, (key, value) in enumerate(zip(chunks, values, strict=True)):
        cache.before_update(index)
        cache.update(key, value)
        cached_key = cache.cached_k()
        cache.after_update(index)

    assert cached_key.shape == (1, 4, 2)
    assert torch.equal(cached_key[:, :2], chunks[1])
    assert torch.equal(cached_key[:, 2:], chunks[2])


def test_3d_rope_builds_cpu_frequency_table() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("einops")
    from worldfoundry.core.attention import RotaryPositionEmbedding3D

    rope = RotaryPositionEmbedding3D(
        head_dim=12,
        len_h=2,
        len_w=3,
        len_t=2,
        interleaved=True,
        device=torch.device("cpu"),
    )

    freqs = rope.shift_t(autoregressive_index=2)

    assert freqs.shape == (12, 1, 1, 12)
    assert freqs.device.type == "cpu"


def test_foundation_layer_inventory_groups_hashes_and_import_metadata(tmp_path: Path) -> None:
    repo = tmp_path
    first = repo / "worldfoundry/base_models/foo/attention.py"
    second = repo / "worldfoundry/synthesis/bar/bar_runtime/attention.py"
    third = repo / "worldfoundry/training/baz/rope.py"
    shim_a = repo / "worldfoundry/base_models/foo/attn_shim.py"
    shim_b = repo / "worldfoundry/pipelines/foo/attn_shim.py"
    symlink = repo / "worldfoundry/base_models/foo/attention_symlink.py"
    for path in (first, second, third, shim_a, shim_b):
        path.parent.mkdir(parents=True, exist_ok=True)
    source = "import torch\n\nclass AttentionBlock:\n    pass\n"
    first.write_text(source, encoding="utf-8")
    second.write_text(source, encoding="utf-8")
    third.write_text("from .attention import AttentionBlock\n\ndef rotate_half(x):\n    return x\n", encoding="utf-8")
    shim_source = "from worldfoundry.base_models.foo.attention import AttentionBlock\n\n__all__ = [\"AttentionBlock\"]\n"
    shim_a.write_text(shim_source, encoding="utf-8")
    shim_b.write_text(shim_source, encoding="utf-8")
    symlink.symlink_to("attention.py")

    entries = build_foundation_layer_inventory(repo, include_runtime=True)
    paths = [entry.path for entry in entries]

    assert paths == [
        "worldfoundry/base_models/foo/attention.py",
        "worldfoundry/base_models/foo/attn_shim.py",
        "worldfoundry/pipelines/foo/attn_shim.py",
        "worldfoundry/synthesis/bar/bar_runtime/attention.py",
        "worldfoundry/training/baz/rope.py",
    ]
    assert file_duplicate_groups(entries) == ((
        "worldfoundry/base_models/foo/attention.py",
        "worldfoundry/synthesis/bar/bar_runtime/attention.py",
    ),)
    assert ast_duplicate_groups(entries) == file_duplicate_groups(entries)
    assert entries[0].owner_package == "worldfoundry.base_models.foo"
    assert entries[0].imports == ("torch",)
    assert entries[0].public_symbols == ("AttentionBlock",)
    assert entries[1].reexport_only is True
    assert entries[3].runtime_owned is True

    filtered = iter_foundation_layer_files(repo, include_runtime=False)
    assert [str(path.relative_to(repo)) for path in filtered] == [
        "worldfoundry/base_models/foo/attention.py",
        "worldfoundry/base_models/foo/attn_shim.py",
        "worldfoundry/pipelines/foo/attn_shim.py",
        "worldfoundry/training/baz/rope.py",
    ]
