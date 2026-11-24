"""Canonical preencoded reader for the embedded LTX-2.5 trainer."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from safetensors.torch import load as load_safetensors

from ...data import (
    TarShardReader,
    read_index,
    resolve_index_path,
    resolver_from_config,
)
from ...data.index import IndexRow
from ...data.sampling import SamplePlan, plan_fingerprint
from ...errors import BackendContractError
from .artifact import PREENCODE_SCHEMA, PREENCODE_VERSION, TENSOR_SPECS
from .geometry import STABLE_GEOMETRY


@dataclass(frozen=True)
class TorchBatch:
    sample_id: str
    start_frame: int
    plan_fingerprint: str
    video_latent: torch.Tensor
    first_frame_latent: torch.Tensor
    video_prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor
    relative_w2c: torch.Tensor
    camera_k: torch.Tensor
    source_indices: torch.Tensor
    camera_source_indices: torch.Tensor


class VerifiedShardResolver:
    """Compatibility adapter over shared transport resolvers.

    Path containment, declared size, and bucket generation handling live in
    :mod:`solarwm.data.transport` so LTX, Wan, and H3 share one contract.
    """

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def resolve(self, row: IndexRow) -> Any:
        return self.delegate.resolve(row)


def verified_resolver_from_config(
    data: Mapping[str, Any],
) -> VerifiedShardResolver:
    transport = data.get("transport")
    if not isinstance(transport, Mapping):
        raise BackendContractError("LTX data.transport must be a mapping")
    kind = str(transport.get("kind") or "").strip().lower()
    root = str(transport.get("root") or "").strip()
    if kind not in {"local", "gcs"} or not root:
        raise BackendContractError("LTX data.transport requires kind local/gcs and a non-empty root")
    if kind == "local":
        if not root.startswith("/"):
            raise BackendContractError("local LTX data.transport.root must be absolute")
        if any(key in transport for key in ("cache_dir", "cache_max_gib")):
            raise BackendContractError("local LTX data.transport cannot configure a GCS cache")
        cache_dir = None
        max_gib = 256.0
    else:
        if not root.startswith("gs://"):
            raise BackendContractError("gcs LTX data.transport.root must be a gs:// URI")
        cache_dir = str(transport.get("cache_dir") or "")
        if not cache_dir.startswith("/"):
            raise BackendContractError("gcs LTX data.transport.cache_dir must be absolute")
        try:
            max_gib = float(transport["cache_max_gib"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendContractError("gcs LTX data.transport.cache_max_gib must be numeric") from exc
        if max_gib <= 0:
            raise BackendContractError("gcs LTX data.transport.cache_max_gib must be positive")
    return VerifiedShardResolver(
        resolver_from_config(
            root,
            cache_dir=cache_dir,
            max_gib=max_gib,
        )
    )


def _metadata(payload: bytes) -> Mapping[str, str]:
    if len(payload) < 10:
        raise BackendContractError("LTX safetensors payload is truncated")
    header_length = struct.unpack("<Q", payload[:8])[0]
    if header_length < 2 or 8 + header_length > len(payload):
        raise BackendContractError("LTX safetensors header length is invalid")
    try:
        header = json.loads(payload[8 : 8 + header_length])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendContractError("LTX safetensors header is invalid") from exc
    if not isinstance(header, Mapping) or set(header) != set(TENSOR_SPECS) | {"__metadata__"}:
        raise BackendContractError("LTX safetensors tensor names differ")
    metadata = header["__metadata__"]
    if not isinstance(metadata, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise BackendContractError("LTX safetensors metadata is invalid")
    return metadata


def _member(row: IndexRow) -> str:
    direct = row.values.get("preencoded_member") or row.values.get("ltx25_preencoded_member")
    if direct:
        return str(direct)
    members = row.values.get("members", {})
    if isinstance(members, Mapping):
        value = members.get("ltx25.safetensors")
        if value:
            return str(value)
    raise BackendContractError(f"preencoded row {row.sample_id!r} lacks its tensor member")


def _validate_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    plan: SamplePlan,
) -> None:
    if set(tensors) != set(TENSOR_SPECS):
        raise BackendContractError("preencoded LTX tensor inventory differs")
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}
    for name, spec in TENSOR_SPECS.items():
        value = tensors[name]
        if tuple(value.shape) != spec.shape or value.dtype != dtypes[spec.dtype]:
            raise BackendContractError(f"preencoded LTX tensor {name} layout differs")
    if not torch.equal(
        tensors["first_frame_latent"],
        tensors["video_latent"][:, :1],
    ):
        raise BackendContractError("preencoded first frame is not bit-equal to latent 0")
    mask = tensors["prompt_attention_mask"]
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.any()):
        raise BackendContractError("preencoded caption mask must be nonempty binary")
    if bool((mask[1:] < mask[:-1]).any()):
        raise BackendContractError("preencoded caption mask is not left padded")
    expected_source = torch.tensor(plan.source_frame_indices, dtype=torch.int64)
    if not torch.equal(tensors["source_indices"], expected_source):
        raise BackendContractError("preencoded source indices differ from canonical plan")
    camera_rows = torch.tensor(
        STABLE_GEOMETRY.camera_pixel_indices,
        dtype=torch.int64,
    )
    if not torch.equal(
        tensors["camera_source_indices"],
        expected_source.index_select(0, camera_rows),
    ):
        raise BackendContractError("preencoded camera rows differ from causal selection")
    viewmats = tensors["relative_w2c"]
    intrinsics = tensors["camera_K"]
    if not bool(torch.isfinite(viewmats).all() and torch.isfinite(intrinsics).all()):
        raise BackendContractError("preencoded LTX camera tensors are non-finite")
    if not torch.allclose(
        viewmats[0],
        torch.eye(4, dtype=torch.float32),
        rtol=0.0,
        atol=1e-5,
    ):
        raise BackendContractError("preencoded relative W2C row zero must be identity")
    if not bool((intrinsics[:, 0, 0] > 0).all() and (intrinsics[:, 1, 1] > 0).all()):
        raise BackendContractError("preencoded camera focal lengths must be positive")


def _fixed_plan(row: IndexRow) -> SamplePlan:
    try:
        start = int(row.values["start_frame"])
        indices = tuple(int(item) for item in row.values["source_frame_indices"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendContractError(f"preencoded row {row.sample_id!r} lacks fixed source indices") from exc
    if len(indices) != STABLE_GEOMETRY.pixel_frames or indices != tuple(
        range(start, start + STABLE_GEOMETRY.pixel_frames)
    ):
        raise BackendContractError(f"preencoded row {row.sample_id!r} source indices are not contiguous")
    return SamplePlan(
        sample_id=row.sample_id,
        key=row.key,
        shard=row.shard,
        row_ordinal=row.ordinal,
        repeat_ordinal=0,
        epoch=0,
        start_frame=start,
        source_frame_indices=indices,
        reader_rank=0,
        worker_id=0,
    )


def load_preencoded_row(
    row: IndexRow,
    *,
    shards: TarShardReader,
    plan: SamplePlan | None = None,
) -> TorchBatch:
    """Strictly materialize one indexed artifact without changing index order."""

    payload = shards.read(row, _member(row))
    metadata = _metadata(payload)
    if metadata.get("schema") != PREENCODE_SCHEMA:
        raise BackendContractError("preencoded LTX schema differs")
    if metadata.get("version") != PREENCODE_VERSION:
        raise BackendContractError("preencoded LTX artifact version differs")
    tensors = load_safetensors(payload)
    if plan is not None:
        selected = plan
    elif "start_frame" in row.values or "source_frame_indices" in row.values:
        selected = _fixed_plan(row)
    else:
        try:
            indices = tuple(int(item) for item in tensors["source_indices"].tolist())
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendContractError(f"preencoded row {row.sample_id!r} lacks tensor source indices") from exc
        if not indices:
            raise BackendContractError(f"preencoded row {row.sample_id!r} has empty tensor source indices")
        selected = SamplePlan(
            sample_id=row.sample_id,
            key=row.key,
            shard=row.shard,
            row_ordinal=row.ordinal,
            repeat_ordinal=0,
            epoch=0,
            start_frame=indices[0],
            source_frame_indices=indices,
            reader_rank=0,
            worker_id=0,
        )
    if (
        selected.row_ordinal != row.ordinal
        or selected.sample_id != row.sample_id
        or selected.key != row.key
        or selected.shard != row.shard
    ):
        raise BackendContractError("preencoded row and sample plan identities differ")
    if metadata.get("sample_id") != selected.sample_id:
        raise BackendContractError("preencoded payload sample_id differs from index")
    _validate_tensors(tensors, plan=selected)
    return TorchBatch(
        sample_id=selected.sample_id,
        start_frame=selected.start_frame,
        plan_fingerprint=plan_fingerprint((selected,)),
        video_latent=tensors["video_latent"].unsqueeze(0),
        first_frame_latent=tensors["first_frame_latent"].unsqueeze(0),
        video_prompt_embeds=tensors["video_prompt_embeds"].unsqueeze(0),
        prompt_attention_mask=tensors["prompt_attention_mask"].unsqueeze(0),
        relative_w2c=tensors["relative_w2c"].unsqueeze(0),
        camera_k=tensors["camera_K"].unsqueeze(0),
        source_indices=tensors["source_indices"].unsqueeze(0),
        camera_source_indices=tensors["camera_source_indices"].unsqueeze(0),
    )


class IndexedPreencodedSource:
    """Ordered random access used by standalone inference and validation."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        index_field: str = "index",
    ) -> None:
        data = config["data"]
        if not isinstance(data, Mapping):
            raise BackendContractError("LTX data config must be a mapping")
        index_data = dict(data)
        if index_field != "index":
            index_data["index"] = config[index_field]
        self.index_path = resolve_index_path(index_data, "index")
        self.rows = read_index(self.index_path)
        resolver = verified_resolver_from_config(data)
        self.shards = TarShardReader(
            resolver,
            max_open=int(data.get("tar_cache_size", 4)),
        )
        self._by_id = {row.sample_id: row for row in self.rows}

    def get(self, sample_id: str) -> TorchBatch:
        try:
            row = self._by_id[sample_id]
        except KeyError as exc:
            raise BackendContractError(f"inference sample {sample_id!r} is outside the fixed index") from exc
        return load_preencoded_row(row, shards=self.shards)

    def case_fingerprint(self, sample_id: str, batch: TorchBatch | None = None) -> str:
        value = batch or self.get(sample_id)
        digest = hashlib.blake2s()
        digest.update(value.relative_w2c.contiguous().numpy().tobytes())
        digest.update(value.camera_k.contiguous().numpy().tobytes())
        return digest.hexdigest()

    def close(self) -> None:
        self.shards.close()


__all__ = [
    "IndexedPreencodedSource",
    "TorchBatch",
    "VerifiedShardResolver",
    "load_preencoded_row",
    "verified_resolver_from_config",
]
