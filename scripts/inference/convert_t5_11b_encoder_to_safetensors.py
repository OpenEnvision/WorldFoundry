#!/usr/bin/env python3
"""Safely extract a T5 encoder from a legacy Transformers pickle checkpoint.

The official ``google-t5/t5-11b`` release uses ``pytorch_model.bin``. This
helper performs a weights-only load in a patched Torch process, keeps exactly
the tensors required by ``T5EncoderModel``, materializes its tied encoder
embedding, and atomically writes ``model.safetensors``.
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import T5Config, T5EncoderModel

_ASSET_NAMES = (
    "config.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _torch_version() -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        raise RuntimeError(f"cannot parse Torch version: {torch.__version__!r}")
    return int(match.group(1)), int(match.group(2))


def _load_tensor_state_dict(source: Path) -> Mapping[str, torch.Tensor]:
    if _torch_version() < (2, 6):
        raise RuntimeError(
            "legacy pickle conversion requires Torch >=2.6 because older weights_only "
            "loading is affected by CVE-2025-32434; use an isolated conversion environment"
        )

    payload: object = torch.load(source, map_location="cpu", weights_only=True)
    while isinstance(payload, Mapping) and len(payload) == 1:
        key = next(iter(payload))
        if key not in {"state_dict", "model", "module", "model_state"}:
            break
        payload = payload[key]
    if not isinstance(payload, Mapping) or not payload:
        raise TypeError("checkpoint did not contain a non-empty state dict")

    invalid = sorted(
        str(name)
        for name, tensor in payload.items()
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
    )
    if invalid:
        raise TypeError(f"checkpoint contains non-tensor state entries: {invalid[:20]}")
    return payload  # type: ignore[return-value]


def _expected_encoder_shapes(config: T5Config) -> dict[str, tuple[int, ...]]:
    with torch.device("meta"):
        model = T5EncoderModel(config)
    shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    del model
    return shapes


def _extract_encoder_state(
    checkpoint: Mapping[str, torch.Tensor],
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    encoder = {
        name: tensor
        for name, tensor in checkpoint.items()
        if name == "shared.weight" or name.startswith("encoder.")
    }
    if "shared.weight" in encoder:
        # Legacy T5 checkpoints omit this tied alias. Safetensors requires a
        # distinct storage for each named tensor, so materialize it explicitly.
        encoder["encoder.embed_tokens.weight"] = encoder["shared.weight"].clone()

    actual = set(encoder)
    expected = set(expected_shapes)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            "encoder key mismatch: "
            f"missing={missing[:20]} unexpected={unexpected[:20]} "
            f"(expected {len(expected)}, found {len(actual)})"
        )

    mismatched = {
        name: {"expected": expected_shapes[name], "actual": tuple(encoder[name].shape)}
        for name in sorted(expected)
        if tuple(encoder[name].shape) != expected_shapes[name]
    }
    if mismatched:
        first = dict(list(mismatched.items())[:20])
        raise RuntimeError(f"encoder tensor shape mismatch: {first}")

    prepared: dict[str, torch.Tensor] = {}
    for name in sorted(expected):
        tensor = encoder[name].detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        prepared[name] = tensor
    return prepared


def _asset_pairs(source_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    return [
        (source_dir / name, output_dir / name)
        for name in _ASSET_NAMES
        if (source_dir / name).is_file()
    ]


def _check_asset_destinations(pairs: list[tuple[Path, Path]]) -> None:
    for source, destination in pairs:
        if destination.exists() or destination.is_symlink():
            try:
                if source.samefile(destination):
                    continue
            except OSError:
                pass
            raise FileExistsError(f"refusing to replace existing asset: {destination}")


def _link_assets(pairs: list[tuple[Path, Path]]) -> None:
    for source, destination in pairs:
        if destination.exists() or destination.is_symlink():
            continue
        relative_target = os.path.relpath(source, start=destination.parent)
        destination.symlink_to(relative_target)


def _validate_output(
    path: Path,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        expected = set(expected_shapes)
        if keys != expected:
            raise RuntimeError(
                f"written safetensors key mismatch: missing={sorted(expected - keys)[:20]} "
                f"unexpected={sorted(keys - expected)[:20]}"
            )
        mismatched = {
            name: {
                "expected": expected_shapes[name],
                "actual": tuple(handle.get_slice(name).get_shape()),
            }
            for name in sorted(expected)
            if tuple(handle.get_slice(name).get_shape()) != expected_shapes[name]
        }
    if mismatched:
        raise RuntimeError(f"written safetensors shape mismatch: {mismatched}")


def convert(
    source_model_dir: Path,
    output_model_dir: Path,
    *,
    source_filename: str = "pytorch_model.bin",
    link_assets: bool = True,
) -> Path:
    source_model_dir = source_model_dir.expanduser().resolve()
    output_model_dir = output_model_dir.expanduser().resolve()
    source = source_model_dir / source_filename
    output = output_model_dir / "model.safetensors"
    incomplete = output_model_dir / ".model.safetensors.incomplete"

    if not source.is_file():
        raise FileNotFoundError(f"missing legacy T5 checkpoint: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if incomplete.exists():
        raise FileExistsError(f"remove stale incomplete output before retrying: {incomplete}")

    config = T5Config.from_pretrained(source_model_dir, local_files_only=True)
    expected_shapes = _expected_encoder_shapes(config)
    asset_pairs = _asset_pairs(source_model_dir, output_model_dir) if link_assets else []
    _check_asset_destinations(asset_pairs)

    checkpoint = _load_tensor_state_dict(source)
    encoder = _extract_encoder_state(checkpoint, expected_shapes)
    del checkpoint

    output_model_dir.mkdir(parents=True, exist_ok=True)
    try:
        save_file(
            encoder,
            incomplete,
            metadata={
                "format": "pt",
                "architecture": "T5EncoderModel",
                "source": source.name,
            },
        )
        _validate_output(incomplete, expected_shapes)
        os.replace(incomplete, output)
        _link_assets(asset_pairs)
    except BaseException:
        incomplete.unlink(missing_ok=True)
        raise

    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a T5EncoderModel safetensors checkpoint from a legacy Transformers "
            "pytorch_model.bin. Requires enough RAM for the source state dict and Torch >=2.6."
        )
    )
    parser.add_argument("source_model_dir", type=Path)
    parser.add_argument("output_model_dir", type=Path)
    parser.add_argument("--source-filename", default="pytorch_model.bin")
    parser.add_argument(
        "--no-link-assets",
        action="store_true",
        help="do not create relative links for available config and tokenizer assets",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = convert(
        args.source_model_dir,
        args.output_model_dir,
        source_filename=args.source_filename,
        link_assets=not args.no_link_assets,
    )
    with safe_open(output, framework="pt", device="cpu") as handle:
        tensor_count = len(handle.keys())
    print(f"{output}\t{output.stat().st_size} bytes\t{tensor_count} tensors")


if __name__ == "__main__":
    main()
