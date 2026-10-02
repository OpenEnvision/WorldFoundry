"""Structured serialization helpers for model-independent data files.

One dump/load surface over json, jsonl, yaml, pickle, numpy, torch, images,
and archives. Format is inferred from suffix so callers do not grow a
per-extension if/else. :func:`jsonable` walks dataclasses and tensors into
JSON-safe trees for logs.

JSONL writers flush on an interval (``WORLDFOUNDRY_JSONL_FLUSH_INTERVAL``)
so a crashed eval job keeps a prefix of results. Atomic replace for
integrity-critical files lives in :mod:`worldfoundry.core.io.filesystem.integrity`,
not here — this module is convenience I/O, not a crash-safe ledger.
"""

from __future__ import annotations

import gzip
import io
import json
import pickle
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import IO, Any, Mapping
from uuid import uuid4

import yaml

from worldfoundry.core.media.types import suffix_for_uri
from worldfoundry.core.io.filesystem.storage import read_binary_uri, read_text_uri, write_binary_uri, write_text_uri

_TEXT_FORMATS = {"txt", "text"}
_JSON_FORMATS = {"json"}
_YAML_FORMATS = {"yaml", "yml"}
_JSONL_FORMATS = {"jsonl", "ndjson"}
_PICKLE_FORMATS = {"pickle", "pkl"}
_GZIP_FORMATS = {"gz", "gzip"}
_NUMPY_FORMATS = {"npy", "npz"}
_TORCH_FORMATS = {"pt", "pth", "ckpt", "bin"}
_TORCHSCRIPT_FORMATS = {"jit", "torchscript", "torchjit"}
_IMAGE_FORMATS = {"jpg", "jpeg", "png", "bmp", "gif", "webp", "tif", "tiff"}
_VIDEO_FORMATS = {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "m4v"}
_MESH_FORMATS = {"ply", "stl", "obj", "glb", "gltf"}
_BYTE_FORMATS = {"byte", "bytes", "binary"}
_CSV_FORMATS = {"csv"}
_PANDAS_FORMATS = {"pandas", "parquet", "feather"}
_TAR_FORMATS = {"tar", "tgz", "tar.gz", "tar.xz", "tar.bz2"}

JSONL_FLUSH_INTERVAL_ENV = "WORLDFOUNDRY_JSONL_FLUSH_INTERVAL"
DEFAULT_JSONL_FLUSH_INTERVAL = 32

# ──────────────────────────────────────────────────────────────────────────
# JSON-safe trees and JSON / JSONL — allow_nan=False so NaN cannot leak
# ──────────────────────────────────────────────────────────────────────────


def _callable_reference(value: Any) -> str:
    """Stable ``module:qualname`` token for evidence logs; fall back to ``repr``."""

    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", "")
    if module and qualname:
        return f"{module}:{qualname}"
    return repr(value)


def jsonable(value: Any) -> Any:
    """Recursively convert common runtime objects into JSON-safe values.

    Args:
        value: Dataclass, ``to_dict`` object, path, mapping, sequence, tensor/
            array-like object with ``tolist``, callable, primitive, or fallback
            object.

    Returns:
        A tree containing only JSON-compatible primitives and containers.

    Notes:
        Mapping keys are stringified, sets become lists, callables become a
        module/qualified-name record, and otherwise unsupported objects fall
        back to ``repr``. This is evidence serialization, not a reversible
        object codec.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return jsonable(to_dict())
        except TypeError:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return jsonable(to_list())
    if callable(value):
        return {"callable": _callable_reference(value)}
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON object file."""

    source_path = Path(path)
    payload = read_json(source_path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object in {source_path}")
    return dict(payload)


def read_json_or_jsonl(path: str | Path) -> Any:
    """Read JSON or JSONL based on the file suffix."""

    source_path = Path(path)
    if source_path.suffix.lower() == ".jsonl":
        return list(iter_jsonl(source_path))
    return read_json(source_path)


def _iter_jsonl_rows(path: Path) -> Iterator[tuple[int, Any]]:
    """Yield ``(1-based line, value)``; blank lines are skipped, not treated as null."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def iter_jsonl(path: str | Path) -> Iterator[Any]:
    """Yield decoded values from the non-empty lines of a local JSONL file."""

    for _, row in _iter_jsonl_rows(Path(path)):
        yield row


def iter_jsonl_objects(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield objects from JSONL without materializing the entire file."""

    source_path = Path(path)
    for line_number, row in _iter_jsonl_rows(source_path):
        if not isinstance(row, Mapping):
            raise TypeError(f"expected JSON object on JSONL line {line_number} in {source_path}")
        yield dict(row)


def read_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file containing one object per non-empty line."""

    return list(iter_jsonl_objects(path))


def _atomic_write_text(path: Path, text: str, *, atomic: bool) -> Path:
    """Write *text*; atomic mode uses a unique sibling temp so concurrent writers cannot interleave."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        # Unique sibling temp name (same pattern as ``write_jsonl``): a fixed
        # ``.name.tmp`` would let two concurrent writers truncate each other's
        # temp file and publish interleaved content.
        tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with tmp_path.open("x", encoding="utf-8") as handle:
                handle.write(text)
            tmp_path.replace(path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    else:
        path.write_text(text, encoding="utf-8")
    return path


def write_text_file(path: str | Path, payload: str, *, atomic: bool = True) -> Path:
    """Write UTF-8 text to a local file with optional sibling-temp replacement."""

    return _atomic_write_text(Path(path), payload, atomic=atomic)


def write_json(path: str | Path, payload: Any, *, atomic: bool = True) -> Path:
    """Write an indented JSON object with stable key ordering."""

    text = json.dumps(
        jsonable(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    return _atomic_write_text(Path(path), text, atomic=atomic)


def _write_jsonl_rows(handle: IO[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Encode each mapping as one JSON line; ``allow_nan=False`` rejects NaN / Inf."""

    handle.writelines(
        json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )


def _resolve_jsonl_flush_interval(value: int | None) -> int:
    """Resolve flush cadence; ``0`` means flush only on close. Negative values raise."""

    if value is None:
        import os

        configured = os.environ.get(JSONL_FLUSH_INTERVAL_ENV, "").strip()
        value = DEFAULT_JSONL_FLUSH_INTERVAL if not configured else int(configured)
    interval = int(value)
    if interval < 0:
        raise ValueError("JSONL flush interval must be non-negative")
    return interval


class JsonlWriter:
    """Keep one JSONL file handle open and periodically publish buffered rows.

    The default flush interval bounds progress loss after an interrupted run
    while avoiding one metadata-heavy open/close cycle per evaluation sample.
    Set ``WORLDFOUNDRY_JSONL_FLUSH_INTERVAL=1`` for the former per-row flush
    behavior, or ``0`` to flush only when the context closes.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "w",
        flush_every: int | None = None,
    ) -> None:
        """Open-on-enter writer; *mode* is ``a`` / ``w`` / ``x`` only."""

        if mode not in {"a", "w", "x"}:
            raise ValueError("JSONL writer mode must be 'a', 'w', or 'x'")
        self.path = Path(path)
        self.mode = mode
        self.flush_every = _resolve_jsonl_flush_interval(flush_every)
        self._handle: IO[str] | None = None
        self._pending_rows = 0

    def __enter__(self) -> "JsonlWriter":
        """Create the parent dir and open the handle; refuse a nested enter."""

        if self._handle is not None:
            raise RuntimeError("JSONL writer is already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open(self.mode, encoding="utf-8", buffering=1024 * 1024)
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        """Append one JSON-safe row; flush when the pending count hits *flush_every*."""

        if self._handle is None:
            raise RuntimeError("JSONL writer must be used inside a context manager")
        encoded = json.dumps(
            jsonable(row),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        self._handle.write(encoded)
        self._handle.write("\n")
        self._pending_rows += 1
        if self.flush_every and self._pending_rows >= self.flush_every:
            self.flush()

    def write_many(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Write each mapping through :meth:`write` so flush cadence is preserved."""

        for row in rows:
            self.write(row)

    def flush(self) -> None:
        """Push the OS buffer and reset the pending-row counter; no-op if already closed."""
        if self._handle is None:
            return
        self._handle.flush()
        self._pending_rows = 0

    def close(self) -> None:
        """Flush then close; idempotent so ``__exit__`` can always call it."""

        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        try:
            handle.flush()
        finally:
            handle.close()
            self._pending_rows = 0

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Always close the handle, including when the ``with`` body raised."""

        del exc_type, exc_value, traceback
        self.close()


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]], *, atomic: bool = True) -> Path:
    """Stream JSONL rows with stable key ordering and optional atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not atomic:
        with destination.open("w", encoding="utf-8") as handle:
            _write_jsonl_rows(handle, rows)
        return destination

    temporary_path = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            _write_jsonl_rows(handle, rows)
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> Path:
    """Append one stable JSON row to a JSONL file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                jsonable(row),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        handle.write("\n")
    return destination


def reset_jsonl(path: str | Path) -> Path:
    """Remove a JSONL file while preserving its parent directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    return destination


# ──────────────────────────────────────────────────────────────────────────
# Suffix-inferred dump / load — pickle is opt-in; torch defaults to weights_only
# ──────────────────────────────────────────────────────────────────────────


def infer_serialization_format(file: str | Path | IO[Any] | None, file_format: str | None = None) -> str:
    """Infer a normalized serialization format key."""

    if file_format:
        return _normalize_format(file_format)
    if file is None:
        raise ValueError("file_format must be provided when file is None")
    if hasattr(file, "read") or hasattr(file, "write"):
        raise ValueError("file_format must be provided for file-like objects")
    suffix = suffix_for_uri(str(file)).lstrip(".")
    if not suffix:
        raise ValueError(f"Cannot infer serialization format from {file!r}")
    return _normalize_format(suffix)


def load_serialized(
    file: str | Path | IO[Any],
    *,
    file_format: str | None = None,
    encoding: str = "utf-8",
    allow_pickle: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a structured object from a URI or file object.

    Pickle and gzip-pickle payloads are rejected by default because unpickling
    executes arbitrary code. Trusted callers must pass ``allow_pickle=True``.
    Torch formats independently default to ``weights_only=True``. ``.gz`` is
    treated as gzip-compressed pickle for backward compatibility; use an
    explicit non-gzip format for other compressed data.
    """

    fmt = infer_serialization_format(file, file_format)
    if fmt in _BYTE_FORMATS:
        return _read_bytes(file)
    if fmt in _TEXT_FORMATS:
        return _read_text(file, encoding=encoding)
    if fmt in _JSON_FORMATS:
        return json.loads(_read_text(file, encoding=encoding), **kwargs)
    if fmt in _YAML_FORMATS:
        loader = kwargs.pop("loader", yaml.safe_load)
        return loader(_read_text(file, encoding=encoding), **kwargs)
    if fmt in _JSONL_FORMATS:
        return [json.loads(line, **kwargs) for line in _read_text(file, encoding=encoding).splitlines() if line.strip()]
    if fmt in _PICKLE_FORMATS:
        _require_pickle_opt_in(file, allow_pickle=allow_pickle)
        return pickle.loads(_read_bytes(file), **kwargs)
    if fmt in _GZIP_FORMATS:
        _require_pickle_opt_in(file, allow_pickle=allow_pickle)
        return pickle.loads(gzip.decompress(_read_bytes(file)), **kwargs)
    if fmt in _NUMPY_FORMATS:
        import numpy as np

        return np.load(io.BytesIO(_read_bytes(file)), **kwargs)
    if fmt in _TORCH_FORMATS:
        import torch

        load_kwargs = {"map_location": kwargs.pop("map_location", "cpu"), **kwargs}
        if "weights_only" not in load_kwargs:
            load_kwargs["weights_only"] = True
        return _torch_load_from_bytes(_read_bytes(file), **load_kwargs)
    if fmt in _TORCHSCRIPT_FORMATS:
        import torch

        return torch.jit.load(io.BytesIO(_read_bytes(file)), **kwargs)
    if fmt in _IMAGE_FORMATS:
        return _load_image(file, **kwargs)
    if fmt in _VIDEO_FORMATS:
        return _load_video(file, fmt=fmt, **kwargs)
    if fmt in _CSV_FORMATS:
        import pandas as pd

        return pd.read_csv(io.BytesIO(_read_bytes(file)), **kwargs)
    if fmt in _PANDAS_FORMATS:
        import pandas as pd

        if fmt == "parquet":
            return pd.read_parquet(io.BytesIO(_read_bytes(file)), **kwargs)
        if fmt == "feather":
            return pd.read_feather(io.BytesIO(_read_bytes(file)), **kwargs)
        _require_pickle_opt_in(file, allow_pickle=allow_pickle)
        return pd.read_pickle(io.BytesIO(_read_bytes(file)), **kwargs)
    if fmt in _MESH_FORMATS:
        import trimesh

        return trimesh.load(io.BytesIO(_read_bytes(file)), file_type=fmt, **kwargs)
    if fmt in _TAR_FORMATS:
        mode = kwargs.pop("mode", "r|*")
        return tarfile.open(fileobj=io.BytesIO(_read_bytes(file)), mode=mode, **kwargs)
    raise TypeError(f"Unsupported serialization format: {fmt}")


def _require_pickle_opt_in(file: Any, *, allow_pickle: bool) -> None:
    """Refuse pickle / gzip-pickle unless the caller opts in on a trusted source."""

    if not allow_pickle:
        raise ValueError(
            f"Refusing to unpickle {file!r}; pass allow_pickle=True only for a trusted source"
        )


def dump_serialized(
    obj: Any,
    file: str | Path | IO[Any] | None = None,
    *,
    file_format: str | None = None,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> str | bytes | None:
    """Dump an object to a URI, file object, or string when ``file`` is None."""

    fmt = infer_serialization_format(file, file_format)
    if fmt in _TEXT_FORMATS:
        return _write_text_or_return(str(obj), file, encoding=encoding)
    if fmt in _JSON_FORMATS:
        return _write_text_or_return(
            json.dumps(obj, **{**kwargs, "allow_nan": False}),
            file,
            encoding=encoding,
        )
    if fmt in _YAML_FORMATS:
        dumper = kwargs.pop("dumper", yaml.safe_dump)
        text = dumper(obj, **{"sort_keys": False, **kwargs})
        return _write_text_or_return(text, file, encoding=encoding)
    if fmt in _JSONL_FORMATS:
        json_kwargs = {**kwargs, "allow_nan": False}
        text = "\n".join(json.dumps(item, **json_kwargs) for item in obj)
        if text:
            text += "\n"
        return _write_text_or_return(text, file, encoding=encoding)
    if fmt in _PICKLE_FORMATS:
        return _write_bytes_or_return(pickle.dumps(obj, **kwargs), file)
    if fmt in _GZIP_FORMATS:
        return _write_bytes_or_return(gzip.compress(pickle.dumps(obj, **kwargs)), file)
    if fmt in _NUMPY_FORMATS:
        import numpy as np

        buffer = io.BytesIO()
        if fmt == "npz":
            if isinstance(obj, dict):
                np.savez(buffer, **obj)
            else:
                np.savez(buffer, obj)
        else:
            np.save(buffer, obj, **kwargs)
        return _write_bytes_or_return(buffer.getvalue(), file)
    if fmt in _TORCH_FORMATS:
        import torch

        buffer = io.BytesIO()
        torch.save(obj, buffer, **kwargs)
        return _write_bytes_or_return(buffer.getvalue(), file)
    if fmt in _TORCHSCRIPT_FORMATS:
        buffer = io.BytesIO()
        obj.save(buffer, **kwargs)
        return _write_bytes_or_return(buffer.getvalue(), file)
    if fmt in _IMAGE_FORMATS:
        return _dump_image(obj, file, fmt=fmt, **kwargs)
    if fmt in _VIDEO_FORMATS:
        return _dump_video(obj, file, fmt=fmt, **kwargs)
    if fmt in _CSV_FORMATS:
        buffer = io.StringIO()
        obj.to_csv(buffer, **kwargs)
        return _write_text_or_return(buffer.getvalue(), file, encoding=encoding)
    if fmt in _PANDAS_FORMATS:
        buffer = io.BytesIO()
        if fmt == "parquet":
            obj.to_parquet(buffer, **kwargs)
        elif fmt == "feather":
            obj.to_feather(buffer, **kwargs)
        else:
            obj.to_pickle(buffer, **kwargs)
        return _write_bytes_or_return(buffer.getvalue(), file)
    if fmt in _TAR_FORMATS:
        mode = kwargs.pop("mode", "w")
        if file is None:
            raise ValueError("tar output requires a file path or file object")
        if isinstance(file, (str, Path)):
            with tarfile.open(str(file), mode=mode, **kwargs) as tar:
                for item in obj:
                    tar.add(item)
            return None
        with tarfile.open(fileobj=file, mode=mode, **kwargs) as tar:
            for item in obj:
                tar.add(item)
        return None
    raise TypeError(f"Unsupported serialization format: {fmt}")


# ──────────────────────────────────────────────────────────────────────────
# Format adapters — URI vs file-like; image / video / torch stay lazy-imported
# ──────────────────────────────────────────────────────────────────────────


def _normalize_format(value: str) -> str:
    """Fold aliases (``jpeg``→``jpg``, ``yml``→``yaml``) so suffix tables stay small."""

    fmt = value.strip().lower().lstrip(".")
    if fmt == "jpeg":
        return "jpg"
    if fmt == "yml":
        return "yaml"
    return fmt


def _read_bytes(file: str | Path | IO[Any]) -> bytes:
    """Read a URI via :func:`read_binary_uri` or consume a file-like as bytes."""

    if isinstance(file, (str, Path)):
        return read_binary_uri(file)
    data = file.read()
    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def _read_text(file: str | Path | IO[Any], *, encoding: str) -> str:
    """Read text from a URI or decode file-like bytes with *encoding*."""

    if isinstance(file, (str, Path)):
        return read_text_uri(file, encoding=encoding)
    data = file.read()
    if isinstance(data, bytes):
        return data.decode(encoding)
    return data


def _write_text_or_return(text: str, file: str | Path | IO[Any] | None, *, encoding: str) -> str | None:
    """Dump to a URI / file-like, or return *text* when *file* is ``None`` (dumps-style)."""

    if file is None:
        return text
    if isinstance(file, (str, Path)):
        write_text_uri(file, text, encoding=encoding)
    else:
        file.write(text)
    return None


def _write_bytes_or_return(data: bytes, file: str | Path | IO[Any] | None) -> bytes | None:
    """Binary counterpart of :func:`_write_text_or_return`."""

    if file is None:
        return data
    if isinstance(file, (str, Path)):
        write_binary_uri(file, data)
    else:
        file.write(data)
    return None


def _load_image(
    file: str | Path | IO[Any], *, fmt: str = "pil", size: int | tuple[int, int] | None = None, **kwargs: Any
) -> Any:
    """Decode an image into PIL / numpy / CHW torch; *fmt* here is the output kind, not the suffix."""

    from PIL import Image

    image = Image.open(io.BytesIO(_read_bytes(file)))
    image.load()
    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        image = image.resize(size)
    if fmt in {"pil", "image"}:
        return image
    if fmt in {"numpy", "np", "npy"}:
        import numpy as np

        return np.array(image, **kwargs)
    if fmt in {"torch", "th"}:
        import numpy as np
        import torch

        tensor = torch.from_numpy(np.array(image, **kwargs))
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        return tensor
    raise ValueError(f"Unsupported image output format: {fmt}")


def _dump_image(obj: Any, file: str | Path | IO[Any] | None, *, fmt: str, **kwargs: Any) -> str | None:
    """Encode a PIL-like object; *file* is required (images have no dumps-to-string form)."""

    if file is None:
        raise ValueError("image output requires a file path or file object")
    buffer = io.BytesIO()
    obj.save(buffer, format="JPEG" if fmt == "jpg" else fmt.upper(), **kwargs)
    return _write_bytes_or_return(buffer.getvalue(), file)


def _load_video(file: str | Path | IO[Any], *, fmt: str, mode: str = "rgb", **kwargs: Any) -> Any:
    """Decode via imageio; ``mode=\"gray\"`` expands to HxWx1 so layouts stay 4-D."""

    import imageio
    import numpy as np

    reader = imageio.get_reader(io.BytesIO(_read_bytes(file)), fmt, **kwargs)
    frames = []
    for frame in reader:
        if mode == "gray":
            import cv2

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.expand_dims(frame, axis=2)
        frames.append(frame)
    return np.array(frames), reader.get_meta_data()


def _dump_video(
    obj: Any,
    file: str | Path | IO[Any] | None,
    *,
    fmt: str,
    fps: int = 24,
    quality: int | None = 5,
    **kwargs: Any,
) -> str | None:
    """Encode a frame stack; *file* is required. ``macro_block_size=1`` avoids silent crops."""

    if file is None:
        raise ValueError("video output requires a file path or file object")
    import imageio
    import numpy as np

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    obj = np.asarray(obj)
    write_kwargs = {"fps": fps, "macro_block_size": 1, **kwargs}
    if quality is not None:
        write_kwargs["quality"] = quality
    buffer = io.BytesIO()
    imageio.mimsave(buffer, obj, fmt, **write_kwargs)
    return _write_bytes_or_return(buffer.getvalue(), file)


def _torch_load_from_bytes(data: bytes, **kwargs: Any) -> Any:
    """``torch.load`` with ``weights_only=True`` by default; never drop that flag on TypeError."""

    import pickle

    import torch

    allow_unsafe_pickle_fallback = bool(kwargs.pop("allow_unsafe_pickle_fallback", False))
    kwargs.setdefault("weights_only", True)
    try:
        return torch.load(io.BytesIO(data), **kwargs)
    except TypeError as exc:
        # Fail closed when the installed PyTorch does not support a requested
        # safety option.  Removing ``weights_only`` here would silently turn a
        # tensor-only load into unrestricted pickle execution.
        if "weights_only" in str(exc) and "weights_only" in kwargs:
            raise RuntimeError(
                "the installed PyTorch does not support safe weights-only deserialization"
            ) from exc
        raise
    except pickle.UnpicklingError as exc:
        if not (kwargs.get("weights_only") and allow_unsafe_pickle_fallback and "Weights only load failed" in str(exc)):
            raise
        kwargs["weights_only"] = False
        return torch.load(io.BytesIO(data), **kwargs)


__all__ = [
    "DEFAULT_JSONL_FLUSH_INTERVAL",
    "JSONL_FLUSH_INTERVAL_ENV",
    "JsonlWriter",
    "dump_serialized",
    "infer_serialization_format",
    "load_serialized",
]
