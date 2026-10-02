"""JSON, YAML, and JSONL convenience wrappers for local paths.

Thin stdlib / PyYAML helpers used by recipes. Structured dump/load with
format inference, URI backends, and ``allow_nan=False`` ledgers live in
:mod:`worldfoundry.core.io.formats.serialization` and
:mod:`worldfoundry.core.io.filesystem.integrity`. These wrappers do **not** ban NaN
unless the caller passes that flag through ``**kwargs``.

Public surface: ``json_*`` / ``yaml_*`` / ``jsonl_*`` plus
:class:`Jsonl` (in-memory reader/writer) and the verb-first aliases.
"""

import json
import os.path as path
from io import StringIO

import yaml

from worldfoundry.core.utils.tensors.arrays import any_to_primitive
from worldfoundry.core.io.filesystem.file_utils import f_join

__all__ = [
    "json_load",
    "json_loads",
    "jsonl_load",
    "yaml_load",
    "yaml_loads",
    "json_dump",
    "json_dumps",
    "jsonl_dump",
    "yaml_dump",
    "yaml_dumps",
    "json_or_yaml_load",
    "json_or_yaml_dump",
    "Jsonl",
    # ---------------- Aliases -----------------
    "load_json",
    "loads_json",
    "load_jsonl",
    "load_yaml",
    "loads_yaml",
    "dump_json",
    "dumps_json",
    "dump_jsonl",
    "dump_yaml",
    "dumps_yaml",
    "load_json_or_yaml",
    "dump_json_or_yaml",
]

from typing import Dict, List

from typing_extensions import Literal

# ──────────────────────────────────────────────────────────────────────────
# JSON / JSONL / YAML — local files only; no URI routing, no NaN ban by default
# ──────────────────────────────────────────────────────────────────────────


def json_load(*file_path, **kwargs):
    """Load JSON from a joined local path; ``**kwargs`` go to :func:`json.load`."""

    file_path = f_join(file_path)
    with open(file_path, "r") as fp:
        return json.load(fp, **kwargs)


def json_loads(string, **kwargs):
    """Parse a JSON string; same kwargs contract as :func:`json.loads`."""

    return json.loads(string, **kwargs)


def jsonl_load(*file_path, **kwargs):
    """Load every line as JSON; blank lines become parse errors, not skips."""

    file_path = f_join(file_path)
    data = []
    for line in open(file_path):
        data.append(json.loads(line, **kwargs))
    return data


def json_dump(data, *file_path, convert_to_primitive=False, **kwargs):
    """Write JSON; optionally coerce tensors via :func:`any_to_primitive` first."""

    if convert_to_primitive:
        data = any_to_primitive(data)
    file_path = f_join(file_path)
    with open(file_path, "w") as fp:
        json.dump(data, fp, **kwargs)


def json_dumps(data, convert_to_primitive=False, **kwargs):
    """
    Returns: string
    """
    if convert_to_primitive:
        data = any_to_primitive(data)
    return json.dumps(data, **kwargs)


def jsonl_dump(data, *file_path):
    """Overwrite a JSONL file; *data* must be a sequence (not a mapping)."""

    from worldfoundry.core.io.filesystem.file_utils import is_sequence

    assert is_sequence(data)
    data = any_to_primitive(data)
    file_path = f_join(file_path)
    with open(file_path, "w") as fp:
        for line in data:
            print(json.dumps(line), file=fp, flush=True)


def yaml_load(*file_path, loader=yaml.safe_load, **kwargs):
    """Load YAML with ``safe_load`` by default so tags cannot construct objects."""

    file_path = f_join(file_path)
    with open(file_path, "r") as fp:
        return loader(fp, **kwargs)


def yaml_loads(string, *, loader=yaml.safe_load, **kwargs):
    """Parse a YAML string; default loader is :func:`yaml.safe_load`."""

    return loader(string, **kwargs)


def yaml_dump(data, *file_path, dumper=yaml.safe_dump, convert_to_primitive=False, **kwargs):
    """Write YAML with stable insertion order (``sort_keys=False``) unless overridden."""

    if convert_to_primitive:
        data = any_to_primitive(data)
    file_path = f_join(file_path)
    indent = kwargs.pop("indent", 2)
    default_flow_style = kwargs.pop("default_flow_style", False)
    sort_keys = kwargs.pop("sort_keys", False)  # preserves original dict order
    with open(file_path, "w") as fp:
        dumper(
            data,
            stream=fp,
            indent=indent,
            default_flow_style=default_flow_style,
            sort_keys=sort_keys,
            **kwargs,
        )


def yaml_dumps(data, *, dumper=yaml.safe_dump, convert_to_primitive=False, **kwargs):
    "Returns: string"
    if convert_to_primitive:
        data = any_to_primitive(data)
    stream = StringIO()
    indent = kwargs.pop("indent", 2)
    default_flow_style = kwargs.pop("default_flow_style", False)
    sort_keys = kwargs.pop("sort_keys", False)  # preserves original dict order
    dumper(
        data,
        stream,
        indent=indent,
        default_flow_style=default_flow_style,
        sort_keys=sort_keys,
        **kwargs,
    )
    return stream.getvalue()


# ──────────────────────────────────────────────────────────────────────────
# Suffix dispatch — .json vs .yml/.yaml only; other extensions raise IOError
# ──────────────────────────────────────────────────────────────────────────


def json_or_yaml_load(*file_path, **loader_kwargs):
    """
    Args:
        file_path: JSON or YAML loader depends on the file extension

    Raises:
        IOError: if extension is not ".json", ".yml", or ".yaml"
    """
    file_path = str(f_join(file_path))
    if file_path.endswith(".json"):
        return json_load(file_path, **loader_kwargs)
    elif file_path.endswith(".yml") or file_path.endswith(".yaml"):
        return yaml_load(file_path, **loader_kwargs)
    else:
        raise IOError(f'unknown file extension: "{file_path}", loader supports only ".json", ".yml", ".yaml"')


def json_or_yaml_dump(data, *file_path, **dumper_kwargs):
    """
    Args:
        file_path: JSON or YAML loader depends on the file extension

    Raises:
        IOError: if extension is not ".json", ".yml", or ".yaml"
    """
    file_path = str(f_join(file_path))
    if file_path.endswith(".json"):
        return json_dump(data, file_path, **dumper_kwargs)
    elif file_path.endswith(".yml") or file_path.endswith(".yaml"):
        return yaml_dump(data, file_path, **dumper_kwargs)
    else:
        raise IOError(f'unknown file extension: "{file_path}", dumper supports only ".json", ".yml", ".yaml"')


# ──────────────────────────────────────────────────────────────────────────
# Verb-first aliases (json_load → load_json) for recipe-style imports
# ──────────────────────────────────────────────────────────────────────────

load_json = json_load
load_yaml = yaml_load
load_jsonl = jsonl_load
loads_json = json_loads
loads_yaml = yaml_loads
dump_json = json_dump
dump_jsonl = jsonl_dump
dump_yaml = yaml_dump
dumps_json = json_dumps
dumps_yaml = yaml_dumps
load_json_or_yaml = json_or_yaml_load
dump_json_or_yaml = json_or_yaml_dump

# ──────────────────────────────────────────────────────────────────────────
# Jsonl — keep the whole file in memory; flush each append (eval ledgers)
# ──────────────────────────────────────────────────────────────────────────


class Jsonl:
    """
    Both reader and writer, as if everything's in-memory
    """

    def __init__(self, *file_path, mode: Literal["r", "w", "a"] = "a"):
        """
        Args:
            mode:
            - 'r': file must already exists
            - 'w': overwrite the file regardless of whether it exists or not
            - 'a': create a new file if doesn't exist, or append to an existing file
        """
        assert mode in "rwa"
        self._file_path = str(f_join(file_path))
        self._mode = mode
        if mode == "r":
            assert path.exists(self._file_path)
            self._fp = None
        else:
            self._fp = open(self._file_path, mode)
        if path.exists(self._file_path) and mode != "w":
            self.data = jsonl_load(self._file_path)
        else:
            self.data = []

    def append(self, data: Dict):
        """Append one mapping to memory and disk; raises in read mode."""

        if self._mode == "r":
            raise RuntimeError("Jsonl read mode cannot call append()")
        self.data.append(data)
        print(json_dumps(data), file=self._fp, flush=True)

    def extend(self, data_list: List[Dict]):
        """Append each mapping through :meth:`append` so flush stays per-row."""

        for data in data_list:
            self.append(data)

    def close(self):
        """Close the write handle; read mode has no handle."""

        if self._fp is not None:
            self._fp.close()

    def __getitem__(self, idx):
        """Index the in-memory snapshot, not a live file seek."""

        return self.data[idx]

    def __len__(self):
        """Number of rows currently held in memory."""

        return len(self.data)

    def __iter__(self):
        """Iterate the in-memory list (includes rows appended this session)."""

        return iter(self.data)

    def __enter__(self):
        """Context-manager identity; the file is already opened in ``__init__``."""

        return self

    def __exit__(self, type, value, traceback):
        """Always close the write handle, including on exception."""

        self.close()

    def __bool__(self):
        """True when at least one row is loaded or appended."""

        return bool(self.data)
