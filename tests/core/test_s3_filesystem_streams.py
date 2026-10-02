from __future__ import annotations

import sys
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import BinaryIO
from unittest.mock import patch

missing_modules: dict[str, ModuleType] = {}
if find_spec("boto3") is None:
    boto3_stub = ModuleType("boto3")

    def _boto3_client(*_args, **_kwargs):
        return None

    boto3_stub.client = _boto3_client
    botocore_stub = ModuleType("botocore")
    botocore_exceptions_stub = ModuleType("botocore.exceptions")

    class _ClientError(Exception):
        pass

    botocore_exceptions_stub.ClientError = _ClientError
    missing_modules.update(
        {
            "boto3": boto3_stub,
            "botocore": botocore_stub,
            "botocore.exceptions": botocore_exceptions_stub,
        }
    )

if find_spec("torch") is None:
    torch_stub = ModuleType("torch")
    torch_distributed_stub = ModuleType("torch.distributed")
    torch_checkpoint_stub = ModuleType("torch.distributed.checkpoint")
    torch_checkpoint_filesystem_stub = ModuleType("torch.distributed.checkpoint.filesystem")

    class _FileSystemBase:
        pass

    class _FileSystemReader:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _FileSystemWriter:
        def __init__(self, *args, **kwargs) -> None:
            pass

    torch_checkpoint_stub.FileSystemReader = _FileSystemReader
    torch_checkpoint_stub.FileSystemWriter = _FileSystemWriter
    torch_checkpoint_filesystem_stub.FileSystemBase = _FileSystemBase
    torch_stub.distributed = torch_distributed_stub
    torch_distributed_stub.checkpoint = torch_checkpoint_stub
    torch_checkpoint_stub.filesystem = torch_checkpoint_filesystem_stub
    missing_modules.update(
        {
            "torch": torch_stub,
            "torch.distributed": torch_distributed_stub,
            "torch.distributed.checkpoint": torch_checkpoint_stub,
            "torch.distributed.checkpoint.filesystem": torch_checkpoint_filesystem_stub,
        }
    )

with patch.dict(sys.modules, missing_modules):
    s3_filesystem = import_module("worldfoundry.core.io.s3_filesystem")


class _FakeS3Client:
    def __init__(self, objects: dict[tuple[str, str], bytes] | None = None) -> None:
        self.objects = dict(objects or {})

    def download_fileobj(self, bucket: str, key: str, stream: BinaryIO) -> None:
        stream.write(self.objects[(bucket, key)])

    def upload_fileobj(self, stream: BinaryIO, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = stream.read()


def _filesystem(client: _FakeS3Client) -> s3_filesystem.S3FileSystem:
    filesystem = object.__new__(s3_filesystem.S3FileSystem)
    filesystem.s3_client = client
    return filesystem


def test_read_stream_rolls_over_and_remains_seekable(monkeypatch) -> None:
    monkeypatch.setattr(s3_filesystem, "S3_STREAM_SPOOL_MAX_SIZE", 8)
    payload = b"0123456789abcdef"
    filesystem = _filesystem(_FakeS3Client({("bucket", "checkpoint.bin"): payload}))

    with filesystem.create_stream("s3://bucket/checkpoint.bin", "rb") as stream:
        assert stream.tell() == 0
        assert getattr(stream, "_rolled") is True
        assert stream.read() == payload
        stream.seek(4)
        assert stream.read(4) == b"4567"

    assert stream.closed is True


def test_write_stream_rolls_over_and_uploads_from_start(monkeypatch) -> None:
    monkeypatch.setattr(s3_filesystem, "S3_STREAM_SPOOL_MAX_SIZE", 8)
    payload = b"0123456789abcdef"
    client = _FakeS3Client()
    filesystem = _filesystem(client)

    with filesystem.create_stream("s3://bucket/checkpoint.bin", "wb") as stream:
        stream.write(payload[:4])
        assert getattr(stream, "_rolled") is False
        stream.write(payload[4:])
        assert getattr(stream, "_rolled") is True
        stream.seek(0)
        assert stream.read() == payload

    assert stream.closed is True
    assert client.objects[("bucket", "checkpoint.bin")] == payload
