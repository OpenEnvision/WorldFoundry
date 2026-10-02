# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""S3-backed filesystem and distributed-checkpoint readers/writers.

PyTorch DCP and sharded safetensors expect a ``FileSystemBase``. This
module implements that ABI with boto3 so the same load/save path can
target ``s3://bucket/key``:

- :class:`S3FileSystem` — stream create/read/write (64 MiB spool then
  disk), recursive listing, download, HEAD with optional checksums.
  Credentials come from a JSON file passed at construction (the same
  shape Cosmos checkpoints use).
- :class:`S3StorageWriter` / :class:`S3StorageReader` — DCP adapters
  that plug the filesystem into ``torch.distributed.checkpoint``.

Not a general object-store client. Rank-0 cache sync for recipes is
:mod:`worldfoundry.core.io.assets.s3_sync`. Generic ``file://`` / ``s3://``
URI I/O for artifacts is :mod:`worldfoundry.core.io.filesystem.storage`.
"""

import io
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Generator, Union
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.filesystem import FileSystemBase


# ──────────────────────────────────────────────────────────────────────────
# DCP FileSystemBase — 64 MiB memory spool, then disk; not a general S3 client
# ──────────────────────────────────────────────────────────────────────────

S3_STREAM_SPOOL_MAX_SIZE = 64 * 1024 * 1024


class S3FileSystem(FileSystemBase):
    """AWS S3-backed implementation of ``FileSystemBase``.

    Wraps a ``boto3`` client so the same code paths used for local checkpoints
    can read and write to S3 transparently. Credentials are loaded from a
    JSON file passed at construction.

    Examples:

      >>> fs = S3FileSystem(credential_path="credentials/s3_checkpoint.secret")
      >>> with fs.create_stream("s3://bucket/key.bin", "rb") as f:
      ...     data = f.read()
    """

    def __init__(self, credential_path: str) -> None:
        """Build a boto3 client from a Cosmos-style credentials JSON file."""

        with open(credential_path, "r") as f:
            config = json.load(f)
        self.s3_client = boto3.client("s3", **config)

    @contextmanager
    def create_stream(self, path: Union[str, os.PathLike], mode: str) -> Generator[io.IOBase, None, None]:
        """Open an S3 object as a binary stream.

        For ``"rb"`` the object is downloaded into a seekable spooled file;
        for ``"wb"`` writes are buffered in the same kind of file and uploaded
        on context exit. Streams remain in memory up to 64 MiB and then roll
        over to a temporary file so large checkpoint shards do not require an
        equally large contiguous memory buffer.

        Args:
            path: ``s3://bucket/key`` URI.
            mode: ``"rb"`` to read or ``"wb"`` to write.

        Raises:
            ValueError: ``mode`` is neither ``"rb"`` nor ``"wb"``.
        """
        path_str = str(path)
        bucket, key = self._parse_s3_uri(path_str)

        if mode == "rb":
            stream = tempfile.SpooledTemporaryFile(max_size=S3_STREAM_SPOOL_MAX_SIZE, mode="w+b")
            try:
                self.s3_client.download_fileobj(bucket, key, stream)
                stream.seek(0)
                yield stream
            finally:
                stream.close()
        elif mode == "wb":
            stream = tempfile.SpooledTemporaryFile(max_size=S3_STREAM_SPOOL_MAX_SIZE, mode="w+b")
            try:
                yield stream
                stream.seek(0)
                self.s3_client.upload_fileobj(stream, bucket, key)
            finally:
                stream.close()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> Union[str, os.PathLike]:
        """Concatenate S3 path with suffix."""
        path_str = str(path)
        if path_str.endswith("/"):
            return f"{path_str}{suffix}"
        return f"{path_str}/{suffix}"

    def rename(self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]) -> None:
        """Move an S3 object via copy + delete (S3 has no rename primitive)."""
        src_bucket, src_key = self._parse_s3_uri(str(path))
        dst_bucket, dst_key = self._parse_s3_uri(str(new_path))

        copy_source = {"Bucket": src_bucket, "Key": src_key}
        self.s3_client.copy(copy_source, dst_bucket, dst_key)
        self.s3_client.delete_object(Bucket=src_bucket, Key=src_key)

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        """Initialize and validate S3 path."""
        path_str = str(path)
        if not path_str.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {path_str}. It must start with 's3://'")
        return path_str

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        """Simulate a directory by writing an empty trailing-slash object."""
        path_str = str(path)
        if not path_str.endswith("/"):
            path_str += "/"

        bucket, key = self._parse_s3_uri(path_str)
        if key:
            self.s3_client.put_object(Bucket=bucket, Key=key)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """Validate if the checkpoint_id is a valid S3 URI."""
        checkpoint_id_str = str(checkpoint_id)
        try:
            if not checkpoint_id_str.startswith("s3://"):
                return False
            parsed = urlparse(checkpoint_id_str)
            return bool(parsed.netloc and parsed.path)  # Must have bucket and key
        except Exception:
            return False

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        """Return True if the S3 object exists."""
        bucket, key = self._parse_s3_uri(str(path))
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                return False
            raise

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        """Remove a file from S3."""
        bucket, key = self._parse_s3_uri(str(path))
        self.s3_client.delete_object(Bucket=bucket, Key=key)

    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """Split an ``s3://bucket/key`` URI into ``(bucket, key)``.

        Raises:
            ValueError: ``uri`` is not an ``s3://`` URI or has no bucket.
        """
        uri = uri if isinstance(uri, str) else str(uri)
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}. Must start with 's3://'")

        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket:
            raise ValueError(f"Invalid S3 URI: {uri}. No bucket specified")

        return bucket, key

    def list_files_recursive(self, s3_dir: Union[str, os.PathLike]) -> list[str]:
        """Return object key suffixes under *s3_dir*, skipping empty folder markers."""
        bucket, prefix = self._parse_s3_uri(str(s3_dir).removesuffix("/"))
        prefix = prefix.removesuffix("/")
        scan_prefix = f"{prefix}/" if prefix else ""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=scan_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") and obj.get("Size", 0) == 0:
                    continue
                suffix = key[len(scan_prefix) :] if scan_prefix else key
                if suffix:
                    out.append(suffix)
        return sorted(out)

    def download_to_local(self, s3_uri: Union[str, os.PathLike], local_path: Union[str, os.PathLike]) -> None:
        """Download one object to *local_path*, creating parent directories."""
        bucket, key = self._parse_s3_uri(str(s3_uri))
        local_path_str = str(local_path)
        os.makedirs(os.path.dirname(local_path_str), exist_ok=True)
        self.s3_client.download_file(bucket, key, local_path_str)

    def head_object(self, s3_uri: Union[str, os.PathLike], checksum_mode: bool = False) -> dict[str, Any]:
        """Return S3 object metadata; enable checksums when *checksum_mode* is set."""
        bucket, key = self._parse_s3_uri(str(s3_uri))
        kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if checksum_mode:
            kwargs["ChecksumMode"] = "ENABLED"
        return self.s3_client.head_object(**kwargs)

    def close(self) -> None:
        """Release the underlying boto3 client."""
        self.s3_client.close()


class S3StorageWriter(FileSystemWriter):
    """S3-backed writer for ``torch.distributed.checkpoint``."""

    def __init__(self, credential_path: str, path: str, **kwargs) -> None:
        """
        Args:
            credential_path: Path to the AWS S3 credentials JSON.
            path: ``s3://`` URI to write checkpoints to.
            **kwargs: Forwarded to the parent writer.
        """
        super().__init__(path=path, sync_files=False, **kwargs)
        self.fs = S3FileSystem(credential_path)
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """True when *checkpoint_id* is an ``s3://`` URI DCP can hand to this writer."""

        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


class S3StorageReader(FileSystemReader):
    """S3-backed reader for ``torch.distributed.checkpoint``."""

    def __init__(self, credential_path: str, path: Union[str, os.PathLike]) -> None:
        """
        Args:
            credential_path: Path to the AWS S3 credentials JSON.
            path: ``s3://`` URI to read checkpoints from.
        """
        super().__init__(path=path)
        self.fs = S3FileSystem(credential_path)
        self.path = self.fs.init_path(path)
        self.sync_files = False

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """True when *checkpoint_id* is an ``s3://`` URI DCP can hand to this reader."""

        return S3FileSystem.validate_checkpoint_id(checkpoint_id)
