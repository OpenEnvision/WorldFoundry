# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the BigVGAN decoder's upstream license texts in public artifacts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[2]
BIGVGAN = "worldfoundry/base_models/diffusion_model/models/autoencoders/kandinsky6/mmaudio/ext/bigvgan_v2"


@pytest.mark.parametrize(
    ("original_path", "expected_sha256"),
    [
        (f"{BIGVGAN}/LICENSE", "5c7f573db5f807a9adc2a755c4901e203ea067f73c7a20fa6b703da7e77d7b35"),
        (f"{BIGVGAN}/incl_licenses/LICENSE_1", "5eec017ce442989e01a46cd9cd6aeb13cd0aa3b3943f681577bda00475afaa23"),
        (f"{BIGVGAN}/incl_licenses/LICENSE_2", "7441474b0fe03def0efc41a6340b4818359453c7cb32423c647839f8799e68df"),
        (f"{BIGVGAN}/incl_licenses/LICENSE_3", "7339fa418c9ad3e8e12e74ad0fd26a9cc4be8703f9c110728a992b193be85cb2"),
        (f"{BIGVGAN}/incl_licenses/LICENSE_4", "9a2fbb788a9d584a2a0766e4bb790c8368911143472d257952581455313cf2c6"),
        (f"{BIGVGAN}/incl_licenses/LICENSE_5", "b6234401b9831a86505fbeecad803a0b2fe17b06997f614d5325c6d0e1a6fa7f"),
        (f"{BIGVGAN}/incl_licenses/LICENSE_6", "3fb4aef5b76dc0ccb1f57499d75c2fe039df5c9793778ede172f5ac7a4045681"),
        (f"{BIGVGAN}/incl_licenses/LICENSE_7", "8f36ba0c214d4f487ffd7ee0b8528da9b67b459bef2d115de9cd001316589f46"),
        (f"{BIGVGAN}/incl_licenses/LICENSE_8", "ff474afe15f36bab44711fc45c1a77314a4b9d8c96fa2d333f3fefe719fd76f9"),
    ],
)
def test_bigvgan_license_texts_are_retained_verbatim(original_path: str, expected_sha256: str) -> None:
    notice = (REPO_ROOT / "THIRD-PARTY-NOTICES").read_bytes()
    label = f"Original-File: {original_path}\n".encode()
    assert notice.count(label) == 1
    tail = notice.split(label, 1)[1]
    record = re.search(rb"SHA256: [0-9a-f]{64}\nBytes: ([0-9]+)\n----- BEGIN RETAINED LICENSE TEXT -----\n", tail)
    assert record is not None
    text = tail[record.end() : record.end() + int(record.group(1))]
    assert hashlib.sha256(text).hexdigest() == expected_sha256
    assert not (REPO_ROOT / original_path).exists()
