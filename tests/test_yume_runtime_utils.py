
import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")
import torch

from worldfoundry.synthesis.visual_generation.yume.yume_runtime.yume.worldfoundry_runtime import (
    YumeRuntime,
)


class _FakeYumeModel:
    patch_size = (1, 2, 2)
    sp_size = 8


def test_yume_runtime_raises_underestimated_seq_len_to_actual_latent_tokens():
    runtime = YumeRuntime(model=_FakeYumeModel(), device="cpu", weight_dtype=torch.bfloat16)
    latent = torch.zeros(16, 14, 68, 120)
    arg_c = {"seq_len": 26520}
    arg_null = {"seq_len": 40000}

    runtime._ensure_arg_seq_len(arg_c, arg_null, latent=latent)

    assert arg_c["seq_len"] == 28560
    assert arg_null["seq_len"] == 40000
