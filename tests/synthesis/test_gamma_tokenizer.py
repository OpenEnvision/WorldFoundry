
import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")
from unittest.mock import Mock

import torch

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference import tokenizer


def test_text_only_vision_processor_omits_empty_fps(monkeypatch) -> None:
    processor = tokenizer.Processor.__new__(tokenizer.Processor)
    processor.name = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor.is_vision_tokenizer = True
    processor.processor = Mock()
    processor.processor.apply_chat_template.return_value = "prompt"
    processor.processor.return_value = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    monkeypatch.setattr(tokenizer, "process_vision_info", lambda *args, **kwargs: (None, None, {}))
    monkeypatch.setattr(tokenizer, "extract_vision_info", lambda messages: [])

    processor.apply_chat_template([{"role": "user", "content": "hello"}])

    assert "fps" not in processor.processor.call_args.kwargs
