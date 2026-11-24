# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from worldfoundry.synthesis.visual_generation.starwm import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


def test_public_model_path_selects_the_served_model_not_a_local_checkpoint(tmp_path: Path) -> None:
    from worldfoundry.pipelines.world_model.pipeline_runtime_manifest import StarWMPipeline

    pipeline = StarWMPipeline.from_pretrained(model_path="served-starwm", device="cpu")
    synthesis = pipeline.synthesis_model
    assert synthesis.options["served_model_id"] == "served-starwm"
    assert "repo_root" not in synthesis.options
    assert synthesis.device == "cpu"
    assert synthesis._checkpoint_paths() == []
    command = synthesis._command(output_path=tmp_path / "result.json", prompt="", plan_path=tmp_path / "plan.json")
    assert command[command.index("--model_id") + 1] == "served-starwm"
    assert synthesis.blocked_reason == runtime.BLOCKED_REASON


@pytest.mark.parametrize("model_option", ["served_model_id", "api_model_id", "vllm_model_id", "upstream_model_id"])
@pytest.mark.parametrize("mode", ["think", "nothink"])
@pytest.mark.parametrize("seed", [None, 0, 42])
def test_client_module_command_preserves_endpoint_options_and_model_aliases(
    model_option: str, mode: str, seed: int | None
) -> None:
    command = runtime.build_command(
        {
            "python": "/env/python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "output_path": "/out/fallback.jsonl",
            "options": {
                model_option: "served-starwm",
                "dataset_path": "/data/prompts.json",
                "output_file": "/out/selected.jsonl",
                "mode": mode,
                "api_base": "http://localhost:18000",
                "api_key": "test-key",
                "max_workers": 2,
                "seed": seed,
            },
        }
    )
    assert command[:3] == [
        "/env/python",
        "-m",
        "worldfoundry.synthesis.visual_generation.starwm.offline_infer.run_inference_client",
    ]
    assert dict(zip(command[3::2], command[4::2], strict=True)) == {
        "--input_file": "/data/prompts.json",
        "--output_file": "/out/selected.jsonl",
        "--mode": mode,
        "--api_base": "http://localhost:18000",
        "--api_key": "test-key",
        "--model_id": "served-starwm",
        "--max_workers": "2",
        **({} if seed is None else {"--seed": str(seed)}),
    }


@pytest.mark.parametrize("seed", [None, 0, 42])
def test_optional_seed_reaches_vllm_without_changing_sampling_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seed: int | None
) -> None:
    from worldfoundry.synthesis.visual_generation.starwm.offline_infer import run_inference_client as client

    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps([{"messages": [{"role": "user", "content": "Predict the next state."}]}]))
    output = tmp_path / "prediction.json"
    args = [
        "client",
        "--input_file",
        str(inputs),
        "--output_file",
        str(output),
        "--model_id",
        "served-starwm",
        "--mode",
        "nothink",
        "--max_workers",
        "1",
    ]
    if seed is not None:
        args.extend(["--seed", str(seed)])
    monkeypatch.setattr(sys, "argv", args)
    requests: list[dict[str, Any]] = []

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "Generated state"}}]}

    def post(url: str, *, headers: dict[str, str], json: dict[str, Any]) -> Response:
        assert url == "http://localhost:12000/v1/chat/completions"
        assert headers["Content-Type"] == "application/json"
        requests.append(json)
        return Response()

    monkeypatch.setattr(client.requests, "post", post)
    client.main()
    assert requests == [
        {
            "model": "served-starwm",
            "messages": [{"role": "user", "content": "Predict the next state.\n\n/no_think"}],
            "temperature": 0.6,
            "max_tokens": 5000,
            "top_p": 0.95,
            "top_k": 20,
            **({} if seed is None else {"seed": seed}),
        }
    ]
    assert json.loads(output.read_text())["output"] == "Generated state"
