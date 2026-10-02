from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_generation import get_benchmark_generation_adapter
from worldfoundry.evaluation.tasks.execution.runners.sana_wm_bench.sana_wm_bench_official_impl import main
from worldfoundry.evaluation.tasks.execution.runners.sana_wm_bench.sana_wm_bench_prompts import (
    CANONICAL_FPS,
    CANONICAL_FRAME_COUNT,
    materialize_sana_wm_bench_generation_requests,
)


def test_fixture_writes_sana_wm_scorecard(tmp_path: Path) -> None:
    assert main(["--run-fixture", "--output-dir", str(tmp_path), "--json"]) == 0
    scorecard = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["benchmark"]["benchmark_id"] == "sana-wm-bench"
    assert scorecard["metrics"]["per_split"]["simple_60s"]["vbench_overall"] == 81.1
    assert scorecard["validation"]["normalizer_only"] is True


def test_result_import_preserves_split_and_metric_conventions(tmp_path: Path) -> None:
    raw = tmp_path / "aggregate_results.json"
    raw.write_text(json.dumps({"records": [{"split": "hard_60s", "vbench": {"quality": 0.7955, "total": 0.7}, "temporal": {"imaging_quality_drop": 0.02}}]}), encoding="utf-8")
    assert main(["--official-results-path", str(raw), "--output-dir", str(tmp_path / "out")]) == 0
    scorecard = json.loads((tmp_path / "out" / "scorecard.json").read_text(encoding="utf-8"))
    values = scorecard["metrics"]["per_split"]["hard_60s"]
    assert values["vbench_overall"] == 79.55
    assert values["delta_iq"] == 0.02


def test_generation_requests_use_official_image_and_npz_layout(tmp_path: Path) -> None:
    root = tmp_path / "SANA-WM-Bench"
    (root / "images").mkdir(parents=True)
    split = root / "benchmark_v2_smooth_60s" / "sanawm_export_v2"
    split.mkdir(parents=True)
    (root / "images" / "game_style_001.png").write_bytes(b"image")
    (split / "game_style_001.npz").write_bytes(b"npz")
    (split / "run_manifest.jsonl").write_text(json.dumps({"id": "game_style_001", "prompt": "A game world.", "image_path": "images/game_style_001.png", "camera_path": "benchmark_v2_smooth_60s/sanawm_export_v2/game_style_001.npz"}) + "\n", encoding="utf-8")
    request = materialize_sana_wm_bench_generation_requests(dataset_root=root, split="simple_60s")[0]
    assert request.inputs["official_video_name"] == "game_style_001_generated.mp4"
    assert request.inputs["fps"] == CANONICAL_FPS
    assert request.inputs["num_frames"] == CANONICAL_FRAME_COUNT


def test_contract_and_generation_adapter_are_registered() -> None:
    contract = get_external_benchmark_contract("sana-wm-bench")
    assert "vbench_overall" in contract.metric_ids
    assert get_benchmark_generation_adapter("sana-wm-bench") is not None
