from __future__ import annotations

import sys

from worldfoundry.runtime.in_tree_cli import execute_in_tree


def test_execute_in_tree_rejects_unchanged_stale_output(tmp_path):
    output = tmp_path / "result.mp4"
    output.write_bytes(b"stale")

    result = execute_in_tree(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        output_path=output,
    )

    assert result["status"] == "failed"
    assert "produced no media artifact" in result["error"]
    assert output.read_bytes() == b"stale"


def test_execute_in_tree_copies_fresh_artifact_from_explicit_search_root(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    output = tmp_path / "normalized.mp4"

    result = execute_in_tree(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('generated/result.mp4').write_bytes(b'fresh')",
        ],
        cwd=tmp_path,
        output_path=output,
        search_roots=(generated,),
    )

    assert result["status"] == "succeeded"
    assert output.read_bytes() == b"fresh"
    assert result["metadata"]["source_artifact"].endswith("generated/result.mp4")
