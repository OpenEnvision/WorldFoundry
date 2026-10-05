from __future__ import annotations

import ast
from pathlib import Path


_KAIROS_DIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "worldfoundry/synthesis/visual_generation/kairos/kairos_runtime/kairos/modules/dits/kairos_dit.py"
)


def test_kairos_dit_uses_upstream_fla() -> None:
    source = _KAIROS_DIT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        (node.module, alias.name, alias.asname)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert ("fla.layers", "GatedDeltaNet", None) in imports
    assert ("fla.models.utils", "Cache", "FlaCache") in imports
    assert "kairos_fla" not in source
    assert "fla_Cahce" not in source


def test_kairos_tp_gated_delta_fallback_is_replicated() -> None:
    source = _KAIROS_DIT_PATH.read_text(encoding="utf-8")

    assert "self.gated_delta_is_replicated = True" in source
    assert "if self.use_tp_in_getaeddeltanet and self.world > 1:" in source
    assert "warnings.warn(" in source
    assert "tensor-parallel optimization" in source
    assert "o_seq_chunk = out_chunk" in source
    assert "_all_gather_seq_chunk" not in source
