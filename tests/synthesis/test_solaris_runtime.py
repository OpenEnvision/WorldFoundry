import ast
from pathlib import Path


def test_solaris_inference_imports_sharding_helpers() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "worldfoundry/synthesis/visual_generation/solaris/solaris_runtime/src/runners/inference.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.Import)
        and any(alias.name == "src.utils.sharding" and alias.asname == "sharding_utils" for alias in node.names)
        for node in tree.body
    )
