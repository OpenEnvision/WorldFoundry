from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

from worldfoundry.evaluation.api.registry import (
    DuplicateRegistryKeyError,
    MetricRegistry,
    MetricSpec,
    MetricSpecRegistry,
    ModelManifest,
    ModelManifestRegistry,
    ModelRegistry,
    UnknownRegistryKeyError,
    WorldModelManifest,
)


class RegistryCoreTest(unittest.TestCase):
    def test_model_registry_resolves_names_and_aliases(self) -> None:
        manifest = ModelManifest(
            model_id="matrix-game-2",
            name="Matrix Game 2",
            aliases=("mg2",),
            capabilities=("navigation",),
        )
        registry = ModelManifestRegistry([manifest])

        self.assertIs(registry.get("matrix-game-2"), manifest)
        self.assertIs(registry.get(" MG2 "), manifest)
        self.assertEqual(registry.resolve_name("matrix game 2"), "matrix-game-2")
        self.assertEqual(registry.keys(), ["matrix-game-2"])
        self.assertEqual(registry.list(), [manifest])
        self.assertIn("MATRIX-GAME-2", registry)

    def test_metric_registry_lists_in_registration_order(self) -> None:
        clip_score = MetricSpec(
            id="clip_score",
            aliases=("clip",),
            display_name="CLIPScore",
            output_unit="score",
            higher_is_better=True,
        )
        motion_accuracy = MetricSpec(
            id="motion_accuracy",
            aliases=("motion",),
            output_unit="score",
        )
        registry = MetricSpecRegistry([clip_score])
        registry.register(motion_accuracy)

        self.assertEqual(registry.list(), [clip_score, motion_accuracy])
        self.assertIs(registry.get("CLIP"), clip_score)
        self.assertEqual(registry.get("motion").output_unit, "score")

    def test_short_model_registry_alias_is_available(self) -> None:
        manifest = ModelManifest(model_id="alias-model")
        model_registry = ModelRegistry([manifest])
        metric = MetricSpec(id="alias-metric")
        metric_registry = MetricRegistry([metric])

        self.assertIs(model_registry.get("alias-model"), manifest)
        self.assertIs(metric_registry.get("alias-metric"), metric)
        self.assertIs(ModelManifest, WorldModelManifest)

    def test_duplicate_model_names_and_aliases_are_rejected(self) -> None:
        with self.assertRaises(DuplicateRegistryKeyError):
            ModelManifestRegistry(
                [
                    ModelManifest(model_id="model-a", aliases=("shared",)),
                    ModelManifest(model_id="model-b", aliases=("SHARED",)),
                ]
            )

        with self.assertRaises(DuplicateRegistryKeyError):
            ModelManifestRegistry(
                [
                    ModelManifest(model_id="model-a", aliases=("model-b",)),
                    ModelManifest(model_id="model-b"),
                ]
            )

    def test_duplicate_metric_aliases_on_one_spec_are_rejected(self) -> None:
        with self.assertRaises(DuplicateRegistryKeyError):
            MetricSpecRegistry(
                [
                    MetricSpec(id="clip_iqa", aliases=("quality", "QUALITY")),
                ]
            )

    def test_unknown_lookup_raises_specific_key_error(self) -> None:
        registry = MetricSpecRegistry([MetricSpec(id="clip_score")])

        with self.assertRaises(UnknownRegistryKeyError):
            registry.get("missing")

    def test_registry_imports_are_stdlib_or_local(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        registry_root = repo_root / "src" / "worldfoundry" / "evaluation" / "registry"
        allowed_modules = set(sys.stdlib_module_names) | {"__future__"}

        for path in registry_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = {alias.name.split(".", 1)[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        continue
                    modules = {node.module.split(".", 1)[0]} if node.module else set()
                else:
                    continue

                unexpected = modules - allowed_modules
                self.assertEqual(unexpected, set(), f"{path} imports {unexpected}")


if __name__ == "__main__":
    unittest.main()
