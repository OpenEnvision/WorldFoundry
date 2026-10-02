from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_py(code: str) -> None:
    """
    Run import contract checks in a fresh interpreter.

    In-process tests that pop worldfoundry.evaluation from sys.modules leave stale
    class objects in already-imported test modules, breaking isinstance and
    rerouted worker imports for later tests.
    """
    env = os.environ.copy()
    src = str(REPO_ROOT)
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not prior else f"{src}{os.pathsep}{prior}"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stdout or "") + (proc.stderr or "")
        raise AssertionError(f"isolated import check failed (exit {proc.returncode}):\n{detail}")


def test_top_level_import_does_not_eager_import_benchmarks_or_cli() -> None:
    _run_isolated_py(
        dedent(
            """
            import importlib
            import sys

            def _drop_module_tree(prefix):
                for name in list(sys.modules):
                    if name == prefix or name.startswith(prefix + "."):
                        sys.modules.pop(name)

            def _import_fresh_public_namespace():
                _drop_module_tree("worldfoundry.evaluation")
                worldfoundry = importlib.import_module("worldfoundry")
                if hasattr(worldfoundry, "evaluation"):
                    delattr(worldfoundry, "evaluation")
                return importlib.import_module("worldfoundry.evaluation")

            ev = _import_fresh_public_namespace()
            assert "worldfoundry.evaluation" in sys.modules
            assert not any(n.startswith("worldfoundry.evaluation.tasks") for n in sys.modules)
            assert not any(n.startswith("worldfoundry.cli") for n in sys.modules)
            need = {
                "api",
                "models",
                "reporting",
                "runner",
                "tasks",
            }
            assert need <= set(ev.__all__)
            assert "runtime" not in ev.__all__
            assert {"execute_evaluate_run", "load_benchmark_zoo_registry"} <= set(ev.__all__)
            """
        ).strip()
    )


def test_public_subpackages_import_from_worldfoundry_namespace() -> None:
    _run_isolated_py(
        dedent(
            """
            import importlib
            import sys

            def _drop_module_tree(prefix):
                for name in list(sys.modules):
                    if name == prefix or name.startswith(prefix + "."):
                        sys.modules.pop(name)

            def _import_fresh_public_namespace():
                _drop_module_tree("worldfoundry.evaluation")
                worldfoundry = importlib.import_module("worldfoundry")
                if hasattr(worldfoundry, "evaluation"):
                    delattr(worldfoundry, "evaluation")
                return importlib.import_module("worldfoundry.evaluation")

            ev = _import_fresh_public_namespace()
            for name in (
                "api",
                "models",
                "reporting",
                "runner",
                "tasks",
            ):
                module = getattr(ev, name)
                assert module is importlib.import_module("worldfoundry.evaluation." + name)
            assert importlib.import_module("worldfoundry.runtime") is not None
            assert "worldfoundry.cli" not in sys.modules
            """
        ).strip()
    )


def test_cli_import_avoids_benchmarks_and_runner() -> None:
    _run_isolated_py(
        dedent(
            """
            import importlib
            import sys

            def _drop_module_tree(prefix):
                for name in list(sys.modules):
                    if name == prefix or name.startswith(prefix + "."):
                        sys.modules.pop(name)

            _drop_module_tree("worldfoundry.evaluation")
            cli = importlib.import_module("worldfoundry.cli")
            assert cli.main is not None
            assert "worldfoundry.cli" in sys.modules
            assert "worldfoundry.evaluation.physics_backend" not in sys.modules
            assert "worldfoundry.evaluation.tasks" not in sys.modules
            assert "worldfoundry.evaluation.runner" not in sys.modules
            """
        ).strip()
    )


def test_lazy_benchmark_exports_resolve_through_public_namespace() -> None:
    _run_isolated_py(
        dedent(
            """
            import importlib
            import sys

            def _drop_module_tree(prefix):
                for name in list(sys.modules):
                    if name == prefix or name.startswith(prefix + "."):
                        sys.modules.pop(name)

            def _import_fresh_public_namespace():
                _drop_module_tree("worldfoundry.evaluation")
                worldfoundry = importlib.import_module("worldfoundry")
                if hasattr(worldfoundry, "evaluation"):
                    delattr(worldfoundry, "evaluation")
                return importlib.import_module("worldfoundry.evaluation")

            ev = _import_fresh_public_namespace()
            assert "worldfoundry.evaluation.tasks" not in sys.modules
            load = ev.load_benchmark_zoo_registry
            from worldfoundry.evaluation.tasks import load_benchmark_zoo_registry as benchmarks_load
            assert load is benchmarks_load
            assert "worldfoundry.evaluation.tasks" in sys.modules
            """
        ).strip()
    )
