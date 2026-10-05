from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HEAVY_MODULES = frozenset({"torch", "cv2", "diffusers", "trimesh", "viser", "open3d", "transformers"})
LIGHTWEIGHT_IMPORTS = (
    "worldfoundry",
    "worldfoundry.evaluation",
    "worldfoundry.evaluation.api",
    "worldfoundry.cli.main",
    "worldfoundry.mcp",
    "worldfoundry.studio",
)


def test_public_imports_do_not_load_heavy_optional_dependencies() -> None:
    offenders = []
    for module in LIGHTWEIGHT_IMPORTS:
        loaded = _heavy_modules_loaded_by_import(module)
        if loaded:
            offenders.append({"module": module, "heavy_modules": loaded})

    assert offenders == []


def _heavy_modules_loaded_by_import(module: str) -> list[str]:
    code = (
        "import importlib, json, sys; "
        f"heavy = {sorted(HEAVY_MODULES)!r}; "
        f"importlib.import_module({module!r}); "
        "print(json.dumps([name for name in heavy if name in sys.modules]))"
    )
    env = os.environ.copy()
    src = str(REPO_ROOT)
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not prior else f"{src}{os.pathsep}{prior}"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(proc.stdout)
