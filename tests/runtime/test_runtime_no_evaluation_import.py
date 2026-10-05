"""Regression tests for SA-10: runtime layer must not import evaluation.

``worldfoundry/runtime/{assets,conda,benchmark_repos}.py`` used to import path
constants and manifest loaders from ``worldfoundry.evaluation.utils`` — an
upward dependency from the runtime layer into the evaluation framework.  The
loaders now live in ``worldfoundry.core.io.manifests`` and the constants are
derived from ``worldfoundry.core.io.paths``; ``evaluation.utils`` re-exports
the same objects so its public contract is unchanged.
"""

from __future__ import annotations

import subprocess
import sys


def test_runtime_modules_import_without_evaluation():
    code = (
        "import sys\n"
        "import worldfoundry.runtime.assets\n"
        "import worldfoundry.runtime.conda\n"
        "import worldfoundry.runtime.benchmark_repos\n"
        "loaded = sorted(name for name in sys.modules "
        "if name.startswith('worldfoundry.evaluation'))\n"
        "assert not loaded, f'runtime pulled in evaluation modules: {loaded}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        (sys.executable, "-c", code),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_evaluation_utils_reexports_core_manifest_loaders():
    import worldfoundry.core.io.manifests as core_manifests
    import worldfoundry.evaluation.utils as evaluation_utils

    assert evaluation_utils.load_manifest is core_manifests.load_manifest
    assert evaluation_utils.load_manifest_collection is core_manifests.load_manifest_collection
    assert evaluation_utils.manifest_paths is core_manifests.manifest_paths
    assert evaluation_utils.MANIFEST_SUFFIXES == core_manifests.MANIFEST_SUFFIXES


def test_runtime_constants_match_evaluation_utils():
    import worldfoundry.evaluation.utils as evaluation_utils
    import worldfoundry.runtime.assets as runtime_assets
    import worldfoundry.runtime.benchmark_repos as runtime_benchmark_repos
    import worldfoundry.runtime.conda as runtime_conda

    assert runtime_conda.REPO_ROOT == evaluation_utils.REPO_ROOT
    assert runtime_conda.DATA_ROOT == evaluation_utils.DATA_ROOT
    assert runtime_assets.REPO_ROOT == evaluation_utils.REPO_ROOT
    assert runtime_assets.BENCHMARKS_DATA_ROOT == evaluation_utils.BENCHMARKS_DATA_ROOT
    assert runtime_benchmark_repos.REPO_ROOT == evaluation_utils.REPO_ROOT


def test_core_io_facade_exposes_manifest_loaders():
    import worldfoundry.core.io as core_io
    import worldfoundry.core.io.manifests as core_manifests

    assert core_io.load_manifest is core_manifests.load_manifest
    assert core_io.load_manifest_collection is core_manifests.load_manifest_collection
    assert core_io.manifest_paths is core_manifests.manifest_paths
    assert core_io.MANIFEST_SUFFIXES == core_manifests.MANIFEST_SUFFIXES
