# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""ViPE's cache loader imports native binaries without executing Python sources."""

import importlib.machinery
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext import build as native_build

pytestmark = pytest.mark.unit


def test_cached_vipe_extension_loads_a_cpu_native_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "vipe_ext.c"
    source.write_text(
        """
#include <Python.h>

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "vipe_ext", NULL, -1, NULL
};

PyMODINIT_FUNC PyInit_vipe_ext(void) {
    const char *names[] = {
        "corr_ext", "droid_net_ext", "grounding_dino_ext", "lietorch_ext",
        "scatter_ext", "slam_ext", "utils_ext", NULL
    };
    PyObject *module = PyModule_Create(&module_def);
    if (module == NULL) return NULL;
    for (int i = 0; names[i] != NULL; ++i) {
        PyObject *submodule = PyModule_New(names[i]);
        if (submodule == NULL || PyModule_AddObject(module, names[i], submodule) < 0) {
            Py_XDECREF(submodule);
            Py_DECREF(module);
            return NULL;
        }
    }
    return module;
}
""",
        encoding="utf-8",
    )
    library = tmp_path / f"vipe_ext{sysconfig.get_config_var('EXT_SUFFIX')}"
    subprocess.run(
        [
            *shlex.split(sysconfig.get_config_var("LDSHARED")),
            "-fPIC",
            f"-I{sysconfig.get_path('include')}",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setitem(sys.modules, "vipe_ext", None)
    paths = list(sys.path)

    module = native_build._load_cached(tmp_path)

    assert module is not None
    assert module.__file__ == str(library)
    assert isinstance(module.__loader__, importlib.machinery.ExtensionFileLoader)
    assert module.slam_ext.__name__ == "slam_ext"
    assert sys.modules["vipe_ext"] is module
    assert native_build.load_native_extension() is module
    assert sys.path == paths


def test_cached_vipe_extension_ignores_python_sources(tmp_path: Path) -> None:
    (tmp_path / "vipe_ext.py").write_text('raise AssertionError("Python source must not run")\n', encoding="utf-8")

    assert native_build._load_cached(tmp_path) is None
