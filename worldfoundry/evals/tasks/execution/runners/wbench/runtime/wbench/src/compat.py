"""
Compatibility stubs for optional third-party dependencies.

Import this module before importing DA3 or other third-party libs that have
heavy optional deps (moviepy, trimesh, plyfile, pycolmap, gsplat, evo, etc.).

Usage:
    import src.compat  # noqa: F401 — must be imported before DA3
"""

import sys
import types

_OPTIONAL_STUBS = [
    "pycolmap",
    "gsplat",
    "gsplat.rendering",
    "moviepy",
    "moviepy.editor",
    "trimesh",
]

for _mod in _OPTIONAL_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# evo — trajectory evaluation library used by DA3 for pose alignment
if "evo" not in sys.modules:
    _evo = types.ModuleType("evo")
    _evo_core = types.ModuleType("evo.core")
    _evo_traj = types.ModuleType("evo.core.trajectory")
    _evo_metrics = types.ModuleType("evo.core.metrics")

    class _PosePath3D:
        def __init__(self, *a, **k):
            pass

    class _PoseTrajectory3D(_PosePath3D):
        pass

    _evo_traj.PosePath3D = _PosePath3D
    _evo_traj.PoseTrajectory3D = _PoseTrajectory3D
    _evo_metrics.APE = type("APE", (), {})
    _evo_metrics.RPE = type("RPE", (), {})

    _evo.core = _evo_core
    _evo_core.trajectory = _evo_traj
    _evo_core.metrics = _evo_metrics

    sys.modules["evo"] = _evo
    sys.modules["evo.core"] = _evo_core
    sys.modules["evo.core.trajectory"] = _evo_traj
    sys.modules["evo.core.metrics"] = _evo_metrics

# plyfile needs PlyData/PlyElement attributes
if "plyfile" not in sys.modules:
    _plyfile = types.ModuleType("plyfile")
    _plyfile.PlyData = type("PlyData", (), {})
    _plyfile.PlyElement = type("PlyElement", (), {"describe": staticmethod(lambda *a, **k: None)})
    sys.modules["plyfile"] = _plyfile
