"""PL-16 leftover: CWD-relative pipeline defaults resolve under artifact_root."""

from __future__ import annotations

import ast
from pathlib import Path

from worldfoundry.core.io import artifact_root_path

REPO_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_DEFAULTS = {
    "worldfoundry/pipelines/cut3r/pipeline_cut3r.py": "cut3r_output",
    "worldfoundry/pipelines/pi3/pipeline_pi3.py": "pi3_output",
    "worldfoundry/pipelines/pi3/pipeline_loger.py": "loger_output",
    "worldfoundry/pipelines/lingbot_map/pipeline_lingbot_map.py": "lingbot_map_output",
    "worldfoundry/pipelines/infinite_vggt/pipeline_infinite_vggt.py": "infinite_vggt_output",
    "worldfoundry/pipelines/vggt_omega/pipeline_vggt_omega.py": "vggt_omega_output",
    "worldfoundry/pipelines/depth_anything/pipeline_depth_anything_v1.py": "vis_depth",
    "worldfoundry/pipelines/worldlabs/pipeline_worldlabs.py": "worldlabs_assets",
    "worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_world_voyager.py": "hunyuan_world_voyager",
}

FORBIDDEN_CWD_DEFAULTS = (
    "./cut3r_output",
    "./pi3_output",
    "./loger_output",
    "./lingbot_map_output",
    "./infinite_vggt_output",
    "./vggt_omega_output",
    "./vis_depth",
    "./vis_video_depth",
    "./output/worldlabs_assets",
    "./output/hunyuan_world_voyager/represent_render",
    "./output/hunyuan_world_voyager/final_render",
)

LAZY_SIGNATURES = (
    ("worldfoundry/pipelines/cut3r/pipeline_cut3r.py", "run_two_stage_3dgs_video", "output_dir"),
    ("worldfoundry/pipelines/cut3r/pipeline_cut3r.py", "run_official_export", "output_dir"),
    ("worldfoundry/pipelines/vggt_omega/pipeline_vggt_omega.py", "run_official_scene_export", "output_dir"),
    (
        "worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_world_voyager.py",
        "from_pretrained",
        "represent_render_dir",
    ),
    (
        "worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_world_voyager.py",
        "__call__",
        "output_save_path",
    ),
)


def _parse(relpath: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relpath).read_text(encoding="utf-8"))


def _load_default_output_dir(relpath: str):
    tree = _parse(relpath)
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_default_output_dir"
    )
    namespace = {"artifact_root_path": artifact_root_path}
    exec(compile(ast.Module(body=[func], type_ignores=[]), relpath, "exec"), namespace)
    return namespace["_default_output_dir"]


def _iter_functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _argument_default(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str):
    args = func.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    offset = len(positional) - len(defaults)
    for index, arg in enumerate(positional):
        if arg.arg == name and index >= offset:
            return defaults[index - offset]
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if arg.arg == name:
            return default
    raise AssertionError(f"{func.name} has no default for {name}")


def test_leftover_cwd_relative_output_literals_are_gone() -> None:
    for relpath in PIPELINE_DEFAULTS:
        source = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        for literal in FORBIDDEN_CWD_DEFAULTS:
            assert literal not in source, f"{relpath} still contains {literal!r}"


def test_default_output_dir_helpers_resolve_under_artifact_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_ARTIFACT_DIR", str(tmp_path))
    for relpath, name in PIPELINE_DEFAULTS.items():
        resolve = _load_default_output_dir(relpath)
        resolved = Path(resolve())
        assert resolved == tmp_path / name
        assert resolved.is_absolute()


def test_named_defaults_stay_under_artifact_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_ARTIFACT_DIR", str(tmp_path))
    depth = _load_default_output_dir(
        "worldfoundry/pipelines/depth_anything/pipeline_depth_anything_v1.py"
    )
    voyager = _load_default_output_dir(
        "worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_world_voyager.py"
    )
    assert Path(depth("vis_video_depth")) == tmp_path / "vis_video_depth"
    assert Path(voyager("hunyuan_world_voyager/represent_render")) == (
        tmp_path / "hunyuan_world_voyager" / "represent_render"
    )
    assert Path(voyager("hunyuan_world_voyager/final_render")) == (
        tmp_path / "hunyuan_world_voyager" / "final_render"
    )


def test_default_helpers_call_artifact_root_lazily() -> None:
    for relpath in PIPELINE_DEFAULTS:
        tree = _parse(relpath)
        func = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_default_output_dir"
        )
        calls = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "artifact_root_path"
        ]
        assert calls, f"{relpath} _default_output_dir does not call artifact_root_path()"


def test_output_signature_defaults_are_none() -> None:
    for relpath, func_name, arg_name in LAZY_SIGNATURES:
        tree = _parse(relpath)
        func = next(node for node in _iter_functions(tree) if node.name == func_name)
        default = _argument_default(func, arg_name)
        assert default is not None
        assert isinstance(default, ast.Constant)
        assert default.value is None, f"{relpath}:{func_name}.{arg_name} default is {default.value!r}"


def test_touched_pipelines_have_no_import_time_chdir_or_sys_path() -> None:
    for relpath in PIPELINE_DEFAULTS:
        tree = _parse(relpath)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr in {"chdir", "insert"}
                ):
                    raise AssertionError(f"{relpath} still mutates process state at import time")
