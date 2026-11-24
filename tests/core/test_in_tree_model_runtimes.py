# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free guard: model runtimes import and launch only in-tree code.

Every model under ``worldfoundry/{synthesis,base_models,pipelines,operators}``
runs from the WorldFoundry package itself. Vendored upstream code is imported
with package-qualified or relative imports; child processes start in-tree
modules through :func:`worldfoundry.core.execution.process.package_module_command`.
The only exclusions are ``base_models`` subpackages that serve evaluation
metrics and judges exclusively; a second test keeps that list honest.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tokenize
from collections.abc import Iterator
from functools import cache
from importlib.util import resolve_name
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "worldfoundry"
MODEL_PACKAGES = ("synthesis", "base_models", "pipelines", "operators")
AREAS = (
    "base_models",
    "base_models/diffusion_model",
    "base_models/llm_mllm_core",
    "base_models/perception_core",
    "base_models/three_dimensions",
    "operators",
    "pipelines",
    "synthesis",
    "synthesis/action_generation",
    "synthesis/visual_generation",
)

# base_models subpackages whose only callers are evaluation metrics and judges
# (worldfoundry/evaluation and benchmark data), directly or through each other.
EVALUATION_ONLY = (
    "base_models/llm_mllm_core/mllm/instance_anomaly_detector",
    "base_models/llm_mllm_core/mllm/internvideo2",
    "base_models/llm_mllm_core/mllm/keye_vl",
    "base_models/llm_mllm_core/mllm/llava_next",
    "base_models/llm_mllm_core/mllm/modelscope_swift",
    "base_models/llm_mllm_core/mllm/perceptionlm",
    "base_models/llm_mllm_core/mllm/umt",
    "base_models/llm_mllm_core/mllm/videocon_physics",
    "base_models/llm_mllm_core/mllm/videophy2_autoeval",
    "base_models/llm_mllm_core/mllm/vila",
    "base_models/perception_core/action_recognition/latent_action",
    "base_models/perception_core/action_recognition/umt",
    "base_models/perception_core/captioning/grit",
    "base_models/perception_core/captioning/tag2text",
    "base_models/perception_core/detection/yolo_world",
    "base_models/perception_core/face/arcface",
    "base_models/perception_core/face/retinaface.py",
    "base_models/perception_core/frame_interpolation/vfimamba",
    "base_models/perception_core/general_perception/dino_embeddings.py",
    "base_models/perception_core/human_anatomy/vit_detector",
    "base_models/perception_core/optical_flow/raft",
    "base_models/perception_core/optical_flow/sea_raft",
    "base_models/perception_core/safety/nudenet",
    "base_models/perception_core/segment/grounded_segment_anything",
    "base_models/perception_core/tracking/aot",
    "base_models/perception_core/tracking/cotracker",
    "base_models/perception_core/tracking/dot",
    "base_models/perception_core/video_quality/dover",
    "base_models/perception_core/video_quality/dreamsim.py",
    "base_models/perception_core/video_quality/dreamsim_runtime",
    "base_models/perception_core/video_quality/fastvqa",
    "base_models/perception_core/video_quality/fetv_stylegan_v",
    "base_models/perception_core/video_quality/hpsv3.py",
    "base_models/perception_core/video_quality/hpsv3_runtime",
    "base_models/perception_core/video_quality/lpips.py",
    "base_models/perception_core/video_quality/transnetv2.py",
    "base_models/perception_core/video_text/blip2_feature_extractor.py",
    "base_models/perception_core/video_text/fetv_blip",
    "base_models/perception_core/video_text/viclip",
    "base_models/perception_core/video_text/vqa_score",
    "base_models/three_dimensions/slam/droid_slam",
    "base_models/three_dimensions/slam/mega_sam_runtime",
    "base_models/three_dimensions/slam/megasam.py",
    "base_models/three_dimensions/slam/megasam_resident.py",
)
# Packages that run models; evaluation-only code must stay unreachable from them.
MODEL_CALLERS = (*MODEL_PACKAGES, "cli", "core", "mcp", "runtime", "studio", "training")

_SYS_PATH_MUTATORS = {"insert", "append", "extend", "remove", "pop", "clear", "__setitem__", "__delitem__", "__iadd__"}
# File loaders that execute code addressed by path. ``ExtensionFileLoader`` is excluded: it loads only
# compiled extension modules (``.so``/``.pyd``), which the goal permits like any compiled extension, and
# cannot execute Python source. Loading Python source by path stays forbidden.
_FILE_LOADERS = {
    "run_path",
    "spec_from_file_location",
    "SourceFileLoader",
    "SourcelessFileLoader",
    "load_source",
    "execfile",
}
_REMOTE_CODE_CALLS = {"get_class_from_dynamic_module", "get_class_in_module"}
_EXTERNAL_SOURCE_HELPERS = {"official_runtime_repo_path", "model_source_root_path"}
_LAUNCHERS = {"torchrun", "accelerate", "deepspeed"}
_DISTRIBUTED_RUNNERS = {"torch.distributed.run", "torch.distributed.launch"}
_PATH_JOINS = {"join", "Path", "PurePath", "PosixPath", "joinpath"}
_PROCESS_CALLS = {
    "system",
    "popen",
    "run",
    "Popen",
    "call",
    "check_call",
    "check_output",
    "getoutput",
    "execv",
    "execvp",
}
_ENV_LOOKUPS = {"getenv", "get", "getenv_registered", "pop", "setdefault"}
_PYTHON_OPTIONS_WITH_VALUE = {"-X", "-W"}
_PYTHON_FILE = re.compile(r".*\w\.py")
_INSTALL = re.compile(r"\bpip3?\s+install\b|\bgit\s+clone\b|\bsetup\.py\s+(?:install|develop|build)")
_THIRDPARTY = re.compile(r"(?:^|[/\\])thirdparty(?:[/\\]|$)")
_MODULE_TARGET = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::[A-Za-z_]\w*)?")
_SOURCE_ENV = re.compile(r"[A-Z][A-Z0-9_]*(?:(?:SOURCE|CHECKOUT)[A-Z0-9_]*|_REPOS?(?:_ROOT|_DIR|_PATH)?)")


def _relative(path: Path) -> str:
    return path.relative_to(PACKAGE).as_posix()


def _excluded(rel: str) -> bool:
    return any(rel == entry or rel.startswith(entry + "/") for entry in EVALUATION_ONLY)


def _area(rel: str) -> str:
    parts = rel.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 and "/".join(parts[:2]) in AREAS else parts[0]


@cache
def _model_sources() -> tuple[tuple[str, ast.Module], ...]:
    sources = []
    for package in MODEL_PACKAGES:
        for path in sorted((PACKAGE / package).rglob("*.py")):
            rel = _relative(path)
            if "__pycache__" in path.parts or _excluded(rel):
                continue
            with tokenize.open(path) as stream:
                sources.append((rel, ast.parse(stream.read(), filename=str(path))))
    return tuple(sources)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _text(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _is_python(node: ast.AST) -> bool:
    """Whether a command element names a Python interpreter (``sys.executable``, ``context["python"]``)."""
    if isinstance(node, ast.Call) and _dotted(node.func) in {"str", "os.fspath"} and len(node.args) == 1:
        return _is_python(node.args[0])
    if isinstance(node, ast.Subscript):
        key = _text(node.slice)
        return key is not None and "python" in key.lower()
    text = _text(node)
    if text is not None:
        return re.fullmatch(r"(?:.*/)?python[0-9.]*", text) is not None
    dotted = _dotted(node)
    if dotted == "sys.executable":
        return True
    # The interpreter is named by the terminal component; ``python_path.parent``/``.parents`` and other
    # path navigations derive a directory, not the interpreter, so match only the last dotted name.
    return "python" in dotted.rsplit(".", 1)[-1].lower()


def _is_environment(node: ast.AST) -> bool:
    name = _dotted(node).lower()
    return name == "os.environ" or name.rsplit(".", 1)[-1] in {"env", "environ", "environment", "child_env"}


class _Scanner:
    """Collect ``(line, rule, detail)`` findings for one parsed module."""

    def __init__(self, tree: ast.Module, family_files: frozenset[str] | None = None) -> None:
        self.tree = tree
        # Tracked files of the scanned module's family; None means every script literal is suspect.
        self.family_files = family_files
        self.found: list[tuple[int, str, str]] = []
        self.documentation = {
            id(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        }
        self.constants: dict[str, str] = {}
        self.hub_names = {"torch.hub"}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                value = _text(node.value)
                if value is not None:
                    self.constants[node.targets[0].id] = value
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "torch":
                self.hub_names.update(alias.asname or alias.name for alias in node.names if alias.name == "hub")
            elif isinstance(node, ast.Import):
                self.hub_names.update(
                    alias.asname for alias in node.names if alias.name == "torch.hub" and alias.asname
                )

    def add(self, node: ast.AST, rule: str, detail: str) -> None:
        self.found.append((getattr(node, "lineno", 0), rule, detail[:120]))

    def resolve(self, node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Name):
            return self.constants.get(node.id)
        return _text(node)

    def scan(self) -> list[tuple[int, str, str]]:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                self.call(node)
            elif isinstance(node, ast.Dict):
                self.mapping(node)
            elif isinstance(node, ast.Assign | ast.AugAssign | ast.Delete):
                self.assignment(node)
            elif isinstance(node, ast.List | ast.Tuple) and isinstance(node.ctx, ast.Load):
                self.command(node)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                self.script_path(node.right)
            elif isinstance(node, ast.Name | ast.Attribute | ast.alias):
                name = (
                    node.name if isinstance(node, ast.alias) else node.id if isinstance(node, ast.Name) else node.attr
                )
                if name in _EXTERNAL_SOURCE_HELPERS:
                    self.add(node, "external-source", name)
                elif name == "prepend_import_path":
                    self.add(node, "import-path", name)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in self.documentation:
                self.constant(node, node.value)
        return self.found

    def call(self, node: ast.Call) -> None:
        name = _dotted(node.func)
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(node.func, ast.Attribute):
            target = _dotted(node.func.value)
            if leaf in _SYS_PATH_MUTATORS and target == "sys.path":
                self.add(node, "import-path", name)
            if leaf in {"load", "list", "help"} and target in self.hub_names:
                self.add(node, "remote-code", name)
            if (
                leaf in {"setdefault", "__setitem__"}
                and _is_environment(node.func.value)
                and _text(node.args[0] if node.args else None) == "PYTHONPATH"
            ):
                self.add(node, "pythonpath", f"{name}('PYTHONPATH')")
        if leaf in {"addsitedir", "addpackage"}:
            self.add(node, "import-path", name)
        if leaf == "putenv" and node.args and _text(node.args[0]) == "PYTHONPATH":
            self.add(node, "pythonpath", name)
        if leaf in _FILE_LOADERS:
            self.add(node, "file-exec", name)
        if leaf in {"exec", "eval"} and any(
            isinstance(inner, ast.Call) and _dotted(inner.func).rsplit(".", 1)[-1] in {"read", "read_text", "open"}
            for argument in node.args
            for inner in ast.walk(argument)
        ):
            self.add(node, "file-exec", f"{leaf} of file contents")
        if leaf in _REMOTE_CODE_CALLS:
            self.add(node, "remote-code", name)
        if leaf in {"torchrun_module_command", "run_torchrun_module"}:
            self.add(node, "module-launch", f"{leaf} (use package_module_command)")
        if leaf in _PATH_JOINS:
            for argument in node.args[1:] if leaf == "join" else node.args:
                self.script_path(argument)
        if leaf in _PROCESS_CALLS and node.args and isinstance(node.args[0], ast.Constant | ast.JoinedStr):
            command_text = "".join(_text(part) or " {} " for part in ast.walk(node.args[0]) if part is not node.args[0])
            if isinstance(node.args[0], ast.Constant):
                command_text = _text(node.args[0]) or ""
            if re.search(r"(?:^|[\s/])python[0-9.]*\s", command_text):
                self.add(node, "file-exec", f"shell command: {command_text[:80]}")
        if leaf in _PROCESS_CALLS and node.args:
            for part in ast.walk(node.args[0]):
                text = _text(part)
                if text is not None and _INSTALL.search(text):
                    self.add(node, "install", text)
        if leaf in _ENV_LOOKUPS and node.args:
            variable = _text(node.args[0])
            if variable and variable != "WORLDFOUNDRY_REPO_ROOT" and _SOURCE_ENV.fullmatch(variable):
                self.add(node, "external-source", variable)
        for keyword in node.keywords:
            if keyword.arg == "trust_remote_code" and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            ):
                self.add(node, "remote-code", f"trust_remote_code={ast.unparse(keyword.value)}")
            if keyword.arg == "PYTHONPATH":
                self.add(node, "pythonpath", "PYTHONPATH keyword")

    def mapping(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values, strict=True):
            if _text(key) == "trust_remote_code" and not (isinstance(value, ast.Constant) and value.value is False):
                self.add(node, "remote-code", "trust_remote_code mapping")
            if _text(key) == "PYTHONPATH":
                self.add(node, "pythonpath", "PYTHONPATH mapping")

    def assignment(self, node: ast.Assign | ast.AugAssign | ast.Delete) -> None:
        targets = node.targets if isinstance(node, ast.Assign | ast.Delete) else [node.target]
        for target in targets:
            base = target.value if isinstance(target, ast.Subscript) else target
            if _dotted(base) == "sys.path":
                self.add(node, "import-path", "sys.path assignment")
            if isinstance(target, ast.Subscript) and _text(target.slice) == "PYTHONPATH":
                self.add(node, "pythonpath", "PYTHONPATH assignment")

    def script_path(self, node: ast.AST) -> None:
        """Flag Python file names that do not exist in the module's family (upstream checkouts)."""
        text = _text(node)
        if text is None or not _PYTHON_FILE.fullmatch(text.strip()):
            return
        name = text.strip().lstrip("./")
        if self.family_files is not None and any(
            path == name or path.endswith("/" + name) for path in self.family_files
        ):
            return
        self.add(node, "file-exec", text)

    def command(self, node: ast.List | ast.Tuple) -> None:
        elements = node.elts
        values = [_text(element) for element in elements]
        if values and values[0] in _LAUNCHERS:
            self.add(node, "module-launch", values[0])
        for index, element in enumerate(elements):
            value = values[index]
            following = elements[index + 1] if index + 1 < len(elements) else None
            if value in _DISTRIBUTED_RUNNERS:
                self.add(node, "module-launch", value)
            elif value == "launch" and index and values[index - 1] in _LAUNCHERS:
                self.add(node, "module-launch", f"{values[index - 1]} launch")
            elif (value, values[index + 1] if following is not None else None) in {
                ("pip", "install"),
                ("pip3", "install"),
                ("git", "clone"),
            }:
                self.add(node, "install", f"{value} {values[index + 1]}")
            elif _is_python(element) and not isinstance(element, ast.Starred):
                previous = values[index - 1] if index else None
                if previous is not None and previous.startswith("-"):
                    # An interpreter path passed as the value of a preceding option flag
                    # (e.g. ``["--python_executable", python]``) is configured data, not a launched
                    # command head; the real executable is the first element or follows an env/launcher.
                    continue
                self.interpreter(node, elements[index + 1 :])

    def interpreter(self, node: ast.AST, arguments: list[ast.expr]) -> None:
        if not arguments:
            self.add(node, "file-exec", "python command extended elsewhere")
            return
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            option = _text(argument)
            if isinstance(argument, ast.Starred):
                self.add(node, "file-exec", f"python {ast.unparse(argument)}")
                return
            if option == "-m":
                module = self.resolve(arguments[index + 1] if index + 1 < len(arguments) else None)
                if module is None or not module.startswith("worldfoundry."):
                    self.add(node, "module-launch", f"-m {module or 'dynamic module'}")
                return
            if option == "-c":
                code = self.resolve(arguments[index + 1] if index + 1 < len(arguments) else None)
                if code is None:
                    self.add(node, "module-launch", "python -c with dynamic code")
                else:
                    for _, rule, detail in _Scanner(ast.parse(code), self.family_files).scan():
                        self.add(node, rule, f"python -c: {detail}")
                return
            if option in _PYTHON_OPTIONS_WITH_VALUE:
                index += 2
                continue
            if option is not None and option.startswith("-"):
                index += 1
                continue
            self.add(node, "file-exec", f"python {ast.unparse(argument)}")
            return

    def constant(self, node: ast.Constant, text: str) -> None:
        if "sitecustomize" in text or re.search(r"\bPYTHONPATH=", text):
            self.add(node, "pythonpath", text)
        if _THIRDPARTY.search(text) or (
            text.startswith("WORLDFOUNDRY_") and text != "WORLDFOUNDRY_REPO_ROOT" and _SOURCE_ENV.fullmatch(text)
        ):
            self.add(node, "external-source", text)


@cache
def _tracked_files() -> frozenset[str]:
    """Python files tracked under ``worldfoundry/``, relative to it."""
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z", "--", "worldfoundry"], cwd=ROOT, capture_output=True, check=True
        ).stdout.decode()
        files = [name for name in listing.split("\0") if name.endswith(".py")]
    except (OSError, subprocess.CalledProcessError):
        files = [path.relative_to(ROOT).as_posix() for path in PACKAGE.rglob("*.py")]
    return frozenset(name.removeprefix("worldfoundry/") for name in files)


@cache
def _tracked_modules() -> frozenset[str]:
    """Dotted names of every module and package directory tracked under ``worldfoundry/``."""
    modules: set[str] = set()
    for name in _tracked_files():
        parts = ["worldfoundry", *name.removesuffix(".py").split("/")]
        if parts[-1] == "__init__":
            parts.pop()
        modules.update(".".join(parts[:index]) for index in range(1, len(parts) + 1))
    return frozenset(modules)


def _family_depth(rel: str) -> int:
    """Path components of a family root: ``synthesis/<kind>/<family>``, ``base_models/<area>/<group>``, ``pipelines/<family>``."""
    return 3 if rel.split("/")[0] in {"synthesis", "base_models"} else 2


def _family_files(rel: str) -> frozenset[str]:
    """Tracked files sharing the module's family root."""
    parts = rel.split("/")
    root = "/".join(parts[: min(_family_depth(rel), len(parts) - 1)]) + "/"
    return frozenset(name for name in _tracked_files() if name.startswith(root))


@cache
def _tracked_children(files: frozenset[str]) -> dict[str, frozenset[str]]:
    """Module and package names directly inside each tracked directory under ``worldfoundry/``."""
    children: dict[str, set[str]] = {}
    for name in files:
        parts = name.removesuffix(".py").split("/")
        for index in range(1, len(parts)):
            children.setdefault("/".join(parts[:index]), set()).add(parts[index])
    return {directory: frozenset(names - {"__init__"}) for directory, names in children.items()}


def _package(rel: str) -> str:
    parts = ["worldfoundry", *Path(rel).with_suffix("").parts]
    return ".".join(parts[:-1])


def _untracked_imports(rel: str, tree: ast.Module, tracked: frozenset[str]) -> Iterator[tuple[int, str, str]]:
    """Package imports and literal targets must name modules that exist in the repository."""
    package = _package(rel)
    documentation = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [(alias.name, False) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            try:
                module = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module
            except ImportError:
                yield node.lineno, "untracked-module", "relative import beyond the package"
                continue
            targets = [(module or "", False)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in documentation:
            text = node.value
            # Runtime labels such as "worldfoundry.openvla.in_tree_runtime" are not module paths;
            # only strings naming a real top-level subpackage are import targets.
            if not (_MODULE_TARGET.fullmatch(text) and ".".join(text.split(":")[0].split(".")[:2]) in tracked):
                continue
            targets = [(text.split(":", 1)[0], ":" not in text)]
        else:
            continue
        for target, may_be_attribute in targets:
            if target != "worldfoundry" and not target.startswith("worldfoundry."):
                continue
            if target in tracked or (may_be_attribute and target.rsplit(".", 1)[0] in tracked):
                continue
            yield node.lineno, "untracked-module", target


@cache
def _module_names(rel: str) -> frozenset[str]:
    """Names a tracked module or package defines: its submodules and the names bound at its top level."""
    files = _tracked_files()
    names = set(_tracked_children(files).get(rel, ()))
    source = f"{rel}/__init__.py" if f"{rel}/__init__.py" in files else f"{rel}.py"
    if source in files and (PACKAGE / source).is_file():
        with tokenize.open(PACKAGE / source) as stream:
            body = ast.parse(stream.read()).body
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(sub.id for target in targets for sub in ast.walk(target) if isinstance(sub, ast.Name))
    return frozenset(names)


def _local_module(directory: str, parts: list[str]) -> str | None:
    """The tracked module ``parts`` names when ``directory`` is an import root, else None."""
    children = _tracked_children(_tracked_files())
    path = directory
    for part in parts:
        if part not in children.get(path, ()):
            return None
        path = f"{path}/{part}"
    return path


def _vendored_imports(rel: str, tree: ast.Module) -> Iterator[tuple[int, str, str]]:
    """Absolute imports of a package vendored inside the module's own family.

    A name that resolves to a module next to the importing file, or next to one of its ancestors up to the
    family root, loads only when that directory is on ``sys.path``. The import counts as vendored when its
    module path and the names it uses all exist in the local copy; ``import torch`` next to a vendored
    ``torch`` helper package that lacks ``nn`` still means the installed framework.
    """
    parts = Path(rel).with_suffix("").parts
    directories = ["/".join(parts[:index]) for index in range(_family_depth(rel), len(parts))]
    attributes: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            attributes.setdefault(node.value.id, set()).add(node.attr)
    for node in ast.walk(tree):
        requests: list[tuple[list[str], set[str]]] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")
                used = attributes.get(alias.asname, set()) if alias.asname else set()
                if len(module) == 1 and not alias.asname:
                    used = attributes.get(module[0], set())
                requests.append((module, used))
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            requests.append((node.module.split("."), {alias.name for alias in node.names} - {"*"}))
        elif (
            isinstance(node, ast.Call)
            and _dotted(node.func).rsplit(".", 1)[-1] in {"import_module", "__import__"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            requests.append((node.args[0].value.split("."), set()))
        for module, used in requests:
            if module[0] == "worldfoundry" or module[0] in sys.stdlib_module_names:
                continue
            for directory in directories:
                local = _local_module(directory, module)
                if local is not None and used <= _module_names(local):
                    assert isinstance(node, (ast.Import, ast.ImportFrom, ast.Call))
                    yield (
                        node.lineno,
                        "vendored-import",
                        f"{'.'.join(module)} (vendored under worldfoundry/{directory}/)",
                    )
                    break


def findings(rel: str, tree: ast.Module, tracked: frozenset[str] | None = None) -> Iterator[tuple[int, str, str]]:
    """Yield ``(line, rule, detail)`` for every external-code dependency in one module.

    Without ``tracked`` (dotted module names) the scan is purely syntactic; with it, imports must
    resolve to tracked modules, may not reach a vendored package through an absolute import, and
    Python file names may refer to tracked files of the same family.
    """
    if rel.endswith("/sitecustomize.py"):
        yield 1, "pythonpath", "sitecustomize module"
    yield from _Scanner(tree, _family_files(rel) if tracked is not None else None).scan()
    if tracked is not None:
        yield from _untracked_imports(rel, tree, tracked)
        yield from _vendored_imports(rel, tree)


def _imports(rel: str, tree: ast.Module) -> Iterator[tuple[int, str]]:
    parts = Path(rel).with_suffix("").parts
    package = ".".join(("worldfoundry", *(parts if parts[-1] == "__init__" else parts[:-1])))
    package = package.removesuffix(".__init__")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module or ""
            yield node.lineno, module
            for alias in node.names:
                yield node.lineno, f"{module}.{alias.name}"


@pytest.mark.parametrize("area", AREAS)
def test_model_runtimes_use_only_in_tree_code(area: str) -> None:
    violations = sorted(
        {
            f"worldfoundry/{rel}:{line}: [{rule}] {detail}"
            for rel, tree in _model_sources()
            if _area(rel) == area
            for line, rule, detail in findings(rel, tree, _tracked_modules())
        }
    )
    assert not violations, f"{len(violations)} external-code dependencies in {area}:\n" + "\n".join(violations)


def test_evaluation_only_base_models_have_no_model_callers() -> None:
    excluded = tuple("worldfoundry." + entry.removesuffix(".py").replace("/", ".") for entry in EVALUATION_ONLY)
    violations = []
    for package in MODEL_CALLERS:
        for path in sorted((PACKAGE / package).rglob("*.py")):
            rel = _relative(path)
            if "__pycache__" in path.parts or _excluded(rel):
                continue
            with tokenize.open(path) as stream:
                tree = ast.parse(stream.read(), filename=str(path))
            violations.extend(
                f"worldfoundry/{rel}:{line}: {target}"
                for line, target in _imports(rel, tree)
                if any(target == entry or target.startswith(entry + ".") for entry in excluded)
            )
    assert not violations, "model code imports evaluation-only base models:\n" + "\n".join(violations)


def test_evaluation_only_entries_exist() -> None:
    assert [entry for entry in EVALUATION_ONLY if not (PACKAGE / entry).exists()] == []


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ("import sys\nsys.path.insert(0, root)", "import-path"),
        ("import sys\nsys.path[:0] = [root]", "import-path"),
        ("prepend_import_path(root)", "import-path"),
        ("site.addsitedir(root)", "import-path"),
        ("env['PYTHONPATH'] = root", "pythonpath"),
        ("subprocess.run(cmd, env={**os.environ, 'PYTHONPATH': root})", "pythonpath"),
        ("os.environ.setdefault('PYTHONPATH', root)", "pythonpath"),
        ("shell = f'PYTHONPATH={root} python run.py'", "pythonpath"),
        ("AutoModel.from_pretrained(path, trust_remote_code=True)", "remote-code"),
        ("AutoModel.from_pretrained(path, **{'trust_remote_code': flag})", "remote-code"),
        ("torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline')", "remote-code"),
        ("from torch import hub\nhub.load(repo, name)", "remote-code"),
        ("runpy.run_path(str(root / 'inference.py'))", "file-exec"),
        ("subprocess.run([sys.executable, str(RUNTIME / 'infer.py')])", "file-exec"),
        ("subprocess.run([self.python_executable, '-u', script, '--seed', '0'])", "file-exec"),
        ("subprocess.run([sys.executable, *script_args])", "file-exec"),
        ("command = [context['python'], context['entrypoint'], '--seed', '0']", "file-exec"),
        ("command = [str(self.python_executable), entry]", "file-exec"),
        ("command = [sys.executable]\ncommand.append(script)", "file-exec"),
        ("subprocess.run(f'python {script} --fast', shell=True)", "file-exec"),
        ("importlib.util.spec_from_file_location('m', path)", "file-exec"),
        ("exec(open(path).read(), namespace)", "file-exec"),
        ("subprocess.run([python, '-m', 'solarwm', 'infer'])", "module-launch"),
        ("RUNNER = 'wan.generate'\nsubprocess.run([python, '-m', RUNNER])", "module-launch"),
        ("subprocess.run([sys.executable, '-c', code])", "module-launch"),
        ("subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, root)'])", "import-path"),
        ("command = ['accelerate', 'launch', script]", "module-launch"),
        ("command = [python, '-m', 'torch.distributed.run', '--nproc_per_node', '2', script]", "module-launch"),
        ("subprocess.run([sys.executable, '-m', 'pip', 'install', 'x'])", "module-launch"),
        ("os.system('git clone https://github.com/x/y')", "install"),
        ("root = repo_root / 'thirdparty' / 'lietorch'", "external-source"),
        ("os.environ.get('WORLDFOUNDRY_KAIROS_REPO_ROOT')", "external-source"),
        ("root = os.getenv('UNIWORLD_VIEW_REPO')", "external-source"),
        ("root = official_runtime_repo_path('LaST-R1')", "external-source"),
    ],
)
def test_guard_detects_external_code_forms(source: str, rule: str) -> None:
    assert rule in {found for _, found, _ in findings("synthesis/demo.py", ast.parse(source))}


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("path = ROOT / 'config' / 'detector.py'", False),
        ("entry = ROOT / 'generate.py'", True),
        ("subprocess.run([sys.executable, str(ROOT / 'config' / 'detector.py')])", True),
    ],
)
def test_guard_allows_family_file_references_but_not_their_execution(
    monkeypatch: pytest.MonkeyPatch, source: str, found: bool
) -> None:
    files = frozenset(
        {"base_models/perception/detector/config/detector.py", "synthesis/visual_generation/x/generate.py"}
    )
    monkeypatch.setattr(sys.modules[__name__], "_tracked_files", lambda: files)
    rel = "base_models/perception/detector/paths.py"
    rules = {rule for _, rule, _ in findings(rel, ast.parse(source), frozenset({"worldfoundry"}))}
    assert ("file-exec" in rules) is found


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("from .local_checkout.model import Net", True),
        ("from worldfoundry.synthesis.local_checkout import runtime", True),
        ("target = 'worldfoundry.synthesis.local_checkout.runtime:Runtime'", True),
        ("from .shipped import Net", False),
        ("from worldfoundry.synthesis import shipped", False),
        ("target = 'worldfoundry.synthesis.shipped.Runtime'", False),
        ("runtime = 'worldfoundry.shipped_runtime.in_tree'", False),
        ("import torch.nn", False),
    ],
)
def test_guard_requires_package_imports_to_resolve_to_tracked_modules(source: str, found: bool) -> None:
    tracked = frozenset({"worldfoundry", "worldfoundry.synthesis", "worldfoundry.synthesis.shipped"})
    rules = {rule for _, rule, _ in findings("synthesis/demo.py", ast.parse(source), tracked)}
    assert ("untracked-module" in rules) is found


_VENDORED_FAMILY = frozenset(
    {
        "synthesis/visual_generation/x/runtime.py",
        "synthesis/visual_generation/x/x_runtime/__init__.py",
        "synthesis/visual_generation/x/x_runtime/logging/__init__.py",
        "synthesis/visual_generation/x/x_runtime/opensora/__init__.py",
        "synthesis/visual_generation/x/x_runtime/opensora/models/net.py",
        "synthesis/visual_generation/x/x_runtime/opensora/utils.py",
        "synthesis/visual_generation/x/x_runtime/scripts/inference.py",
        "synthesis/visual_generation/x/x_runtime/activation/torch/__init__.py",
        "synthesis/visual_generation/x/x_runtime/activation/torch/act.py",
        "synthesis/visual_generation/y/runtime.py",
    }
)


@pytest.mark.parametrize(
    ("rel", "source", "found"),
    [
        ("x/x_runtime/opensora/models/net.py", "from opensora import utils", True),
        ("x/x_runtime/scripts/inference.py", "import opensora.models.net", True),
        ("x/runtime.py", "module = importlib.import_module('x_runtime.scripts.inference')", True),
        ("x/x_runtime/opensora/models/net.py", "from ..utils import load", False),
        ("x/x_runtime/opensora/models/net.py", "from worldfoundry.synthesis.visual_generation.x import runtime", False),
        ("x/x_runtime/opensora/models/net.py", "import logging\nimport torch", False),
        ("y/runtime.py", "import opensora", False),
        ("runtime_manifest.py", "import x", False),
        ("x/x_runtime/activation/torch/act.py", "import torch\nlayer = torch.nn.Linear(2, 2)", False),
        ("x/x_runtime/activation/torch/act.py", "import torch.nn.functional as F", False),
        ("x/x_runtime/activation/torch/act.py", "from torch import nn", False),
        ("x/x_runtime/activation/filter.py", "from torch import act", True),
    ],
)
def test_guard_detects_absolute_imports_of_vendored_packages(
    monkeypatch: pytest.MonkeyPatch, rel: str, source: str, found: bool
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "_tracked_files", lambda: _VENDORED_FAMILY)
    rel = f"synthesis/visual_generation/{rel}"
    rules = {rule for _, rule, _ in findings(rel, ast.parse(source), frozenset({"worldfoundry"}))}
    assert ("vendored-import" in rules) is found


@pytest.mark.parametrize(
    "source",
    [
        '"""Workers inherit PYTHONPATH; see inference.py and thirdparty/."""\nimport sys',
        "command = package_module_command('worldfoundry.synthesis.demo.worker', args, nproc_per_node=2)",
        "subprocess.run(['ffmpeg', '-i', src, '-c', 'copy', dst])",
        "command = [python, '-m', 'worldfoundry.synthesis.demo.worker']",
        "WORKER = 'worldfoundry.synthesis.demo.worker'\ncommand = [python, '-m', WORKER]",
        "subprocess.run([python, '-c', 'import importlib.util, sys; sys.exit(importlib.util.find_spec(sys.argv[1]) is None)'])",
        "model = AutoModel.from_pretrained(path, trust_remote_code=False)",
        "if root not in sys.path:\n    pass",
        "inherited = os.environ.get('PYTHONPATH', '')",
        "root = Path(os.environ['WORLDFOUNDRY_REPO_ROOT'])",
        "repo_id = os.environ.get('WORLDFOUNDRY_WAN_REPO_ID')",
        "__all__ = ['DEFAULT_WAN_REPO', 'OFFICIAL_SOURCE_REPO']",
        "raise ImportError('Install it with `pip install librosa`.')",
        "packages = ('torch', 'transformers', 'accelerate')",
        "owner = 'worldfoundry/base_models/capabilities.py'",
        "encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')",
        "python_rng, numpy_rng = random.getstate(), np.random.get_state()",
        "command = package_module_command(module, [*args, '--python_executable', python])",
        "arguments = ['--seed', '0', '--python_executable', python_executable]",
        "loader = ExtensionFileLoader(name, str(compiled_so))",
        "spec = spec_from_loader(name, ExtensionFileLoader(name, str(cached_ext)))",
    ],
)
def test_guard_accepts_in_tree_forms(source: str) -> None:
    assert list(findings("synthesis/demo.py", ast.parse(source))) == []
