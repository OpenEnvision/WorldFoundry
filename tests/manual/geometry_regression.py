"""Replay pinned local geometry/video/world inference and compare numerical outputs.

Run each revision in a separate process, using the same model environment. See
the validation guide for the case schema and commands. Nothing updates a
reference run automatically; unavailable assets and inference failures fail.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand(value, output_dir: Path):
    if isinstance(value, dict):
        return {key: expand(item, output_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, output_dir) for item in value]
    if isinstance(value, str):
        value = os.path.expandvars(value.replace("${OUTPUT_DIR}", str(output_dir)))
        if "${" in value:
            raise ValueError(f"Unset case variable: {value}")
    return value


def hash_assets(assets: dict) -> dict:
    if not assets:
        raise ValueError("Declare local input and checkpoint assets to pin the replay")
    hashes = {}
    for name, value in sorted(assets.items()):
        root = Path(value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        files = (
            [root]
            if root.is_file()
            else sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts)
        )
        if not files:
            raise ValueError(f"Empty asset: {name}")
        for path in files:
            key = name if root.is_file() else f"{name}/{path.relative_to(root).as_posix()}"
            hashes[key] = sha256(path)
    return hashes


def collect_arrays(value, arrays: dict, key: str = "result") -> None:
    # AttrDict-style configuration objects fabricate missing attributes. Handle
    # mappings first so probing for tensor methods cannot invoke or mutate them.
    if isinstance(value, dict):
        for name, item in sorted(value.items()):
            collect_arrays(item, arrays, f"{key}.{name}")
        return
    if callable(getattr(value, "detach", None)):
        value = value.detach().cpu()
        # NumPy cannot represent bfloat16. Preserve its values in float32.
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "biufc":
            arrays[key] = value
    elif isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except (ValueError, TypeError):
            array = None
        if array is not None and array.dtype.kind in "biufc" and array.size:
            arrays[key] = array
        else:
            for index, item in enumerate(value):
                collect_arrays(item, arrays, f"{key}.{index}")
    elif hasattr(value, "__dict__"):
        # Result objects, including cameras and Gaussian reconstruction outputs.
        for name, item in sorted(vars(value).items()):
            if not name.startswith("_"):
                collect_arrays(item, arrays, f"{key}.{name}")


def validate_arrays(arrays: dict, required: list[str]) -> None:
    if not arrays:
        raise ValueError("Inference returned no numerical outputs")
    for key in required:
        if not any(name == key or name.startswith(key + ".") for name in arrays):
            raise ValueError(f"Missing required output: {key}")
    for key, value in arrays.items():
        if not value.size or not np.isfinite(value).all():
            raise ValueError(f"Empty or non-finite output: {key}")


def validate_contracts(arrays: dict, contracts: dict, comparisons: list[dict]) -> None:
    """Check geometry and request isolation independently of the reference run."""
    for key, contract in contracts.items():
        if key not in arrays:
            raise ValueError(f"Missing contracted output: {key}")
        value = arrays[key]
        if set(contract) - {"shape", "dtype", "min", "max", "values"}:
            raise ValueError(f"Unknown array contract option: {key}")
        if "shape" in contract and list(value.shape) != contract["shape"]:
            raise ValueError(f"Contracted output shape changed: {key}: {value.shape}")
        if "dtype" in contract and str(value.dtype) != contract["dtype"]:
            raise ValueError(f"Contracted output dtype changed: {key}: {value.dtype}")
        for bound in ("min", "max"):
            if bound in contract and not np.isfinite(contract[bound]):
                raise ValueError(f"Non-finite contract bound: {key}")
        if "min" in contract and np.any(value < contract["min"]):
            raise ValueError(f"Output below contracted range: {key}")
        if "max" in contract and np.any(value > contract["max"]):
            raise ValueError(f"Output above contracted range: {key}")
        if "values" in contract and not np.array_equal(value, np.asarray(contract["values"])):
            raise ValueError(f"Contracted output values changed: {key}")
    for comparison in comparisons:
        if set(comparison) != {"left", "right", "relation"}:
            raise ValueError("Output comparisons need left/right/relation")
        left, right = comparison["left"], comparison["right"]
        if left not in arrays or right not in arrays:
            raise ValueError(f"Missing comparison outputs: {left}, {right}")
        a, b = arrays[left], arrays[right]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Comparison shape or dtype differs: {left}, {right}")
        relation = comparison["relation"]
        if relation not in {"equal", "different"}:
            raise ValueError(f"Unsupported output relation: {relation}")
        if np.array_equal(a, b) != (relation == "equal"):
            raise ValueError(f"Output relation {relation} failed: {left}, {right}")


def run_sequence(pipeline, sequence: list[dict], arrays: dict) -> list[dict]:
    """Exercise one resident model, including failures, reset and A/B/A requests."""
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("A request sequence must be nonempty")
    names, events = set(), []
    for step in sequence:
        name = step.get("name", "")
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", name) or name in names:
            raise ValueError(f"Invalid or duplicate sequence name: {name!r}")
        names.add(name)
        if set(step) - {"name", "method", "call", "outputs", "expect_error"}:
            raise ValueError(f"Unknown request sequence option: {name}")
        expected = step.get("expect_error")
        if expected is not None and (
            not isinstance(expected, dict)
            or set(expected) != {"type", "match"}
            or not isinstance(expected["match"], str)
            or not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+", expected["type"])
        ):
            raise ValueError(f"Expected errors need qualified type and message pattern: {name}")
        method_name = step.get("method", "__call__")
        # Resolve method errors outside the expected-inference-error boundary.
        method = getattr(pipeline, method_name)
        try:
            result = method(**step.get("call", {}))
        except Exception as exc:
            actual = f"{type(exc).__module__}.{type(exc).__qualname__}"
            if expected is None or actual != expected["type"] or not re.search(expected["match"], str(exc)):
                raise
            events.append({"name": name, "status": "expected_error", "type": actual})
            continue
        if expected is not None:
            raise ValueError(f"Expected inference failure did not occur: {name}")
        step_arrays = {}
        outputs = step.get("outputs")
        if outputs is None:
            collect_arrays(result, step_arrays, name)
        else:
            if not isinstance(outputs, list) or len(set(outputs)) != len(outputs):
                raise ValueError(f"Invalid output selection: {name}")
            for path in outputs:
                selected = result
                for attribute in path.split("."):
                    selected = selected[attribute] if isinstance(selected, dict) else getattr(selected, attribute)
                collect_arrays(selected, step_arrays, name + "." + path)
        # Some resident runtimes return views of reusable CPU buffers. Preserve
        # each response now so later requests cannot rewrite the earlier evidence.
        arrays.update({key: value.copy() for key, value in step_arrays.items()})
        events.append({"name": name, "status": "passed"})
    return events


def exported_arrays(root: Path, files: list[str]) -> dict:
    arrays = {}
    for file in files:
        path = Path(file)
        key = "export." + path.relative_to(root).as_posix()
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty exported artifact: {path}")
        if path.suffix == ".npy":
            collect_arrays(np.load(path, allow_pickle=False), arrays, key)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                for name in data.files:
                    collect_arrays(data[name], arrays, key + "." + name)
        elif path.suffix == ".ply":
            from plyfile import PlyData

            for element in PlyData.read(path).elements:
                for name in element.data.dtype.names:
                    collect_arrays(element.data[name], arrays, key + "." + element.name + "." + name)
        elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            from PIL import Image

            with Image.open(path) as image:
                collect_arrays(np.asarray(image), arrays, key)
        elif path.suffix == ".json":
            collect_arrays(json.loads(path.read_text()), arrays, key)
        elif path.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}:
            import av

            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                rate = stream.average_rate
                if rate is None or rate <= 0:
                    raise ValueError(f"Video has no positive frame rate: {path}")
                frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
                if not frames:
                    raise ValueError(f"Video has no decodable frames: {path}")
                collect_arrays(np.stack(frames), arrays, key + ".frames")
                collect_arrays(np.array([rate.numerator, rate.denominator], dtype=np.int64), arrays, key + ".fps")
    return arrays


def runtime_metadata(torch) -> dict:
    packages = {
        dist.metadata["Name"].lower(): dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"] and dist.metadata["Name"].lower() != "worldfoundry"
    }
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name()
    return {
        "python": platform.python_version(),
        "packages": dict(sorted(packages.items())),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }


def run_case(case_file: Path, source_root: Path, output_dir: Path, case_id: str | None = None) -> dict:
    # Refuse to overwrite evidence, particularly an approved reference run.
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    case = json.loads(case_file.read_text())
    if case_id is not None:
        case = case[case_id]
    manifest = {"status": "failed", "case": case, "source_root": str(source_root)}
    try:
        sys.path.insert(0, str(source_root))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        deterministic = case.get("deterministic", False)
        if not isinstance(deterministic, bool):
            raise ValueError("deterministic must be a boolean")
        if deterministic:
            # cuBLAS needs this before CUDA initialization for deterministic GEMM.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        seed = int(case.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        resolved = expand(case, output_dir)
        manifest["assets"] = hash_assets(resolved["assets"])
        module_name, class_name = resolved["target"].split(":")
        module = importlib.import_module(module_name)
        if not Path(module.__file__).resolve().is_relative_to(source_root):
            raise RuntimeError("The requested source tree was not imported")
        pipeline = getattr(module, class_name).from_pretrained(**resolved["load"])
        arrays = {}
        for model_path in resolved.get("capture_modules", []):
            model = pipeline
            for attribute in model_path.split("."):
                model = getattr(model, attribute)

            def capture(_module, _args, prediction, key="forward." + model_path):
                collect_arrays(prediction, arrays, key)

            model.register_forward_hook(capture)
        if "sequence" in resolved:
            if {"call", "method", "extra_calls", "save"} & set(resolved):
                raise ValueError("A request sequence cannot also declare call/method/extra_calls/save")
            manifest["sequence_events"] = run_sequence(pipeline, resolved["sequence"], arrays)
        else:
            method = getattr(pipeline, resolved.get("method", "__call__"))
            result = method(**resolved["call"])
            collect_arrays(result, arrays)
            for name, extra in resolved.get("extra_calls", {}).items():
                collect_arrays(getattr(pipeline, extra["method"])(**extra["call"]), arrays, name)
        if resolved.get("save", False):
            files = result.save(str(output_dir / "export"))
            if not files or any(not Path(path).is_file() or not Path(path).stat().st_size for path in files):
                raise ValueError("Inference exported missing or empty artifacts")
            manifest["exported_files"] = files
        for pattern in resolved.get("export_artifacts", []):
            if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                raise ValueError(f"Export pattern escapes evidence directory: {pattern}")
            files = sorted(output_dir.glob(pattern))
            if not files or any(not p.is_file() or p.is_symlink() for p in files):
                raise ValueError(f"Missing exported artifacts: {pattern}")
            manifest.setdefault("exported_files", []).extend(str(p) for p in files)
        # Some pipelines return a path to raw predictions instead of arrays.
        for pattern in resolved.get("output_arrays", []):
            matches = sorted(output_dir.glob(pattern))
            if not matches:
                raise ValueError(f"No numerical artifacts match: {pattern}")
            for path in matches:
                key = "artifact." + path.relative_to(output_dir).as_posix()
                if path.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as data:
                        for name in data.files:
                            collect_arrays(data[name], arrays, key + "." + name)
                elif path.suffix == ".npy":
                    collect_arrays(np.load(path, allow_pickle=False), arrays, key)
                else:
                    raise ValueError(f"Unsupported numerical artifact: {path}")
        validate_arrays(arrays, resolved["required_outputs"])
        exports = exported_arrays(output_dir, manifest.get("exported_files", []))
        validate_arrays({**arrays, **exports}, resolved["required_outputs"])
        validate_contracts({**arrays, **exports}, resolved.get("array_contracts", {}), resolved.get("comparisons", []))
        np.savez_compressed(output_dir / "arrays.npz", **arrays)
        sources = {}
        for imported in list(sys.modules.values()):
            path = getattr(imported, "__file__", None)
            if path and Path(path).is_relative_to(source_root / "worldfoundry"):
                path = Path(path).resolve()
                if path.is_file():
                    sources[path.relative_to(source_root).as_posix()] = sha256(path)
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        manifest.update(
            status="passed",
            source_revision=revision.stdout.strip() if revision.returncode == 0 else None,
            source_hashes=sources,
            runtime=runtime_metadata(torch),
            arrays={key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
            arrays_sha256=sha256(output_dir / "arrays.npz"),
        )
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
    finally:
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def compare_runs(reference: Path, candidate: Path, *, atol: float = 0, rtol: float = 0) -> dict:
    reference, candidate = reference.resolve(), candidate.resolve()
    if not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0:
        raise ValueError("Tolerances must be finite and nonnegative")
    meta_a = json.loads((reference / "manifest.json").read_text())
    meta_b = json.loads((candidate / "manifest.json").read_text())
    for meta, root in ((meta_a, reference), (meta_b, candidate)):
        if meta["status"] != "passed":
            raise ValueError(f"Inference did not pass: {root}")
        if meta["arrays_sha256"] != sha256(root / "arrays.npz"):
            raise ValueError(f"Numerical evidence was modified: {root}")
    for key in ("case", "assets", "runtime"):
        if meta_a[key] != meta_b[key]:
            raise ValueError(f"Replay conditions differ: {key}")
    exports_a = meta_a.get("exported_files", [])
    exports_b = meta_b.get("exported_files", [])
    if sorted(Path(p).relative_to(reference).as_posix() for p in exports_a) != sorted(
        Path(p).relative_to(candidate).as_posix() for p in exports_b
    ):
        raise ValueError("Exported artifact names differ")
    report = {"status": "passed", "atol": atol, "rtol": rtol, "outputs": {}}
    with (
        np.load(reference / "arrays.npz", allow_pickle=False) as left,
        np.load(candidate / "arrays.npz", allow_pickle=False) as right,
    ):
        if set(left.files) != set(right.files) or not left.files:
            raise ValueError("Numerical output keys differ or are empty")
        arrays_a = {**dict(left), **exported_arrays(reference, exports_a)}
        arrays_b = {**dict(right), **exported_arrays(candidate, exports_b)}
        if arrays_a.keys() != arrays_b.keys():
            raise ValueError("Exported numerical fields differ")
        for key in sorted(arrays_a):
            a, b = arrays_a[key], arrays_b[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"Output shape or dtype changed: {key}")
            validate_arrays({key: a}, [])
            validate_arrays({key: b}, [])
            # Masks, images, counts and IDs must remain exact even with float tolerances.
            exact = np.array_equal(a, b)
            passed = exact if a.dtype.kind in "biu" else bool(np.allclose(b, a, atol=atol, rtol=rtol, equal_nan=False))
            dtype = np.complex128 if a.dtype.kind == "c" else np.float64
            delta = np.abs(b.astype(dtype) - a.astype(dtype))
            report["outputs"][key] = {
                "passed": passed,
                "exact": exact,
                "shape": list(a.shape),
                "max_abs_error": float(delta.max()),
            }
            if not passed:
                report["status"] = "failed"
    return report


def audit_matrix(matrix: Path, reference: Path, candidate: Path, source_root: Path, case_ids: list[str]) -> dict:
    cases = json.loads(matrix.read_text())
    selected = case_ids or list(cases)
    report = {"status": "passed", "cases": {}}
    source_root = source_root.resolve()
    hashes = {}
    for name in selected:
        try:
            if name not in cases:
                raise ValueError(f"Unknown matrix case: {name}")
            manifest = json.loads((candidate / name / "manifest.json").read_text())
            if manifest.get("case") != cases[name]:
                raise ValueError("Run does not match the requested matrix case")
            if not manifest.get("source_hashes"):
                raise ValueError("Run has no imported-source evidence")
            for path, expected in manifest["source_hashes"].items():
                if Path(path).is_absolute() or ".." in Path(path).parts:
                    raise ValueError(f"Invalid imported-source path: {path}")
                if path not in hashes:
                    hashes[path] = sha256(source_root / path)
                if hashes[path] != expected:
                    raise ValueError(f"Inference evidence is stale: {path}")
            result = compare_runs(reference / name, candidate / name)
        except (ValueError, OSError, KeyError, ImportError) as exc:
            result = {"status": "failed", "error": str(exc)}
        report["cases"][name] = result
        if result["status"] != "passed":
            report["status"] = "failed"
    if not selected:
        raise ValueError("The regression matrix is empty")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--case", type=Path, required=True)
    run.add_argument("--case-id", help="Select a case from the published case matrix")
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--atol", type=float, default=0)
    compare.add_argument("--rtol", type=float, default=0)
    compare.add_argument("--report", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--matrix", type=Path, required=True)
    audit.add_argument("--reference", type=Path, required=True)
    audit.add_argument("--candidate", type=Path, required=True)
    audit.add_argument("--source-root", type=Path, required=True)
    audit.add_argument("--case-id", action="append", default=[])
    audit.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        report = run_case(args.case.resolve(), args.source_root.resolve(), args.output_dir.resolve(), args.case_id)
        print(
            json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "error") if key in report}),
            flush=True,
        )
    else:
        try:
            if args.command == "audit":
                report = audit_matrix(args.matrix, args.reference, args.candidate, args.source_root, args.case_id)
            else:
                report = compare_runs(args.reference, args.candidate, atol=args.atol, rtol=args.rtol)
        except (ValueError, OSError, KeyError, ImportError) as exc:
            report = {"status": "failed", "error": str(exc)}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
