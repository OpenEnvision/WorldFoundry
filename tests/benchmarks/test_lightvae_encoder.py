"""Encoder diagnostics measure executed candidates and retain quality results."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks.inference import lightvae_encoder as diagnostic


class _TeacherVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.model = torch.nn.Module()
        self.model.encoder = torch.nn.Identity()
        self.decode_autocast_dtype = None
        self.eval()


class _StudentVAE(_TeacherVAE):
    pass


class _TeacherCodec:
    def __init__(self, *, factor=1.0, count_encode=True):
        self.vae = _StudentVAE() if isinstance(self, _StudentCodec) else _TeacherVAE()
        self.device = torch.device("cpu")
        self.tiled = False
        self.temporal_chunk_size = 0
        self.parallel_degree = 1
        self.offload_effective = "resident"
        self.factor = factor
        self.count_encode = count_encode
        self.encodes = self.decodes = 0
        self.encode_callback = None

    @property
    def dtype(self):
        return self.vae.weight.dtype

    def encode(self, pixels):
        for video in pixels:
            for index in range(1 + (pixels.shape[2] - 1) // 4):
                start, end = (0, 1) if index == 0 else (1 + 4 * (index - 1), 1 + 4 * index)
                self.vae.model.encoder(video.unsqueeze(0)[:, :, start:end])
        self.encodes += int(self.count_encode)
        if self.encode_callback:
            self.encode_callback()
        shape = (pixels.shape[0], 16, 1 + (pixels.shape[2] - 1) // 4, pixels.shape[3] // 8, pixels.shape[4] // 8)
        return pixels.flatten(1).mean(dim=1).reshape(-1, 1, 1, 1, 1).expand(shape).clone() * self.factor

    def decode(self, latents):
        self.decodes += 1
        rgb = torch.cat((latents[:, :3, :1], latents[:, :3, 1:].repeat_interleave(4, dim=2)), dim=2)
        return rgb.repeat_interleave(8, dim=3).repeat_interleave(8, dim=4).clamp(-1, 1)

    def runtime_optimization_report(self):
        student = isinstance(self, _StudentCodec)
        runtime = {
            "lightvae_encode_calls": self.encodes if student else 0,
            "lifetime": {"dense_decode_calls": self.decodes, "lightvae_decode_calls": self.decodes if student else 0},
        }
        return {
            "effective": {"vae_variant": "lightvae-wan21"} if student else {},
            "runtime": runtime,
            "fallbacks": [],
            "quality_tier": "algorithmically-approximate" if student else "exact",
        }


class _StudentCodec(_TeacherCodec):
    pass


def _payload():
    return {"sample": torch.full((1, 3, 5, 8, 8), 0.5), "latents": torch.full((1, 16, 2, 1, 1), 0.5)}


@pytest.fixture
def codecs(monkeypatch):
    monkeypatch.setattr(diagnostic, "WanVideoDecoder", _TeacherCodec)
    monkeypatch.setattr(diagnostic, "Wan21LightVAECodec", _StudentCodec)
    monkeypatch.setattr(diagnostic, "WanVideoVAE", _TeacherVAE)
    monkeypatch.setattr(diagnostic, "Wan21LightVAE", _StudentVAE)
    return _TeacherCodec(), _StudentCodec()


def test_reference_accepts_real_rgb_geometry(tmp_path):
    path = tmp_path / "reference.pt"
    torch.save(_payload(), path)
    assert torch.equal(diagnostic._load_reference(path)["sample"], _payload()["sample"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(extra=torch.ones(1)),
        lambda payload: payload.update(sample="RGB"),
        lambda payload: payload.update(sample=payload["sample"].to(torch.bfloat16)),
        lambda payload: payload.update(sample=payload["sample"][:, :, :4]),
        lambda payload: payload.update(sample=payload["sample"][:, :, :, :7]),
        lambda payload: payload.update(latents=payload["latents"][:, :15]),
        lambda payload: payload.update(latents=payload["latents"][:, :, :1]),
        lambda payload: payload["sample"].fill_(float("nan")),
        lambda payload: payload["latents"].fill_(float("inf")),
        lambda payload: payload["sample"].fill_(1.1),
    ],
)
def test_invalid_reference_is_rejected_before_gpu_work(tmp_path, mutation):
    path = tmp_path / "reference.pt"
    payload = _payload()
    mutation(payload)
    torch.save(payload, path)
    with pytest.raises(ValueError):
        diagnostic._load_reference(path)


@pytest.mark.parametrize("student", [False, True])
def test_codec_admission_requires_fp32_untiled_native_eval(codecs, student):
    codec = codecs[int(student)]
    diagnostic._validate_codec(codec, student=student)
    codec.tiled = True
    with pytest.raises(ValueError, match="untiled"):
        diagnostic._validate_codec(codec, student=student)
    codec.tiled = False
    codec.vae.train()
    with pytest.raises(ValueError, match="eval"):
        diagnostic._validate_codec(codec, student=student)
    codec.vae.eval().to(torch.bfloat16)
    with pytest.raises(ValueError, match="FP32"):
        diagnostic._validate_codec(codec, student=student)


def test_evaluation_uses_common_teacher_decoder_and_counts_native_encoders(codecs):
    teacher, student = codecs
    row = {}
    pixels, reference, candidate = diagnostic._evaluate_case(teacher, student, _payload(), row)
    assert row["saved_rgb_provenance"]["passed"]
    assert row["quality"]["passed"] and row["execution_gate"]["passed"]
    assert row["teacher_encoder_execution"]["encoder_module_calls"] == 2
    assert row["student_encoder_execution"]["lightvae_encode_call_delta"] == 1
    assert teacher.decodes == 3 and student.decodes == 0
    assert torch.equal(reference.sample, candidate.sample) and torch.equal(pixels, _payload()["sample"])
    json.dumps(row, allow_nan=False)


def test_mismatched_saved_rgb_provenance_rejected_before_any_encode(codecs):
    teacher, student = codecs
    payload = _payload()
    payload["sample"].fill_(0.4)
    with pytest.raises(AssertionError):
        diagnostic._evaluate_case(teacher, student, payload, {})
    assert teacher.encodes == student.encodes == 0


def test_student_error_is_measured_in_both_latents_and_common_teacher_rgb(codecs):
    teacher, student = codecs
    student.factor = 1.1
    row = {}
    diagnostic._evaluate_case(teacher, student, _payload(), row)
    assert row["quality"]["latent_relative_l2"] > 0.09
    assert row["quality"]["video_psnr_db"] < diagnostic.BUDGET["video_psnr_min_db"]
    assert not row["quality"]["passed"]
    assert row["execution_gate"]["passed"]


def test_pilot_hook_is_removed_when_encoder_raises(codecs):
    teacher, _ = codecs

    def failed(pixels):
        raise RuntimeError("encoder failed")

    teacher.encode = failed
    with pytest.raises(RuntimeError, match="encoder failed"):
        diagnostic._encode_pilot(teacher, _payload()["sample"])
    assert not teacher.vae.model.encoder._forward_hooks


def test_unadvanced_student_encoder_counter_rejects_execution(codecs):
    teacher, student = codecs
    student.count_encode = False
    row = {}
    diagnostic._evaluate_case(teacher, student, _payload(), row)
    assert row["quality"]["passed"]
    assert not row["execution_gate"]["passed"]
    assert not row["execution_gate"]["checks"]["student_encode_count"]


@pytest.mark.parametrize("failed_gate", ["quality", "execution_gate"])
def test_rejected_case_cannot_reach_timing_or_device_admission(monkeypatch, failed_gate):
    row = {"quality": {"finite": True, "passed": True}, "execution_gate": {"passed": True}}
    row[failed_gate]["finite" if failed_gate == "quality" else "passed"] = False

    def unexpected():
        raise AssertionError("rejected case reached device admission")

    monkeypatch.setattr(diagnostic, "cuda_device_admission", unexpected)
    diagnostic._time_case(None, None, None, None, None, row, rounds=3, quality_only=False)
    assert row["status"].startswith("rejected_")
    assert "speedup_median" not in row


@pytest.mark.parametrize(("quality_only", "clean"), [(True, True), (False, False)])
def test_quality_only_or_shared_device_cannot_publish_timings(monkeypatch, quality_only, clean):
    monkeypatch.setattr(diagnostic, "cuda_device_admission", lambda: {"timing_qualified": clean})
    row = {"quality": {"finite": True, "passed": True}, "execution_gate": {"passed": True}}
    diagnostic._time_case(None, None, None, None, None, row, rounds=3, quality_only=quality_only)
    assert row["status"] == "not_timed"
    assert "teacher_wall_s" not in row and "speedup_median" not in row


@pytest.mark.parametrize("failure", [None, "device", "execution", "quality"])
def test_three_paired_rounds_alternate_order_and_reject_contamination(monkeypatch, failure):
    admissions, order = [], []

    def admission():
        admissions.append(None)
        return {"timing_qualified": not (failure == "device" and len(admissions) == 4)}

    def measure(codec, pixels, pilot_latents, *, student):
        order.append(student)
        return (1.0 if student else 2.0), {"passed": not (failure == "execution" and len(order) == 4)}

    monkeypatch.setattr(diagnostic, "cuda_device_admission", admission)
    monkeypatch.setattr(diagnostic, "_measure_encode", measure)
    row = {"quality": {"finite": True, "passed": True}, "execution_gate": {"passed": True}}
    row["quality"]["passed"] = failure != "quality"
    output = SimpleNamespace(latents=torch.ones(1))
    diagnostic._time_case(None, None, None, output, output, row, rounds=3, quality_only=False)
    assert order == [False, True, True, False, False, True]
    assert len(admissions) == 13
    assert len(row["teacher_wall_s"]) == len(row["student_wall_s"]) == 3
    if failure in ("device", "execution"):
        assert row["status"].startswith("rejected_") and "speedup_median" not in row
    else:
        assert row["status"] == ("measured_quality_tradeoff" if failure == "quality" else "qualified_for_test_case")
        assert row["speedup_median"] == 2.0 and row["speedup_ci95"] == [2.0, 2.0]


def test_less_than_three_pairs_rejected(monkeypatch):
    monkeypatch.setattr(diagnostic, "cuda_device_admission", lambda: {"timing_qualified": True})
    row = {"quality": {"finite": True, "passed": True}, "execution_gate": {"passed": True}}
    with pytest.raises(ValueError, match="three paired"):
        diagnostic._time_case(None, None, None, None, None, row, rounds=2, quality_only=False)


@pytest.mark.parametrize("changed", ["output", "counter"])
def test_timed_encoding_must_match_pilot_and_advance_student_counter(codecs, monkeypatch, changed):
    _, student = codecs
    pixels = _payload()["sample"]
    pilot, _ = diagnostic._encode_pilot(student, pixels)
    if changed == "output":
        student.factor = 1.1
    else:
        student.count_encode = False
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    clocks = iter((10.0, 11.0))
    monkeypatch.setattr(diagnostic.time, "perf_counter", lambda: next(clocks))
    elapsed, receipt = diagnostic._measure_encode(student, pixels, pilot, student=True)
    assert elapsed == 1.0 and not receipt["passed"]


def _main_args(tmp_path):
    teacher, student, reference = (tmp_path / name for name in ("teacher.pth", "student.pth", "reference.pt"))
    teacher.write_bytes(b"teacher checkpoint")
    student.write_bytes(b"student checkpoint")
    torch.save(_payload(), reference)
    return [
        "--teacher",
        str(teacher),
        "--student",
        str(student),
        "--reference",
        str(reference),
        "--out",
        str(tmp_path / "results"),
        "--quality-only",
        "--rounds",
        "1",
    ]


def _patch_main(monkeypatch, codecs):
    teacher, student = codecs
    monkeypatch.setattr(
        diagnostic, "load_wan_video_codec", lambda path, **kwargs: student if kwargs.get("variant") else teacher
    )
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(diagnostic, "capture_runtime_fingerprint", lambda **kwargs: SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(diagnostic, "cuda_device_admission", lambda: {"timing_qualified": True})


def test_main_saves_outputs_and_input_metadata(tmp_path, monkeypatch, codecs):
    _patch_main(monkeypatch, codecs)
    diagnostic.main(_main_args(tmp_path))
    record = json.loads((tmp_path / "results/results.json").read_text())
    assert record["status"] == "completed_diagnostic"
    assert record["inputs_unchanged"]
    assert record["files"] == record["files_after"]
    assert all("mtime_ns" in item and "sha256" not in item for item in record["files"])
    assert record["cases"][0]["status"] == "not_timed"
    assert (tmp_path / "results/student_encoded_0.pt").is_file()
    assert (tmp_path / "results/teacher_encoded_0.pt").is_file()


def test_main_records_failed_reference_comparison(tmp_path, monkeypatch, codecs):
    _patch_main(monkeypatch, codecs)
    args = _main_args(tmp_path)
    payload = _payload()
    payload["sample"].fill_(0.4)
    torch.save(payload, tmp_path / "reference.pt")
    with pytest.raises(AssertionError):
        diagnostic.main(args)
    record = json.loads((tmp_path / "results/results.json").read_text())
    assert record["status"] == "failed" and record["inputs_unchanged"]
    assert "quality" not in record["cases"][0]
    assert not list((tmp_path / "results").glob("*_encoded_*.pt"))


def test_changed_checkpoint_invalidates_otherwise_passing_diagnostic(tmp_path, monkeypatch, codecs):
    _patch_main(monkeypatch, codecs)
    args = _main_args(tmp_path)
    codecs[1].encode_callback = lambda: (tmp_path / "student.pth").write_bytes(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="inputs changed"):
        diagnostic.main(args)
    record = json.loads((tmp_path / "results/results.json").read_text())
    assert record["status"] == "failed" and not record["inputs_unchanged"]
    assert "manifest_verification_error" in record


def test_existing_output_directory_rejected_before_gpu_work(tmp_path, monkeypatch):
    args = _main_args(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results/kept").write_text("existing artifact")
    with pytest.raises(FileExistsError):
        diagnostic.main(args)
    assert (tmp_path / "results/kept").read_text() == "existing artifact"
    assert not (tmp_path / "results/results.json").exists()


def test_unrelated_source_edit_does_not_discard_encoder_results(tmp_path, monkeypatch, codecs):
    _patch_main(monkeypatch, codecs)
    unrelated = tmp_path / "other_model.py"
    unrelated.write_text("original source")
    codecs[1].encode_callback = lambda: unrelated.write_text("updated source")
    diagnostic.main(_main_args(tmp_path))
    record = json.loads((tmp_path / "results/results.json").read_text())
    assert record["status"] == "completed_diagnostic"
    assert record["inputs_unchanged"]
    assert record["cases"][0]["quality"]["passed"]


def test_cli_rejects_duplicate_references_and_short_performance_rounds(tmp_path):
    args = _main_args(tmp_path)
    with pytest.raises(SystemExit):
        diagnostic._parse_args(args + ["--reference", str(tmp_path / "reference.pt")])
    args.remove("--quality-only")
    with pytest.raises(SystemExit):
        diagnostic._parse_args(args)
