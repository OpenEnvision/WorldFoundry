"""JEDi cache, process-group ownership, and feature-report regressions."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

from worldfoundry.evaluation.tasks.metrics.jedi.JEDi import JEDiMetric
from worldfoundry.evaluation.tasks.metrics.jedi.jedi_runtime import (
    JEDiScorerConfig,
    run_jedi_scorer,
    score_from_feature_payload,
)
from worldfoundry.evaluation.tasks.metrics.jedi.wrapper import bundled_config_path, compute_jedi_from_features


class TinyExtractor:
    def __init__(self, **kwargs):
        self.encoder = torch.nn.Linear(3, 3)

    def get_feats(self, videos):
        return videos[:, 0, :, 0, 0].numpy()


class TinyEncoder(torch.nn.Module):
    embed_dim = 3
    num_heads = 1

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, videos):
        return videos.mean(dim=(2, 3, 4)).unsqueeze(1) * self.weight


class TinyAggregation(torch.nn.Module):
    def __init__(self, encoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.num_heads = encoder.num_heads

    def forward(self, clips):
        return [self.encoder(clips[0][0])]


class TinyProbe(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.pooler = torch.nn.Linear(3, 3, bias=False)


class ChannelNormalize:
    """The C,T,H,W broadcast contract of vjepa 0.1.2's video Normalize."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)[:, None, None, None]
        self.std = torch.tensor(std)[:, None, None, None]

    def __call__(self, clip):
        return (clip - self.mean) / self.std


class JediRuntimeTests(unittest.TestCase):
    def test_mock_seed_reproduces_features_and_score_across_processes(self):
        program = """
import json
from worldfoundry.evaluation.tasks.metrics.jedi.wrapper import deterministic_feature_matrix, compute_mock_jedi
features = deterministic_feature_matrix(seed='jedi-train', num_samples=3, feature_dim=4)
score = compute_mock_jedi(num_samples=8, feature_dim=4)
print(json.dumps({'features': features.tolist(), 'score': score}))
"""
        outputs = [
            subprocess.check_output([sys.executable, "-B", "-c", program], text=True, timeout=30) for _ in range(2)
        ]
        self.assertEqual(outputs[0], outputs[1])

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="jedi-tests-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def start_caller_group(self):
        dist.init_process_group("gloo", init_method=(self.root / "group").as_uri(), rank=0, world_size=1)
        self.addCleanup(lambda: dist.destroy_process_group() if dist.is_initialized() else None)

    def loader(self, features):
        return [
            (torch.from_numpy(batch[:, None, :, None, None]).float(), None) for batch in np.array_split(features, 2)
        ]

    def test_cached_and_extracted_features_apply_the_same_sample_cap(self):
        train, test = np.arange(18).reshape(6, 3), np.arange(24).reshape(8, 3)
        np.save(self.root / "train.npy", train)
        np.save(self.root / "test.npy", test)
        cached = JEDiMetric(feature_path=str(self.root)).load_features(num_samples=2)
        extractor_module = types.ModuleType("worldfoundry.evaluation.tasks.metrics.jedi.V_JEPA")
        extractor_module.VJEPA = TinyExtractor
        with patch.dict(sys.modules, {extractor_module.__name__: extractor_module}):
            fresh = JEDiMetric(feature_path=str(self.root / "fresh")).load_features(
                self.loader(train), self.loader(test), num_samples=2
            )
        for actual_cached, actual_fresh, original in zip(cached, fresh, (train, test)):
            np.testing.assert_array_equal(actual_cached, original[:2])
            np.testing.assert_array_equal(actual_cached, actual_fresh)
        self.assertEqual(np.load(self.root / "train.npy").shape, (6, 3))

    def test_feature_cleanup_preserves_caller_group_and_releases_extractor(self):
        self.start_caller_group()
        extractor_refs = []

        def create_extractor(**kwargs):
            extractor = TinyExtractor()
            extractor_refs.append(weakref.ref(extractor))
            return extractor

        extractor_module = types.ModuleType("worldfoundry.evaluation.tasks.metrics.jedi.V_JEPA")
        extractor_module.VJEPA = create_extractor
        metric = JEDiMetric(feature_path=str(self.root))
        features = np.ones((4, 3))
        with patch.dict(sys.modules, {extractor_module.__name__: extractor_module}):
            metric.load_features(self.loader(features), self.loader(features), num_samples=2)
        self.assertTrue(dist.is_initialized())
        value = torch.ones(1)
        dist.all_reduce(value)
        self.assertEqual(value.item(), 1)
        self.assertIsNone(extractor_refs[0]())

    def vjepa_module(self):
        modules = {
            name: types.ModuleType(name)
            for name in (
                "vjepa",
                "vjepa.models",
                "vjepa.models.attentive_pooler",
                "vjepa.utils",
                "vjepa.utils.distributed",
                "worldfoundry.evaluation.tasks.metrics.jedi.V_JEPA_utils",
            )
        }
        for size in ("tiny", "small", "base", "large", "huge", "giant", "gigantic"):
            setattr(modules["vjepa.models"], f"vit_{size}", TinyEncoder)
        modules["vjepa.models.attentive_pooler"].AttentiveClassifier = TinyProbe
        modules["vjepa.utils.distributed"].init_distributed = lambda: (1, 0)
        helpers = modules["worldfoundry.evaluation.tasks.metrics.jedi.V_JEPA_utils"]
        helpers.FrameAggregation = helpers.ClipAggregation = TinyAggregation
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        return importlib.import_module("worldfoundry.evaluation.tasks.metrics.jedi.V_JEPA")

    def test_vjepa_probe_loading_and_inference_do_not_manage_process_groups(self):
        # Substitute the optional network package, exercising real checkpoint loading.
        module = self.vjepa_module()
        probe_weights = {"module.pooler.weight": 2 * torch.eye(3)}
        torch.save({"epoch": 1, "classifier": probe_weights}, self.root / "ssv2-probe.pth.tar")
        self.start_caller_group()
        with (
            patch.object(module, "init_model", return_value=TinyEncoder()),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            encoder, classifier = module.get_default_vjepa(
                config_fname=str(bundled_config_path()), model_dir=str(self.root)
            )
        self.assertTrue(dist.is_initialized())
        self.assertIsInstance(classifier, TinyProbe)
        torch.testing.assert_close(classifier.pooler.weight, 2 * torch.eye(3))
        extractor = module.VJEPA.__new__(module.VJEPA)
        extractor.encoder, extractor.classifier = encoder, classifier
        extractor.transforms, extractor.finetuned = torch.nn.Identity(), True
        np.testing.assert_array_equal(extractor.get_feats(torch.ones(2, 1, 3, 1, 1)), np.full((2, 3), 2))
        # The loader also accepts a checkpoint saved without a DDP wrapper.
        torch.save({"epoch": 1, "classifier": {"pooler.weight": 3 * torch.eye(3)}}, self.root / "plain.pth")
        restored = module.load_checkpoint(torch.device("cpu"), str(self.root / "plain.pth"), TinyProbe())
        torch.testing.assert_close(restored.pooler.weight, 3 * torch.eye(3))

    def test_video_normalization_uses_channels_for_short_and_full_clips(self):
        module = self.vjepa_module()
        extractor = module.VJEPA.__new__(module.VJEPA)
        extractor.encoder = TinyAggregation(TinyEncoder())
        extractor.classifier = TinyProbe()
        extractor.classifier.pooler.weight.data.copy_(2 * torch.eye(3))
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        extractor.transforms = ChannelNormalize(mean, std)
        for frames in (3, 16):
            for finetuned in (False, True):
                with self.subTest(frames=frames, finetuned=finetuned):
                    extractor.finetuned = finetuned
                    videos = torch.arange(2 * frames * 3 * 4).reshape(2, frames, 3, 2, 2).float() / (24 * frames)
                    pixels = videos.mean((1, 3, 4)).numpy()
                    expected = (pixels - np.array(mean)) / np.array(std)
                    if finetuned:
                        expected *= 2
                    np.testing.assert_allclose(extractor.get_feats(videos), expected, atol=1e-6)

    def test_probe_loading_propagates_missing_and_mismatched_weights(self):
        module = self.vjepa_module()
        path = self.root / "probe.pth"
        for weights in ({}, {"module.pooler.weight": torch.ones(4, 4)}):
            with self.subTest(weights=list(weights)):
                torch.save({"epoch": 1, "classifier": weights}, path)
                with self.assertRaises(RuntimeError):
                    module.load_checkpoint(torch.device("cpu"), str(path), TinyProbe())
        with self.assertRaises(FileNotFoundError):
            module.load_checkpoint(torch.device("cpu"), str(self.root / "absent.pth"), TinyProbe())

    def test_encoder_loading_preserves_prefixes_and_rejects_incomplete_weights(self):
        module = self.vjepa_module()
        path = self.root / "encoder.pth"
        for key in ("target_encoder", "encoder"):
            with self.subTest(checkpoint_key=key):
                torch.save({"epoch": 1, key: {"module.backbone.weight": torch.tensor(3.0)}}, path)
                encoder = module.load_pretrained(TinyEncoder(), str(path))
                self.assertEqual(encoder.weight.item(), 3.0)
        for weights in ({}, {"weight": torch.ones(2, 2)}):
            with self.subTest(weights=list(weights)):
                torch.save({"epoch": 1, "target_encoder": weights}, path)
                with self.assertRaises(RuntimeError):
                    module.load_pretrained(torch.nn.Linear(3, 3, bias=False), str(path))

    def test_precomputed_report_describes_actual_arrays_and_file_inputs(self):
        config = JEDiScorerConfig(backend="official", num_samples=99, feature_dim=1280)
        for n_reference, n_generated in ((2, 2), (2, 5)):
            with self.subTest(n_generated=n_generated):
                reference = np.arange(n_reference * 4).reshape(n_reference, 4)
                generated = np.arange(n_generated * 4).reshape(n_generated, 4) + 1
                path = self.root / "generated.npy"
                np.save(path, generated)
                payload = {"reference_features": reference, "generated_features_path": str(path)}
                result = run_jedi_scorer(output_dir=self.root / "report", config=config, feature_payload=payload)
                self.assertEqual(result["reference_num_samples"], n_reference)
                self.assertEqual(result["generated_num_samples"], n_generated)
                self.assertEqual(result["feature_dim"], 4)
                self.assertEqual(result["num_samples"], n_reference if n_reference == n_generated else None)
                expected = compute_jedi_from_features(reference, generated)
                self.assertEqual(result["score"], expected)
                self.assertEqual(score_from_feature_payload(payload), expected)
                saved = json.loads(Path(result["results_path"]).read_text())
                self.assertEqual(saved, {key: value for key, value in result.items() if key != "results_path"})

    def test_mock_report_keeps_configured_dimensions(self):
        result = run_jedi_scorer(
            output_dir=self.root, config=JEDiScorerConfig(backend="mock", num_samples=4, feature_dim=3)
        )
        self.assertEqual((result["reference_num_samples"], result["generated_num_samples"]), (4, 4))
        self.assertEqual((result["num_samples"], result["feature_dim"]), (4, 3))


if __name__ == "__main__":
    unittest.main()
