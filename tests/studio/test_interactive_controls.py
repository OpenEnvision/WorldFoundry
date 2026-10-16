from __future__ import annotations

import tempfile
import unittest

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import BaseRuntimeDriver, PipelineContext, PreparedInputs, StudioManager
from worldfoundry.studio.ui.interfaces import compact_interface_summary, interface_spec_for_entry
from worldfoundry.studio.visualization.backends.frontends import spark_viewer_html
from worldfoundry.studio.visualization.backends.world import world_frontend_css, world_frontend_js


class WorldFoundryStudioInteractiveControlsTest(unittest.TestCase):
    def test_interface_spec_tracks_local_gui_and_viewer_sources(self) -> None:
        gen3c = interface_spec_for_entry(find_entry("gen3c"))
        self.assertEqual(gen3c.template_id, "interactive-world")
        self.assertEqual(gen3c.local_repo.status, "present")
        self.assertIn("worldfoundry/pipelines/gen3c", gen3c.local_repo.path)
        self.assertIn("pipeline_gen3c.py", gen3c.local_repo.entrypoints)
        self.assertFalse(any("GEN3C authoring GUI" in ref for ref in gen3c.gui_refs))
        self.assertFalse(any("gui/api/client.py" in hint for hint in gen3c.launch_hints))

        scene = compact_interface_summary(find_entry("vggt"))
        self.assertEqual(scene["template_id"], "scene-3d")
        self.assertTrue(any("In-tree Spark 3DGS viewer" in ref for ref in scene["gui_refs"]))

        api_spec = interface_spec_for_entry(find_entry("wan-2p5"))
        self.assertEqual(api_spec.template_id, "hosted-api")

    def test_default_driver_routes_data_path_models(self) -> None:
        class DepthPipeline:
            def __call__(self, data_path: str, grayscale: bool = False):
                return {"data_path": data_path, "grayscale": grayscale}

        request = PreparedInputs(
            prompt="",
            input_path="",
            image=None,
            image_path="/tmp/worldfoundry-input.png",
            video_path=None,
            last_frame=None,
            last_frame_path=None,
            reference_images=[],
            reference_image_paths=[],
            interactions=None,
            camera_view=None,
            task_type="",
            intrinsics=None,
            meta_path="",
            panorama_path="",
            scene_name="",
            fps=16,
            num_frames=0,
            output_dir="/tmp/worldfoundry-run",
            output_path="/tmp/worldfoundry-run/out.mp4",
            call_kwargs={"grayscale": True},
            load_kwargs={},
            model_ref="",
            backend="from_pretrained",
            endpoint="",
            api_key="",
            device="cpu",
        )
        ctx = PipelineContext(
            entry=find_entry("depth-anything-v2"),
            pipeline=DepthPipeline(),
            cache_key="depth-test",
            backend="from_pretrained",
            model_ref="",
            endpoint="",
            load_kwargs={},
            device="cpu",
        )

        self.assertEqual(
            BaseRuntimeDriver()._invoke(ctx, request, mode="run"),
            {"data_path": "/tmp/worldfoundry-input.png", "grayscale": True},
        )

    def test_default_driver_requests_structured_fresh_results(self) -> None:
        class StructuredPipeline:
            def __call__(self, *, return_dict: bool = False):
                return {"return_dict": return_dict}

        request = PreparedInputs(
            prompt="",
            input_path="",
            image=None,
            image_path=None,
            video_path=None,
            last_frame=None,
            last_frame_path=None,
            reference_images=[],
            reference_image_paths=[],
            interactions=None,
            camera_view=None,
            task_type="",
            intrinsics=None,
            meta_path="",
            panorama_path="",
            scene_name="",
            fps=16,
            num_frames=0,
            output_dir="/tmp/worldfoundry-run",
            output_path="/tmp/worldfoundry-run/out.mp4",
            call_kwargs={},
            load_kwargs={},
            model_ref="",
            backend="from_pretrained",
            endpoint="",
            api_key="",
            device="cpu",
        )
        ctx = PipelineContext(
            entry=find_entry("mmaudio"),
            pipeline=StructuredPipeline(),
            cache_key="structured-test",
            backend="from_pretrained",
            model_ref="",
            endpoint="",
            load_kwargs={},
            device="cpu",
        )

        self.assertEqual(BaseRuntimeDriver()._invoke(ctx, request, mode="run"), {"return_dict": True})

    def test_prepare_inputs_defaults_action_models_to_structured_json_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            request = StudioManager(workspace_root=tmp_dir).prepare_inputs(
                entry=find_entry("openvla"),
                prompt="pick up the block",
                input_path="",
                image=None,
                video=None,
                last_frame=None,
                reference_files=None,
                interactions_text='{"robot_action": [0, 0, 0, 0, 0, 0, 1]}',
                camera_view_text="",
                task_type="",
                intrinsics_text="",
                meta_path="",
                panorama_path="",
                scene_name="",
                fps=16,
                num_frames=0,
                call_kwargs_text="{}",
                load_kwargs_text="{}",
                model_ref="",
                backend="from_pretrained",
                endpoint="",
                api_key="",
                device="cpu",
            )

        self.assertTrue(request.output_path.endswith("openvla.json"))
        self.assertIs(request.call_kwargs["return_dict"], True)
        self.assertEqual(request.call_kwargs["run_dir"], request.output_dir)

    def test_action_next_falls_back_to_structured_call_when_stream_is_absent(self) -> None:
        class ActionPipeline:
            def __init__(self) -> None:
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return {"action_trace": [{"action": "noop"}], "metadata": {"ok": True}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = StudioManager(workspace_root=tmp_dir)
            output_dir = f"{tmp_dir}/run"
            request = PreparedInputs(
                prompt="pick up the block",
                input_path="",
                image=None,
                image_path=None,
                video_path=None,
                last_frame=None,
                last_frame_path=None,
                reference_images=[],
                reference_image_paths=[],
                interactions={"robot_action": [0, 0, 0, 0, 0, 0, 1]},
                camera_view=None,
                task_type="",
                intrinsics=None,
                meta_path="",
                panorama_path="",
                scene_name="",
                fps=16,
                num_frames=0,
                output_dir=output_dir,
                output_path=f"{output_dir}/openvla.json",
                call_kwargs={"return_dict": True, "run_dir": output_dir},
                load_kwargs={},
                model_ref="",
                backend="from_pretrained",
                endpoint="",
                api_key="",
                device="cpu",
            )
            pipeline = ActionPipeline()
            ctx = PipelineContext(
                entry=find_entry("openvla"),
                pipeline=pipeline,
                cache_key="openvla-test",
                backend="from_pretrained",
                model_ref="",
                endpoint="",
                load_kwargs={},
                device="cpu",
            )
            record = BaseRuntimeDriver().run_continue(manager, ctx, request)

        self.assertEqual(record.mode, "stream")
        self.assertEqual(pipeline.calls[0]["return_dict"], True)
        self.assertEqual(pipeline.calls[0]["run_dir"], output_dir)

    def test_world_frontend_smooths_input_and_frame_swaps(self) -> None:
        js = world_frontend_js()
        css = world_frontend_css()

        self.assertIn("new RTCPeerConnection({ iceServers: state.iceServers })", js)
        self.assertIn('createDataChannel("controls", { ordered: true })', js)
        self.assertIn("el.video.srcObject = stream;", js)
        self.assertIn('action: { event: active ? "keydown" : "keyup", key: normalized }', js)
        self.assertNotIn("function stepLoop", js)
        self.assertNotIn("STEP_INTERVAL_MS", js)
        self.assertNotIn("function startPreviewMotion", js)
        self.assertNotIn("--preview-offset-x", css)
        self.assertIn("will-change: opacity;", css)

    def test_standalone_spark_viewer_avoids_wasteful_render_work(self) -> None:
        html = spark_viewer_html(title="Test", default_asset="")

        self.assertIn("const pixelRatio = () =>", html)
        self.assertIn("viewer.running", html)
        self.assertIn('document.visibilityState === "hidden"', html)
        self.assertIn("viewer.width !== width || viewer.height !== height", html)


if __name__ == "__main__":
    unittest.main()
