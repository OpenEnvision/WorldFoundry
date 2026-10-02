from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "gradio" dependency at import time; skip when it is unavailable.
pytest.importorskip("gradio")

import inspect
import json
import tempfile
import unittest

from worldfoundry.studio.ui.gradio_app import (
    StudioLaunchConfig,
    _apply_camera_path_json,
    _category_frontend_profile,
    _default_camera_path_json,
    _live_controls_bridge_update,
    _resolve_primary_action,
    _run_button_label,
    _stream_button_update,
    _uses_state_init,
    build_demo,
)
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import BaseRuntimeDriver, PipelineContext, PreparedInputs, StudioManager
from worldfoundry.studio.ui.gradio_runtime import _install_api_info_guard
from worldfoundry.studio.ui.interfaces import compact_interface_summary, interface_spec_for_entry
from worldfoundry.studio.ui import (
    CUSTOM_CSS,
    HEAD_HTML,
    SPARK_MODULE_PATH,
    SPARK_ROOT,
    THREE_CORE_MODULE_PATH,
    THREE_MODULE_PATH,
)
from worldfoundry.studio.visualization.backends.frontends import spark_viewer_html
from worldfoundry.studio.visualization.backends.world import world_frontend_css, world_frontend_js


class WorldFoundryStudioInteractiveControlsTest(unittest.TestCase):
    def test_gradio_api_info_guard_handles_runtime_signature_drift(self) -> None:
        source = inspect.getsource(_install_api_info_guard)
        self.assertIn('"unexpected keyword"', source)
        self.assertIn("return original(self)", source)

    def test_head_html_includes_mobile_viewport_meta(self) -> None:
        self.assertIn('name="viewport"', HEAD_HTML)
        self.assertIn('width=device-width, initial-scale=1, viewport-fit=cover', HEAD_HTML)
        self.assertIn("document.documentElement.dataset.waViewport = mode;", HEAD_HTML)
        self.assertIn('document.querySelectorAll(".wa-main-grid, .wa-site-nav-shell")', HEAD_HTML)
        self.assertIn('const FOCUS_STORAGE_KEY = "worldfoundry-studio-stage-focus";', HEAD_HTML)
        self.assertIn('const PERFORMANCE_STORAGE_KEY = "worldfoundry-studio-performance-mode";', HEAD_HTML)

    def test_head_html_uses_in_tree_spark_runtime(self) -> None:
        self.assertEqual(SPARK_MODULE_PATH.name, "spark.module.min.js")
        self.assertEqual(THREE_MODULE_PATH.name, "three.module.js")
        self.assertEqual(THREE_CORE_MODULE_PATH.name, "three.core.js")
        self.assertEqual(SPARK_ROOT.name, "spark")
        self.assertIn(f"/file={SPARK_MODULE_PATH.resolve().as_posix()}", HEAD_HTML)
        self.assertIn(f"/file={THREE_MODULE_PATH.resolve().as_posix()}", HEAD_HTML)
        self.assertIn("python -m worldfoundry.studio.ui.vendor_assets", HEAD_HTML)
        self.assertNotIn("sparkjs.dev/releases", HEAD_HTML)
        self.assertNotIn("cdnjs.cloudflare.com", HEAD_HTML)

    def test_head_html_preview_tab_selectors_survive_gradio_injection(self) -> None:
        self.assertIn("document.querySelector('.wa-preview-panel .tabs [role=\"tablist\"]')", HEAD_HTML)
        self.assertIn("querySelectorAll('button[role=\"tab\"]')", HEAD_HTML)
        self.assertIn("querySelector('button[role=\"tab\"][aria-selected=\"true\"]')", HEAD_HTML)
        self.assertNotIn('document.querySelector(".wa-preview-panel .tabs [role="tablist"]")', HEAD_HTML)
        self.assertNotIn('querySelector("button[role="tab"][aria-selected="true"]")', HEAD_HTML)

    def test_stream_button_visible_for_stream_models(self) -> None:
        update = _stream_button_update(find_entry("infinite-world"))

        self.assertTrue(update["visible"])
        self.assertEqual(update["value"], "STEP")

    def test_stream_button_uses_action_and_scene_copy_for_non_video_streamers(self) -> None:
        self.assertEqual(_stream_button_update(find_entry("vggt"))["value"], "RENDER")
        openvla_update = _stream_button_update(find_entry("openvla"))
        self.assertEqual(openvla_update["value"], "NEXT")
        self.assertTrue(openvla_update["visible"])
        self.assertEqual(_stream_button_update(find_entry("lapa"))["value"], "NEXT")
        self.assertEqual(_stream_button_update(find_entry("being-h05"))["value"], "NEXT")
        self.assertEqual(_stream_button_update(find_entry("animatediff"))["value"], "EXTEND")

    def test_stream_button_hidden_for_one_shot_models(self) -> None:
        update = _stream_button_update(find_entry("depth-anything-v2"))

        self.assertFalse(update["visible"])
        self.assertEqual(update["value"], "STEP")

    def test_live_controls_bridge_tracks_live_navigation_models(self) -> None:
        self.assertTrue(_live_controls_bridge_update(find_entry("infinite-world"))["visible"])
        self.assertFalse(_live_controls_bridge_update(find_entry("vggt"))["visible"])
        self.assertFalse(_live_controls_bridge_update(find_entry("openvla"))["visible"])
        self.assertFalse(_live_controls_bridge_update(find_entry("lapa"))["visible"])
        self.assertFalse(_live_controls_bridge_update(find_entry("depth-anything-v2"))["visible"])

    def test_run_button_label_switches_for_stream_models(self) -> None:
        self.assertEqual(_run_button_label(find_entry("infinite-world")), "INIT")
        self.assertEqual(_run_button_label(find_entry("vggt")), "BUILD")
        self.assertEqual(_run_button_label(find_entry("wan-2p5")), "CALL")
        self.assertEqual(_run_button_label(find_entry("openvla")), "ACT")
        self.assertEqual(_run_button_label(find_entry("being-h05")), "ACT")
        self.assertEqual(_run_button_label(find_entry("lapa")), "INFER")
        self.assertEqual(_run_button_label(find_entry("animatediff")), "RUN")
        self.assertEqual(_run_button_label(find_entry("depth-anything-v2")), "RUN")

    def test_primary_action_uses_init_for_empty_stream_interactions(self) -> None:
        self.assertEqual(_resolve_primary_action("infinite-world", ""), "init")
        self.assertEqual(_resolve_primary_action("infinite-world", "[]"), "init")
        self.assertEqual(_resolve_primary_action("infinite-world", '["forward"]'), "init")
        self.assertEqual(_resolve_primary_action("vggt", ""), "run")
        self.assertEqual(_resolve_primary_action("depth-anything-v2", ""), "run")
        self.assertTrue(_uses_state_init(find_entry("lingbot-world")))
        self.assertFalse(_uses_state_init(find_entry("vggt")))

    def test_category_frontend_profile_specializes_model_families(self) -> None:
        depth_profile = _category_frontend_profile(find_entry("depth-anything-v2"))
        self.assertFalse(depth_profile["prompt_visible"])
        self.assertFalse(depth_profile["actions_visible"])
        self.assertEqual(depth_profile["path_label"], "Data Path")

        scene_profile = _category_frontend_profile(find_entry("vggt"))
        self.assertEqual(scene_profile["actions_label"], "Camera Tokens")
        self.assertTrue(scene_profile["camera_panel_visible"])
        self.assertTrue(scene_profile["spatial_panel_visible"])

        remote_profile = _category_frontend_profile(find_entry("wan-2p5"))
        self.assertTrue(remote_profile["endpoint_visible"])
        self.assertTrue(remote_profile["api_key_visible"])
        self.assertTrue(remote_profile["runtime_advanced_open"])

        embodied_profile = _category_frontend_profile(find_entry("openvla"))
        self.assertEqual(embodied_profile["mode_title"], "Embodied Policy")
        self.assertEqual(embodied_profile["prompt_label"], "Instruction")
        self.assertEqual(embodied_profile["actions_label"], "Action Tokens")
        self.assertTrue(embodied_profile["more_inputs_open"])
        self.assertTrue(embodied_profile["json_visible"])

        visual_action_profile = _category_frontend_profile(find_entry("lapa"))
        self.assertEqual(visual_action_profile["mode_title"], "Visual Action")
        self.assertEqual(visual_action_profile["actions_label"], "Latent Tokens")
        self.assertEqual(visual_action_profile["image_label"], "Context Frame")

        embodied_video_profile = _category_frontend_profile(find_entry("being-h05"))
        self.assertEqual(embodied_video_profile["mode_title"], "Embodied Policy")
        self.assertTrue(embodied_video_profile["json_visible"])
        self.assertFalse(_live_controls_bridge_update(find_entry("being-h05"))["visible"])

        conditioned_video_profile = _category_frontend_profile(find_entry("animatediff"))
        self.assertEqual(conditioned_video_profile["mode_title"], "Video World")
        self.assertFalse(conditioned_video_profile["actions_visible"])

        gen3c_profile = _category_frontend_profile(find_entry("gen3c"))
        self.assertEqual(gen3c_profile["mode_title"], "Camera Workbench")
        self.assertEqual(gen3c_profile["actions_label"], "Camera Tokens")
        self.assertTrue(gen3c_profile["camera_panel_visible"])
        self.assertTrue(gen3c_profile["camera_panel_open"])
        self.assertTrue(gen3c_profile["video_visible"])
        self.assertTrue(gen3c_profile["path_visible"])

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

    def test_camera_path_json_updates_call_kwargs(self) -> None:
        camera_path = _default_camera_path_json("gen3c")
        call_kwargs, status = _apply_camera_path_json("gen3c", camera_path, '{"fps": 24}')
        payload = json.loads(call_kwargs)

        self.assertEqual(payload["fps"], 24)
        self.assertIn("camera_path", payload)
        self.assertTrue(payload["export_cameras"])
        self.assertIn("camera path applied", status)

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

    def test_head_html_uses_wasd_and_jkli_keyboard_mapping(self) -> None:
        self.assertIn('w: "wa-live-forward"', HEAD_HTML)
        self.assertIn('a: "wa-live-left"', HEAD_HTML)
        self.assertIn('s: "wa-live-backward"', HEAD_HTML)
        self.assertIn('d: "wa-live-right"', HEAD_HTML)
        self.assertIn('j: "wa-live-camera-left"', HEAD_HTML)
        self.assertIn('l: "wa-live-camera-right"', HEAD_HTML)
        self.assertIn('i: "wa-live-camera-up"', HEAD_HTML)
        self.assertIn('k: "wa-live-camera-down"', HEAD_HTML)
        self.assertNotIn('q: "wa-live-camera-left"', HEAD_HTML)
        self.assertIn('syncKeyboardStickVisual(key, true);', HEAD_HTML)
        self.assertIn('document.addEventListener("keyup"', HEAD_HTML)
        self.assertIn('readPreference(JOYSTICK_STORAGE_KEY, fallback)', HEAD_HTML)
        self.assertIn('const keyboardStickState = {', HEAD_HTML)
        self.assertIn('Array.from(activeKeys).sort().join(",")', HEAD_HTML)
        self.assertIn("queueMappedControl(key);", HEAD_HTML)
        self.assertIn("hold keyboard directions to autoroll step by step", HEAD_HTML)
        self.assertIn("liveDispatchCooldownUntil", HEAD_HTML)

    def test_head_html_preserves_css_url_regex_backreference(self) -> None:
        self.assertIn(r"url\(", HEAD_HTML)
        self.assertIn(r"\1\)", HEAD_HTML)
        self.assertNotIn("\x01", HEAD_HTML)

    def test_head_html_handles_input_ready_stage_state(self) -> None:
        self.assertIn('lower.includes("input scene ready")', HEAD_HTML)
        self.assertIn('lower.includes("input source staged")', HEAD_HTML)
        self.assertIn('{ state: "input", label: "INPUT READY" }', HEAD_HTML)
        self.assertIn('activeTab !== "Preview Image"', HEAD_HTML)
        self.assertIn('selectPreviewTab("Preview Image")', HEAD_HTML)

    def test_head_html_hides_footer_meta_when_visual_preview_is_active(self) -> None:
        self.assertIn("const showWorldLabel = !activeTabHasVisualMedia() && info.state !== \"input\";", HEAD_HTML)
        self.assertIn('const showTime = getActiveTabName() === "Preview Video" && hasPreviewVideo();', HEAD_HTML)
        self.assertIn('worldNode.classList.toggle("is-hidden", !showWorldLabel);', HEAD_HTML)
        self.assertIn('timeNode.classList.add("is-hidden");', HEAD_HTML)

    def test_head_html_throttles_expensive_client_sync_work(self) -> None:
        self.assertIn("let trayThumbsDirty = true;", HEAD_HTML)
        self.assertIn("let uiTranslationDirty = true;", HEAD_HTML)
        self.assertIn('const fallback = viewport === "phone" || viewport === "narrow" ? "closed" : "open";', HEAD_HTML)
        self.assertIn("if (uiTranslationDirty) {", HEAD_HTML)
        self.assertIn("if (trayThumbsDirty) {", HEAD_HTML)
        self.assertIn('requestSync({ ui: shouldRefreshDomText, tray: shouldRefreshTray });', HEAD_HTML)

    def test_head_html_optimizes_spark_viewer_runtime(self) -> None:
        self.assertIn("const resolveViewerPixelRatio = () => {", HEAD_HTML)
        self.assertIn('document.visibilityState !== "hidden"', HEAD_HTML)
        self.assertIn("viewer.renderer.setPixelRatio(pixelRatio);", HEAD_HTML)
        self.assertIn("viewer.renderer.setSize(width, height, false);", HEAD_HTML)
        self.assertIn('window.addEventListener("wa-performance-change", requestSync);', HEAD_HTML)
        self.assertIn("const templateDefaultTab = (template) => {", HEAD_HTML)
        self.assertIn('if (template === "scene-3d") return "3D World";', HEAD_HTML)
        self.assertIn('if (template === "embodied-policy") return "Embodied Sim";', HEAD_HTML)
        self.assertIn('if (template === "visual-action" || template === "hosted-api") return "Artifacts";', HEAD_HTML)

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

    def test_custom_css_constrains_stage_column_to_preview_shell_width(self) -> None:
        self.assertIn("flex: 0 1 var(--wa-stage-shell-max) !important;", CUSTOM_CSS)
        self.assertIn("width: var(--wa-stage-shell-max) !important;", CUSTOM_CSS)
        self.assertIn("max-width: var(--wa-stage-shell-max) !important;", CUSTOM_CSS)

    def test_custom_css_hides_hidden_action_buttons(self) -> None:
        self.assertIn(".wa-control-dock .hidden,", CUSTOM_CSS)
        self.assertIn(".wa-control-dock .wa-action-step.hidden,", CUSTOM_CSS)
        self.assertIn(".wa-control-dock button.wa-action-hidden", CUSTOM_CSS)
        self.assertIn("display: none !important;", CUSTOM_CSS)

    def test_custom_css_supports_data_driven_stacked_mobile_layout(self) -> None:
        self.assertIn('.wa-main-grid[data-wa-viewport="stacked"]', CUSTOM_CSS)
        self.assertIn('.wa-main-grid[data-wa-viewport="phone"]', CUSTOM_CSS)
        self.assertIn('.wa-main-grid[data-wa-viewport="narrow"]', CUSTOM_CSS)
        self.assertIn('html[data-wa-viewport="narrow"] .wa-site-nav {', CUSTOM_CSS)
        self.assertIn('--wa-panel-pad-bottom: 232px;', CUSTOM_CSS)
        self.assertIn('bottom: 340px !important;', CUSTOM_CSS)
        self.assertIn('.wa-main-grid[data-wa-viewport="narrow"] .wa-player-footer', CUSTOM_CSS)
        self.assertIn('html[data-wa-viewport="narrow"] .wa-preview-panel .wa-status', CUSTOM_CSS)
        self.assertIn('.wa-preview-panel:not(.wa-joystick-open) .wa-joystick-dock', CUSTOM_CSS)

    def test_custom_css_adds_cli_surface_and_full_stage_media_frame(self) -> None:
        self.assertIn(".wa-player-footer-center.is-hidden,", CUSTOM_CSS)
        self.assertIn("--wa-media-frame-height:", CUSTOM_CSS)
        self.assertIn("#wa-main-preview-video .video-container,", CUSTOM_CSS)
        self.assertIn(".wa-stick-shell.is-key-active .wa-stick {", CUSTOM_CSS)

    def test_custom_css_adds_focus_and_lite_performance_modes(self) -> None:
        self.assertIn('html[data-wa-focus="stage"] .wa-left-rail,', CUSTOM_CSS)
        self.assertIn('html[data-wa-focus="stage"] .wa-preview-panel', CUSTOM_CSS)
        self.assertIn('html[data-wa-performance="lite"] .wa-panel-block,', CUSTOM_CSS)
        self.assertIn("backdrop-filter: none !important;", CUSTOM_CSS)
        self.assertIn("@media (prefers-reduced-motion: reduce)", CUSTOM_CSS)

    def test_custom_css_splits_template_workbench_surfaces(self) -> None:
        self.assertIn(".wa-template-workbench", CUSTOM_CSS)
        self.assertIn(".wa-preview-panel:not(.wa-template-interactive-world)", CUSTOM_CSS)
        self.assertIn(".wa-preview-panel:not(.wa-template-interactive-world) .wa-world-tray-shell", CUSTOM_CSS)
        self.assertIn(".wa-preview-panel:not(.wa-template-interactive-world) .wa-input-tray-gallery", CUSTOM_CSS)
        self.assertIn(".wa-template-depth-geometry .wa-joystick-dock-shell", CUSTOM_CSS)
        self.assertIn(
            '.wa-template-embodied-policy .tabs [role="tablist"] button:nth-child(1)',
            CUSTOM_CSS,
        )
        self.assertIn(".wa-template-hosted-api .wa-input-tray-gallery", CUSTOM_CSS)

    def test_build_demo_uses_fixed_launch_model_without_switcher_ui(self) -> None:
        demo = build_demo(StudioLaunchConfig(model_id="lingbot-world", variant_id="fast"))
        component_props = {
            component["props"].get("label"): component["props"]
            for component in demo.config["components"]
            if component.get("props", {}).get("label")
        }
        layout_components = [
            component["props"]
            for component in demo.config["components"]
            if "elem_classes" in component.get("props", {})
        ]
        html_markup = "\n".join(
            str(component.get("props", {}).get("value", ""))
            for component in demo.config["components"]
        )
        main_grid = next(props for props in layout_components if "wa-main-grid" in props["elem_classes"])
        left_rail = next(props for props in layout_components if "wa-left-rail" in props["elem_classes"])

        self.assertIs(component_props["Prompt"]["visible"], True)
        self.assertIs(component_props["Actions"]["visible"], True)
        self.assertIs(component_props["Device"]["visible"], True)
        self.assertIs(component_props["Task"]["visible"], False)
        self.assertIs(component_props["Last Frame"]["visible"], False)
        self.assertIs(component_props["Camera Pose"]["visible"], False)
        self.assertIs(component_props["3DGS"]["visible"], False)
        self.assertIs(component_props["Checkpoint"]["visible"], False)
        self.assertIs(component_props["Load JSON"]["visible"], False)
        self.assertIs(component_props["Call JSON"]["visible"], False)
        self.assertNotIn("Active Model", component_props)
        self.assertNotIn("Model Variant", component_props)
        self.assertNotIn("Studio CLI", component_props)
        self.assertNotIn("Search Models", component_props)
        self.assertIn("wa-main-grid-simple", main_grid["elem_classes"])
        self.assertIs(left_rail["visible"], False)
        self.assertIn('data-wa-nav="focus"', html_markup)
        self.assertIn('data-wa-nav="performance"', html_markup)

    def test_build_demo_specializes_controls_for_3d_depth_and_api_models(self) -> None:
        scene_demo = build_demo(StudioLaunchConfig(model_id="vggt"))
        scene_props = {
            component["props"].get("label"): component["props"]
            for component in scene_demo.config["components"]
            if component.get("props", {}).get("label")
        }
        scene_layout = [
            component["props"]
            for component in scene_demo.config["components"]
            if "elem_classes" in component.get("props", {})
        ]
        scene_grid = next(props for props in scene_layout if "wa-main-grid" in props["elem_classes"])
        scene_buttons = {
            component["props"].get("value"): component["props"]
            for component in scene_demo.config["components"]
            if isinstance(component.get("props", {}).get("value"), str)
        }
        self.assertIn("wa-category-3d-scene", scene_grid["elem_classes"])
        self.assertIs(scene_props["Camera Tokens"]["visible"], True)
        self.assertIs(scene_props["Camera Pose"]["visible"], True)
        self.assertIs(scene_props["Camera Path JSON"]["visible"], True)
        self.assertIs(scene_props["3DGS"]["visible"], True)
        self.assertIs(scene_props["Task"]["visible"], True)
        self.assertIs(scene_buttons["RENDER"]["visible"], True)

        gen3c_demo = build_demo(StudioLaunchConfig(model_id="gen3c"))
        gen3c_props = {
            component["props"].get("label"): component["props"]
            for component in gen3c_demo.config["components"]
            if component.get("props", {}).get("label")
        }
        gen3c_markup = "\n".join(
            str(component.get("props", {}).get("value", ""))
            for component in gen3c_demo.config["components"]
        )
        self.assertIs(gen3c_props["Camera Tokens"]["visible"], True)
        self.assertIs(gen3c_props["Seed Image"]["visible"], True)
        self.assertIs(gen3c_props["Seed Video"]["visible"], True)
        self.assertIs(gen3c_props["Preprocessed Seed Path"]["visible"], True)
        self.assertIs(gen3c_props["Camera Path JSON"]["visible"], True)
        self.assertIn("Camera Workbench", gen3c_markup)
        self.assertIn("Interactive World Bench", gen3c_markup)
        self.assertIn("pipeline_gen3c.py", gen3c_markup)
        self.assertNotIn("GEN3C authoring GUI", gen3c_markup)
        self.assertNotIn("gui/api/client.py", gen3c_markup)

        openvla_demo = build_demo(StudioLaunchConfig(model_id="openvla"))
        openvla_markup = "\n".join(
            str(component.get("props", {}).get("value", ""))
            for component in openvla_demo.config["components"]
        )
        self.assertIn("Embodied Policy Console", openvla_markup)
        self.assertIn("Embodied Sim", openvla_markup)
        self.assertIn("No Policy Action Yet", openvla_markup)

        depth_demo = build_demo(StudioLaunchConfig(model_id="depth-anything-v2"))
        depth_props = {
            component["props"].get("label"): component["props"]
            for component in depth_demo.config["components"]
            if component.get("props", {}).get("label")
        }
        self.assertIs(depth_props["Prompt"]["visible"], False)
        self.assertIs(depth_props["Actions"]["visible"], False)
        self.assertIs(depth_props["Depth Image"]["visible"], True)
        self.assertIs(depth_props["Depth Video"]["visible"], True)
        self.assertIs(depth_props["Data Path"]["visible"], True)

        api_demo = build_demo(StudioLaunchConfig(model_id="wan-2p5"))
        api_props = {
            component["props"].get("label"): component["props"]
            for component in api_demo.config["components"]
            if component.get("props", {}).get("label")
        }
        self.assertIs(api_props["Endpoint"]["visible"], True)
        self.assertIs(api_props["API Key"]["visible"], True)
        self.assertIs(api_props["Call JSON"]["visible"], True)
        self.assertIs(api_props["Condition Image"]["visible"], True)

        embodied_demo = build_demo(StudioLaunchConfig(model_id="openvla"))
        embodied_props = {
            component["props"].get("label"): component["props"]
            for component in embodied_demo.config["components"]
            if component.get("props", {}).get("label")
        }
        embodied_buttons = {
            component["props"].get("value"): component["props"]
            for component in embodied_demo.config["components"]
            if isinstance(component.get("props", {}).get("value"), str)
        }
        self.assertIs(embodied_props["Instruction"]["visible"], True)
        self.assertIs(embodied_props["Action Tokens"]["visible"], True)
        self.assertIs(embodied_props["Observation Image"]["visible"], True)
        self.assertIs(embodied_props["Observation Video"]["visible"], True)
        self.assertIs(embodied_props["Episode Path"]["visible"], True)
        self.assertIs(embodied_props["References"]["visible"], True)
        self.assertIs(embodied_props["Load JSON"]["visible"], True)
        self.assertIs(embodied_props["Call JSON"]["visible"], True)
        self.assertIs(embodied_buttons["NEXT"]["visible"], True)
        self.assertIs(embodied_buttons["ACT"]["visible"], True)

        visual_action_demo = build_demo(StudioLaunchConfig(model_id="lapa"))
        visual_action_props = {
            component["props"].get("label"): component["props"]
            for component in visual_action_demo.config["components"]
            if component.get("props", {}).get("label")
        }
        visual_action_buttons = {
            component["props"].get("value"): component["props"]
            for component in visual_action_demo.config["components"]
            if isinstance(component.get("props", {}).get("value"), str)
        }
        self.assertIs(visual_action_props["Latent Tokens"]["visible"], True)
        self.assertIs(visual_action_props["Context Frame"]["visible"], True)
        self.assertIs(visual_action_props["Context Video"]["visible"], True)
        self.assertIs(visual_action_buttons["NEXT"]["visible"], True)
        self.assertIs(visual_action_buttons["INFER"]["visible"], True)

    def test_build_demo_hides_default_gradio_progress_for_run_actions(self) -> None:
        demo = build_demo(StudioLaunchConfig(model_id="lingbot-world", variant_id="fast"))
        dependencies = demo.config["dependencies"]
        hidden_api_names = {
            dependency.get("api_name")
            for dependency in dependencies
            if dependency.get("show_progress") == "hidden"
        }

        self.assertIn("_run_start_action", hidden_api_names)
        self.assertIn("_run_stream_action", hidden_api_names)
        self.assertGreaterEqual(sum(name.startswith("handler") for name in hidden_api_names if name), 8)


if __name__ == "__main__":
    unittest.main()
