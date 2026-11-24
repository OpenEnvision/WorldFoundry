.PHONY: help install-core install-dev test test-infer test-infer-tensors test-infer-contracts test-infer-cuda-contracts test-geometry test-eval-core docs-check docs-dev-fast docs-dev-ssd docs-dev-local docs-build-fast cli-entrypoint-check lint ruff-check format-check shell-check data-check runtime-registry-check workspace-registry-check check-cuda-constraints packaging-check compile-eval cli-check precommit precommit-install preflight
.PHONY: test-infer-video-tensors plan-inference replay-inference coverage-inference verify-inference
.PHONY: test-docs-contracts pr-check
.PHONY: test-mg2-checkpoint-contracts test-mg2-trajectory-contracts test-mg2-conditioned-trajectory-contracts
.PHONY: test-infer-sol-cuda-contracts test-lightvae-checkpoint-contracts

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PRE_COMMIT ?= $(PYTHON) -m pre_commit
PYTHONPATH ?= .
WORLDFOUNDRY_EVAL ?= $(PYTHON) -m worldfoundry.cli
PREFLIGHT_PROFILE ?= all
PREFLIGHT_OUTPUT ?= tmp/preflight
CLI_CHECK_OUTPUT ?= tmp/ci-cli-check
TEST_ARGS ?=
BASE_REF ?=
HEAD_REF ?= HEAD
PUBLIC_TEST_MANIFEST ?= tmp/public-cpu-selection.json
PUBLIC_TEST_JUNIT ?= tmp/public-cpu-contracts.xml
DOCS_TEST_MANIFEST ?= tmp/docs-contract-selection.json
DOCS_TEST_JUNIT ?= tmp/docs-contracts.xml
INFER_TENSOR_MANIFEST ?= tmp/inference-cpu-selection.json
INFER_TENSOR_JUNIT ?= tmp/inference-cpu-contracts.xml
STRICT_TEST_RUNNER = $(PYTHON) tests/manual/validate_junit_contract.py run
PUBLIC_GATE_CONTRACTS = \
	tests/evaluation/execution/test_stevo_bench_eval_utils.py \
	tests/evaluation/test_source_cache.py \
	tests/eval_core/test_videoverse_local_vlm.py \
	tests/eval_core/test_libero_mem_recording.py \
	tests/eval_core/test_model_benchmark_resume_controls.py \
	tests/eval_core/test_run_manifest.py \
	tests/runtime/test_geometry_regression_lazy_dispatch.py \
	tests/runtime/test_native_coverage_policy.py \
	tests/runtime/test_geometry_regression_declarations.py \
	tests/runtime/test_junit_contract_validator.py \
	tests/core/test_repository_policies.py \
	tests/core/test_inference_gate_selection.py \
	tests/scripts/test_pr_checks.py \
	tests/scripts/test_model_env_install_shell_compatibility.py \
	tests/runtime/test_conda_setup_commands.py \
	tests/core/test_docs_toolchain.py \
	tests/core/test_agent_skills.py \
	tests/core/test_inference_validation_guidance.py \
	tests/cli/test_mcp_content_conversion.py \
	tests/cli/test_config_run.py \
	tests/cli/test_mcp_run_results.py \
	tests/cli/test_zoo_benchmark_fetch.py \
	tests/runtime/test_inference_actions_evidence.py \
	tests/synthesis/test_matrix_game_2_historical_reference.py \
	tests/studio/test_canonical_catalog_lookup.py \
	tests/studio/test_ltx25_workspace_contract.py \
	tests/eval_core/test_ltx25_integration.py::test_ltx25_catalog_binding_profile_and_environment_are_wired \
	tests/eval_core/test_ltx25_integration.py::test_ltx25_pipeline_loads_from_standard_runner_spec \
	tests/studio/test_catalog_class_constants.py \
	tests/base_models/diffusion_model/test_module_boundaries.py \
	tests/studio/test_pipeline_owned_stream_state.py \
	tests/studio/test_dispatch_lifecycle.py \
	tests/studio/test_stream_init.py \
	tests/studio/test_stream_step.py \
	tests/studio_visualization/test_viser_initial_camera.py \
	tests/core/utils/python/test_functional_utils.py \
	tests/core/utils/python/test_pattern_filters.py \
	tests/core/utils/python/test_tree_utils.py
VIDEO_TENSOR_CONTRACTS = \
	tests/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/comm/test_padding.py \
	tests/core/test_video_tensor_regression.py \
	tests/core/test_video_codec_contracts.py \
	tests/core/test_causal_video_cache.py \
	tests/base_models/diffusion_model/optimizations/test_static_cross_kv.py \
	tests/base_models/diffusion_model/test_matrix_game4.py \
	tests/eval_core/test_matrix_game4_integration.py
STREAMING_CPU_CONTRACTS = \
	tests/synthesis/test_base_synthesis_lazy_grad.py \
	tests/synthesis/test_matrix_game_2_encoder_precision.py \
	tests/synthesis/test_matrix_game_2_conditioning.py \
	tests/operators/test_matrix_game_2_perception.py \
	tests/base_models/test_flashdreams_sana_wm_streaming.py \
	tests/base_models/test_flashdreams_fastvideo_causal_wan.py \
	tests/base_models/test_self_forcing_native.py \
	tests/base_models/test_self_forcing_long_rollout.py \
	tests/base_models/test_causal_forcing_native.py \
	tests/base_models/diffusion_model/test_causal_attention_padding.py \
	tests/base_models/diffusion_model/test_causal_cache_positions.py \
	tests/base_models/diffusion_model/optimizations/test_causal_qkv_fusion.py \
	tests/base_models/diffusion_model/test_autoregressive_window_runner.py \
	tests/base_models/diffusion_model/models/autoencoders/test_wan_vae_acceleration_policy.py \
	tests/base_models/diffusion_model/models/autoencoders/test_wan_streaming_chunk.py \
	tests/base_models/diffusion_model/models/autoencoders/test_wan21_tae_preview.py \
	tests/core/acceleration/test_fused_fp8_ffn.py \
	tests/core/acceleration/test_convolution_layout.py \
	tests/core/attention/test_native_cudnn_fp8.py \
	tests/core/attention/test_native_causal_bias.py \
	tests/core/attention/test_triton_tma.py \
	tests/core/test_streaming_step_control.py \
	tests/synthesis/test_matrix_game_2_optimizations.py \
	tests/synthesis/test_matrix_game_2_decode_overlap.py \
	tests/synthesis/test_matrix_game_2_realtime.py \
	tests/synthesis/test_sana_wm_realtime.py \
	tests/studio_visualization/test_realtime_nvenc.py \
	tests/studio_visualization/test_realtime_presentation.py \
	tests/studio_visualization/test_realtime_shutdown_ownership.py \
	tests/studio_visualization/test_world_realtime.py \
	tests/studio_visualization/test_realtime_postprocess_presets.py \
	tests/studio_visualization/test_realtime_input.py \
	tests/studio_visualization/test_realtime_input_consumers.py \
	tests/studio_visualization/test_realtime_input_golden.py \
	tests/runtime/test_inference_benchmark_correctness.py \
	tests/studio_visualization/test_realtime_delivery.py \
	tests/runtime/test_realtime_replay.py \
	tests/studio_visualization/test_realtime_sensors.py \
	tests/studio_visualization/test_realtime_sensor_transport.py
DIFFUSION_ACCELERATION_CPU_CONTRACTS = \
	tests/core/acceleration/test_plugins.py \
	tests/core/acceleration/test_easycache.py \
	tests/core/acceleration/test_fp8_conv.py \
	tests/core/acceleration/test_svdquant.py \
	tests/core/acceleration/test_mha_schedule.py \
	tests/core/test_sol_attention.py \
	tests/base_models/diffusion_model/test_camera_attention_dispatch.py \
	tests/base_models/diffusion_model/optimizations/test_calibrated_plugins.py \
	tests/base_models/diffusion_model/optimizations/test_selective_fp8.py \
	tests/base_models/diffusion_model/optimizations/test_kv_fusion.py \
	tests/base_models/diffusion_model/optimizations/test_plugin_lifecycle.py \
	tests/base_models/diffusion_model/optimizations/test_acceleration_plugins.py \
	tests/base_models/diffusion_model/optimizations/test_wan_dual_easycache.py \
	tests/base_models/diffusion_model/optimizations/test_wan_feature_cache_config.py \
	tests/base_models/diffusion_model/models/denoisers \
	tests/base_models/diffusion_model/loaders/test_module_acceleration_plugins.py \
	tests/base_models/test_wan22_a14b_inference.py \
	tests/benchmarks/test_plugin_diagnostics.py \
	tests/benchmarks/test_lightvae_encoder.py \
	tests/base_models/diffusion_model/models/autoencoders/test_lightvae_wan21.py::test_default_teacher_and_explicit_student_geometry_remain_distinct \
	tests/base_models/diffusion_model/models/autoencoders/test_lightvae_wan21.py::test_strict_converter_accepts_only_all_matched_student_keys_and_shapes \
	tests/base_models/diffusion_model/models/autoencoders/test_lightvae_wan21.py::test_teacher_and_48_channel_checkpoint_rejected \
	tests/base_models/diffusion_model/models/autoencoders/test_lightvae_wan21.py::test_non_inference_student_build_rejected_before_checkpoint_read \
	tests/base_models/diffusion_model/models/autoencoders/test_lightvae_wan21.py::test_wan22_and_unknown_variant_rejected_before_loading_or_preview \
	tests/base_models/diffusion_model/models/autoencoders/test_lightvae_wan21.py::test_unvalidated_student_options_rejected_before_loading \
	tests/base_models/diffusion_model/models/autoencoders/test_lightvae_wan21.py::test_student_tiling_and_parallel_are_explicitly_unvalidated \
	tests/base_models/diffusion_model/models/autoencoders/test_lightvae_wan21.py::test_student_compile_offload_and_lower_precision_rejected \
	tests/core/utils/test_inference_graph.py \
	tests/benchmarks/test_wan_execution_preparation.py \
	tests/base_models/diffusion_model/test_wan_execution_preparation.py \
	tests/base_models/diffusion_model/test_request_batching.py \
	tests/base_models/diffusion_model/test_image_request_batching.py \
	tests/base_models/diffusion_model/loaders/test_module_compile_wiring.py
INFER_TENSOR_CONTRACTS = \
	tests/base_models/test_transnetv2_unsupported_options.py \
	tests/base_models/test_device_coefficient_flow_unipc_scheduler.py \
	tests/base_models/diffusion_model/models/autoencoders/wan/test_shared_leaves.py \
	tests/base_models/diffusion_model/models/autoencoders/wan/test_diffusers_blocks.py \
	tests/base_models/diffusion_model/models/encoders/wan/test_clip_bindings.py \
	tests/synthesis/test_navigation_input_contracts.py \
	tests/synthesis/test_wan_camera_component_reuse.py \
	tests/synthesis/test_open_magvit2_component_reuse.py \
	tests/synthesis/test_conditioning_component_reuse.py \
	tests/synthesis/test_perception_component_reuse.py \
	tests/synthesis/test_mmaudio_component_reuse.py \
	tests/core/test_sharded_safetensors.py \
	tests/synthesis/test_go1_checkpoint_loading.py \
	tests/synthesis/test_lda_1b_checkpoint_loading.py \
	tests/synthesis/test_galaxea_vla_mixture.py \
	tests/core/attention/test_qk_rotary_embedding.py \
	tests/base_models/llm_mllm_core/test_qwen_layers.py \
	tests/base_models/llm_mllm_core/test_qwen_rotary.py \
	tests/base_models/llm_mllm_core/test_qwen_vision.py \
	tests/base_models/llm_mllm_core/test_action_tensor_ops.py \
	tests/base_models/llm_mllm_core/test_lingbot_components.py \
	tests/base_models/llm_mllm_core/test_rms_norm_variants.py \
	tests/base_models/llm_mllm_core/test_diffusion_policy_components.py \
	tests/base_models/llm_mllm_core/test_cogact_components.py \
	tests/base_models/llm_mllm_core/test_molmo_preprocessing.py \
	tests/base_models/llm_mllm_core/test_mllm_sampling.py \
	tests/base_models/llm_mllm_core/test_action_transformer.py \
	tests/base_models/llm_mllm_core/test_rdt.py \
	tests/base_models/diffusion_model/models/networks/dit/test_token.py \
	tests/base_models/diffusion_model/models/representations/ltx/test_conditioning_helpers.py \
	tests/base_models/perception_core/optical_flow/test_raft_features.py \
	tests/base_models/diffusion_model/models/networks/helios/test_shared_components.py \
	tests/core/nn/blocks/test_volume.py \
	tests/base_models/three_dimensions/point_clouds/pi3_inference/test_conv_head.py \
	tests/core/nn/blocks/test_module_properties.py \
	tests/core/nn/blocks/test_layers.py \
	tests/core/attention/rotary/test_rope_2d.py \
	tests/base_models/llm_mllm_core/test_prismatic.py \
	tests/base_models/diffusion_model/models/autoencoders/test_alias_free.py \
	tests/base_models/llm_mllm_core/action/gr00t/test_dit_components.py \
	tests/base_models/diffusion_model/models/networks/lvdm/test_shared_feedforward.py \
	tests/core/nn/diffusion/test_schedulers.py \
	tests/core/nn/blocks/test_normalization.py \
	tests/core/nn/diffusion/test_utils.py \
	tests/core/attention/rotary/test_interleaved.py \
	tests/core/attention/rotary/test_rope_nd.py \
	tests/core/attention/rotary/test_embedding.py \
	tests/base_models/three_dimensions/point_clouds/pi3_inference/test_prope.py \
	tests/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/test_inference_helpers.py \
	tests/base_models/diffusion_model/models/networks/gamma_world/test_multiagent_rope.py \
	tests/base_models/diffusion_model/models/networks/wan/test_varlen_cross_attention.py \
	tests/core/media/processing/test_video_transforms.py \
	tests/base_models/test_hyperflow_components.py \
	tests/synthesis/test_hyperflow_runtime.py \
	tests/base_models/test_minimax_h3_dit.py \
	tests/pipelines/test_pika_h3_flex.py \
	tests/pipelines/test_h3_world.py \
	tests/base_models/test_h3_tiny_components.py \
	tests/pipelines/test_h3_tiny_pipeline.py \
	tests/base_models/test_pi_h3_components.py \
	tests/pipelines/test_pi_h3_pipeline.py \
	tests/base_models/test_fastframes_components.py \
	tests/pipelines/test_fastframes_pipeline.py \
	tests/base_models/test_pi4_components.py \
	tests/pipelines/test_pi4_pipeline.py \
	tests/base_models/diffusion_model/models/initializers/test_easyanimate_conditioning.py \
	tests/synthesis/test_easyanimate_runtime.py \
	tests/synthesis/test_lyra1_attention_reuse.py \
	tests/synthesis/test_adaworld_component_reuse.py \
	tests/base_models/diffusion_model/models/networks/animatediff/test_resnet.py \
	tests/synthesis/test_wan_text_encoder_reuse.py \
	tests/synthesis/test_wan_vae_component_reuse.py \
	tests/synthesis/test_wan_transformer_component_reuse.py \
	tests/synthesis/test_wan_family_components.py \
	tests/synthesis/test_rcm_wan_components.py \
	tests/synthesis/test_gr00t_preprocessing.py \
	tests/synthesis/test_being_h05_components.py \
	tests/base_models/test_omnivinci_components.py \
	tests/synthesis/test_omnivinci_runtime.py \
	tests/synthesis/test_spatia_runtime.py \
	tests/synthesis/test_versecrafter_runtime.py \
	tests/synthesis/test_uni3c_synthesis.py \
	tests/synthesis/test_minwm_runtime.py \
	tests/synthesis/test_evoke_runtime.py \
	tests/synthesis/test_evoke_shared_geometry.py \
	tests/synthesis/test_official_runtime_launches.py \
	tests/synthesis/test_unianimate_worldfoundry_runner.py \
	tests/synthesis/test_kairos_config.py \
	tests/synthesis/test_moverse_runtime.py \
	tests/synthesis/test_mira_dino.py \
	tests/base_models/test_mosaic_gpt.py \
	tests/synthesis/test_roboflamingo_factory.py \
	tests/synthesis/test_official_policy_runtime.py \
	tests/synthesis/test_official_policy_dispatch.py \
	tests/synthesis/test_molmobot_runtime.py \
	tests/synthesis/test_molmobot_llm.py \
	tests/synthesis/test_lingbot_va_runtime.py \
	tests/synthesis/test_tesseract_runtime.py \
	tests/synthesis/test_dreamzero_runtime.py \
	tests/synthesis/test_lyra2_imports.py \
	tests/synthesis/test_starwm_runtime.py \
	tests/synthesis/test_pusa_vidgen_adapter.py \
	tests/synthesis/test_open_oasis_runtime.py \
	tests/synthesis/test_open_magvit2_runtime.py \
	tests/synthesis/test_nwm_runtime.py \
	tests/synthesis/test_le_wm_runtime.py \
	tests/synthesis/test_hunyuan_worldplay_runtime.py \
	tests/synthesis/test_giga_world_0_runtime.py \
	tests/synthesis/test_easyanimate_runtime.py \
	tests/synthesis/test_bernini_runtime.py \
	tests/synthesis/test_allegro_runtime.py \
	tests/synthesis/test_adaworld_runtime.py \
	tests/synthesis/test_show_o_runtime.py \
	tests/synthesis/test_mineworld_runtime.py \
	tests/synthesis/test_diamond_runtime.py \
	tests/synthesis/test_dino_wm_runtime.py \
	tests/synthesis/test_scope_runtime.py \
	tests/synthesis/test_scope_device_routing.py \
	tests/synthesis/test_hydra_runtime.py \
	tests/synthesis/test_vmem_runtime.py \
	tests/synthesis/test_neoverse_runtime.py \
	tests/synthesis/test_longvie_runtime.py \
	tests/synthesis/test_animatediff_runtime.py \
	tests/synthesis/test_liveworld_runtime.py \
	tests/synthesis/test_ctrl_world_runtime.py \
	tests/synthesis/test_warp_as_history_runtime.py \
	tests/synthesis/test_magic_world_runtime.py \
	tests/synthesis/test_video_x_fun_runtime.py \
	tests/synthesis/test_solaris_runtime.py \
	tests/synthesis/test_longcat_video_runtime.py \
	tests/synthesis/test_ac3d_controlnet_pipeline.py \
	tests/pipelines/test_ac3d_integration.py \
	tests/synthesis/test_genie_envisioner_runtime.py \
	tests/synthesis/test_wan2pt1_t2v_causal_infer.py \
	tests/base_models/test_vqascore_llama32_molmo_gemini.py \
	tests/pipelines/test_kandinsky6_replay_recipes.py \
	tests/base_models/diffusion_model/test_build_purpose.py \
	tests/synthesis/test_inspatio_v15_tensors.py \
	tests/synthesis/test_inspatio_reproducibility.py \
	tests/synthesis/test_inspatio_python_orchestration.py \
	tests/synthesis/test_inspatio_world_runtime.py \
	tests/core/execution \
	tests/core/platform \
	tests/core/model_loading/test_checkpoint_roundtrip.py \
	tests/core/model_loading/test_checkpoint_transforms.py \
	tests/core/acceleration/test_checkpoint_quantization.py \
	tests/core/test_bt709_audio_video.py \
	tests/base_models/test_ltx25_canonical_components.py \
	tests/base_models/test_ltx_attention_masks.py \
	tests/base_models/test_ltx_sigma_schedules.py \
	tests/base_models/test_ltx25_native_recipe.py \
	tests/base_models/test_ltx_ic_lora.py \
	tests/pipelines/test_ltx2_v2v.py \
	tests/base_models/diffusion_model/test_joint_multistage_hooks.py \
	tests/eval_core/test_ltx25_integration.py \
	tests/core/geometry/test_geometry_conventions.py \
	tests/core/geometry/test_shared_rotation_transforms.py \
	tests/core/geometry/test_projection.py \
	tests/base_models/three_dimensions/depth/midas/test_fusion.py \
	tests/base_models/three_dimensions/depth/depth_anything/test_geometry_bindings.py \
	tests/base_models/three_dimensions/depth/test_temporal.py \
	tests/base_models/perception_core/general_perception/dinov2/test_shared_vision_leaves.py \
	tests/base_models/perception_core/general_perception/dinov3/test_omega_components.py \
	tests/base_models/three_dimensions/point_clouds/vggt/test_shared_blocks.py \
	tests/base_models/three_dimensions/point_clouds/vggt/test_head_utils.py \
	tests/core/attention/test_indexed_2d.py \
	tests/base_models/perception_core/segment/sam_components/test_heads.py \
	tests/base_models/perception_core/segment/sam_components/test_mask_ops.py \
	tests/base_models/perception_core/segment/sam_components/test_memory.py \
	tests/base_models/perception_core/segment/sam_components/test_transformer.py \
	tests/core/attention/test_window.py \
	tests/core/geometry/test_boxes.py \
	tests/runtime/test_geometry_regression.py \
	tests/runtime/test_geometry_regression_declarations.py \
	tests/synthesis/test_dreamx_world_backend_policy.py \
	tests/synthesis/test_solarwm_component_reuse.py \
	tests/synthesis/test_solarwm_inference_only.py \
	tests/synthesis/test_solarwm_ltx_component_reuse.py \
	tests/synthesis/test_solarwm_wan_network_reuse.py \
	tests/synthesis/test_joyai_ltx_component_reuse.py \
	tests/synthesis/test_ltx_media_reuse.py \
	tests/synthesis/test_joyai_ltx_transformer_reuse.py \
	tests/synthesis/test_alayaworld_v11_ltx_reuse.py \
	tests/core/test_diffusion_rounding.py \
	tests/core/test_kernel_dispatch_receipt.py \
	tests/core/test_diffusion_mutating_kernel.py \
	$(VIDEO_TENSOR_CONTRACTS) \
	$(STREAMING_CPU_CONTRACTS) \
	$(DIFFUSION_ACCELERATION_CPU_CONTRACTS) \
	tests/base_models/diffusion_model/test_helixworld.py \
	tests/eval_core/test_helixworld_integration.py::test_helixworld_catalog_recipe_and_public_pipeline_resolve \
	tests/base_models/diffusion_model/test_worldplay2.py \
	tests/base_models/diffusion_model/test_worldplay2_compressor_pdd.py \
	tests/base_models/diffusion_model/test_worldplay2_upstream_parity.py \
	tests/eval_core/test_worldplay2_integration.py \
	tests/core/utils/tensors \
	tests/pipelines/test_replay_recipe_wiring.py \
	tests/core/test_video_chunk_transport.py \
	tests/core/test_video_ffmpeg_fallback.py \
	tests/core/test_video_window.py \
	tests/core/test_video_postprocess.py \
	tests/core/test_video_postprocess_rtx.py \
	tests/core/test_postprocess_presets.py \
	tests/pipelines/test_native_output_contracts.py \
	tests/pipelines/test_non_native_output_contracts.py \
	tests/pipelines/test_non_native_batch_outputs.py \
	tests/pipelines/test_trajectory_output_contracts.py \
	tests/pipelines/test_astra_request_isolation.py \
	tests/synthesis/test_astra_state.py \
	tests/pipelines/test_independent_native_outputs.py \
	tests/studio/test_independent_output_dispatch.py \
	tests/pipelines/test_cosmos_predict2_workspace_contract.py \
	tests/pipelines/test_reason1_runtime_options.py \
	tests/pipelines/test_cosmos_inference_parallel_options.py \
	tests/pipelines/test_hunyuan_video_checkpoint_paths.py \
	tests/pipelines/test_hunyuan_mirror_masks.py \
	tests/studio/test_cosmos_predict25_local_overrides.py \
	tests/studio/test_cosmos_predict2_workspace_defaults.py \
	tests/studio/test_hunyuan_safe_checkpoint_overrides.py \
	tests/studio/test_t2v_turbo_catalog.py \
	tests/studio/test_full_quality_video_defaults.py \
	tests/eval_core/test_cosmos3_in_tree_integration.py \
	tests/eval_core/test_vchitect_runner.py \
	tests/pipelines/test_gen3c_workspace_contract.py \
	tests/studio/test_trajectory_output_dispatch.py \
	tests/studio/test_lyra1_catalog.py \
	tests/eval_core/test_lyra1_reconstruction.py \
	tests/synthesis/test_lyra2_output_contracts.py \
	tests/synthesis/test_lyra2_runtime_paths.py \
	tests/synthesis/test_visual_frame_memory_outputs.py \
	tests/synthesis/test_lingbot_v2_batch_outputs.py \
	tests/synthesis/test_longvie2_segments.py \
	tests/synthesis/test_non_native_realtime_outputs.py \
	tests/synthesis/test_matrix_game_3_realtime.py \
	tests/synthesis/test_dreamx_world_ar_realtime.py \
	tests/synthesis/test_helios_realtime.py \
	tests/pipelines/test_native_diffusion_runtime_policy.py \
	tests/pipelines/test_native_diffusion_policy_normalization.py \
	tests/base_models/diffusion_model/test_hy_nomas.py \
	tests/pipelines/test_hy_nomas.py \
	tests/pipelines/test_forcing_runtime.py \
	tests/pipelines/test_forcing_integration.py \
	tests/studio/test_forcing_catalog_defaults.py \
	tests/pipelines/test_native_product_batching.py \
	tests/studio/test_native_request_batching.py \
	tests/studio/test_inference_requests_attention.py \
	tests/studio/test_matrix_game3_torchrun_bridge.py \
	tests/synthesis/test_lingbot_world_realtime.py \
	tests/synthesis/test_lingbot_realtime_policy.py \
	tests/synthesis/test_lingbot_decoder_contracts.py \
	tests/synthesis/test_lingbot_world_v2_realtime.py \
	tests/synthesis/test_lingbot_world_v2_small.py \
	tests/synthesis/test_lingbot_world_v2_runtime.py \
	tests/synthesis/test_lingbot_video_runtime.py \
	tests/synthesis/test_lingbot_video_checkpoint_loading.py \
	tests/synthesis/test_hunyuan_worldplay_realtime.py \
	tests/synthesis/test_hunyuan_worldplay_state.py \
	tests/synthesis/test_lingbot_realtime_cache_cursors.py \
	tests/synthesis/test_lingbot_i2v_condition_reuse.py \
	tests/synthesis/test_lingbot_static_inputs.py \
	tests/core/attention/test_complex_rotary_preparation.py \
	tests/core/attention/test_rotary_imports.py \
	tests/pipelines/test_sana_integration.py \
	tests/operators/test_worldfm_operator.py \
	tests/pipelines/test_matrix_game_batch_outputs.py \
	tests/pipelines/test_matrix_game_1.py \
	tests/synthesis/test_matrix_game_3p5_runtime_assets.py \
	tests/pipelines/test_matrix_game_2_seed_repro.py \
	tests/pipelines/test_worldfm_contracts.py \
	tests/pipelines/test_lingbot_distributed_loading.py \
	tests/pipelines/test_wonderworld_integration.py \
	tests/base_models/test_sd2_marigold_components.py \
	tests/base_models/test_decoded_depth_guidance.py \
	tests/base_models/test_scene_segmentation.py \
	tests/base_models/test_moge_uniworld_variant.py \
	tests/base_models/test_metric3d_config.py \
	tests/base_models/test_midas_checkpoint_compat.py \
	tests/base_models/test_vggsfm_dino_loading.py \
	tests/base_models/test_open_flamingo_imports.py \
	tests/base_models/test_vipe_native_extension.py \
	tests/core/test_vit_qkv.py \
	tests/synthesis/test_wonderworld_stages.py \
	tests/core/test_gaussian_surfels.py \
	tests/base_models/diffusion_model/models/encoders/test_wan_animate_pose_render.py \
	tests/base_models/diffusion_model/models/encoders/test_wan_animate_pose_sequence.py \
	tests/core/distributed \
	tests/core/test_compile_cache_layering.py \
	tests/core/test_cache_atomicity.py \
	tests/core/test_memory_store.py \
	tests/runtime/test_bounded_command.py \
	tests/runtime/test_jobs_store_persistence.py \
	tests/synthesis/test_db_cogact_runtime.py \
	tests/synthesis/test_db_cogact_sampling.py \
	tests/studio/test_inference_output_integrity.py \
	tests/studio/test_workspace_app.py \
	tests/pipelines/test_component_exports.py \
	tests/core/attention/test_range_attention.py \
	tests/synthesis/test_dreamdojo_runtime.py \
	tests/synthesis/test_dreamdojo_student.py \
	tests/synthesis/test_dreamdojo_distcp_conversion.py \
	tests/pipelines/test_magi2_refiner.py \
	tests/pipelines/test_magi2_pipeline.py \
	tests/synthesis/test_magi2_unified_runtime.py \
	tests/eval_core/test_catalog_integration_status.py \
	$(EVALUATION_CPU_CONTRACTS)
EVALUATION_CPU_CONTRACTS = \
	tests/evaluation/execution/test_videobench_text_alignment.py \
	tests/evaluation/execution/test_wbench_sam2_tracking_summary.py \
	tests/evaluation/execution/test_stevo_bench_eval_utils.py \
	tests/evaluation/execution/test_t2vworldbench_evaluator_imports.py \
	tests/evaluation/execution/test_t2v_compbench_mllm_metric_imports.py \
	tests/evaluation/execution/test_fetv_blip_modes.py \
	tests/evaluation/execution/test_fetv_video_frames.py \
	tests/evaluation/execution/test_vmbench_qalign_helpers.py \
	tests/evaluation/execution/test_vmbench_cas_writer.py \
	tests/evaluation/execution/test_vbench_sampling.py \
	tests/evaluation/execution/test_vbench_metadata_errors.py \
	tests/eval_core/test_worldmembench_official_runner.py \
	tests/eval_core/test_worldmembench_runtime.py \
	tests/evaluation/execution/test_resume_options.py \
	tests/evaluation/execution/test_result_integrity.py \
	tests/evaluation/execution/test_helper_reuse.py \
	tests/evaluation/execution/test_pipeline_boundaries.py \
	tests/evaluation/execution/test_generation_artifact_materialization.py \
	tests/evaluation/execution/test_shared_scoring_backend.py \
	tests/evaluation/execution/test_shared_quality.py \
	tests/evaluation/execution/test_worldmembench_camera.py \
	tests/evaluation/execution/test_vbench_appearance_reduction.py \
	tests/evaluation/execution/test_worldscore_evaluator_lifecycle.py \
	tests/evaluation/metrics/test_scalar_aggregation.py \
	tests/evaluation/metrics/test_worldscore_aggregation.py \
	tests/evaluation/metrics/test_fvd_batch_preprocessing.py \
	tests/evaluation/metrics/test_fvd_spatial_squeeze.py \
	tests/cli/test_cli_mcp_contracts.py \
	tests/cli/test_cli_mcp_usage_context.py \
	tests/cli/test_mcp_content_conversion.py \
	tests/core/test_logging_console_lifecycle.py

# Select numerical CUDA nodes explicitly: older CUDA tests do not all carry the gpu marker.
INFER_CUDA_CONTRACTS = \
	tests/base_models/diffusion_model/test_action_attention_dispatch.py \
	tests/synthesis/test_inspatio_renderer_gpu.py::test_strict_renderer_matches_cpu_and_repeats \
	tests/core/execution/test_cuda_frame_transfer.py::test_cuda_host_pixels_wait_for_producer_and_preserve_frame_order \
	tests/core/execution/test_cuda_frame_transfer.py::test_cuda_prefetch_failure_uses_the_correct_blocking_fallback \
	tests/core/execution/test_cuda_frame_transfer.py::test_failed_cuda_callback_drains_queued_work_before_reuse \
	tests/core/execution/test_inference_graph_cuda.py \
	tests/core/test_diffusion_mutating_kernel.py::test_bf16_mutating_kernel_preserves_alias_version_and_eager_bits \
	tests/core/test_diffusion_rounding.py::test_eager_rounding_and_explicit_model_backend \
	tests/core/test_diffusion_rounding.py::test_rounding_with_strided_modulation_across_supported_ranks \
	tests/core/test_diffusion_rounding.py::test_layer_norm_modulation_preserves_vendor_bits_and_upcast \
	tests/base_models/diffusion_model/optimizations/test_selective_fp8.py::test_real_cuda_scaled_mm_receipts_layout_and_exact_removal \
	tests/base_models/diffusion_model/optimizations/test_kv_fusion.py::test_cuda_bf16_real_native_cross_attention_is_finite_and_close \
	tests/base_models/diffusion_model/optimizations/test_acceleration_plugins.py::test_sana_cuda_default_fusion_keeps_native_strides_and_output \
	tests/core/acceleration/test_fp8_conv.py::test_fp8_implicit_gemm_matches_reference \
	tests/core/acceleration/test_svdquant.py::test_packed_integer_kernel_matches_independent_groupwise_reference \
	tests/core/acceleration/test_mha_schedule.py::test_pointer_fa2_executes_unequal_sequence_lengths \
	tests/core/acceleration/test_mha_schedule.py::test_pointer_fa2_wan_head_width_and_fused_projection_strides \
	tests/core/acceleration/test_mha_schedule.py::test_optional_schedule_provider_matches_torch \
	tests/core/acceleration/test_mha_schedule.py::test_scheduled_projection_uses_real_low_precision_kernel \
	tests/core/acceleration/test_mha_schedule.py::test_e5m2_masked_gemm_matches_independent_quantized_reference \
	tests/core/attention/test_kv_arena.py::test_cuda_long_sequence_attention_and_stream_ownership \
	tests/core/attention/test_kv_arena.py::test_cuda_graph_replay_reads_updated_current_segment \
	tests/core/attention/test_gqa_storage.py::test_efficient_cuda_gqa_matches_math \
	tests/core/attention/test_native_causal_bias.py::test_varlen_packed_causal_automatically_uses_symbolic_bias \
	tests/base_models/diffusion_model/models/autoencoders/test_wan_streaming_chunk.py::test_small_chunk_compile_and_channels_last_match_eager \
	tests/base_models/diffusion_model/test_hy_nomas_cuda.py \
	tests/runtime/test_inference_benchmark_correctness.py::test_denoise_benchmark_checks_every_step_before_timing \
	tests/runtime/test_inference_benchmark_correctness.py::test_full_stack_actual_fp8_cache_or_graph_execution \
	tests/core/execution/test_cuda_nvenc_transport.py \
	tests/core/execution/test_cuda_nvenc_generation.py \
	tests/synthesis/test_lingbot_cuda_frames.py \
	tests/studio_visualization/test_realtime_nvenc.py::test_abgr_conversion_channel_order_and_alpha \
	tests/core/acceleration/test_fused_fp8_ffn.py::test_fused_gelu_quantization_matches_materialized_activation \
	tests/core/acceleration/test_fused_fp8_ffn.py::test_fused_ffn_real_fp8_graph_and_policy_fallback \
	tests/core/acceleration/test_fused_fp8_ffn.py::test_fused_ffn_empty_input_and_invalid_activation \
	tests/core/acceleration/test_fused_fp8_ffn.py::test_fused_ffn_inductor_preserves_real_fp8_math \
	tests/core/acceleration/test_convolution_layout.py::test_mixed_layout_inductor_preserves_decode_pixels \
	tests/synthesis/test_matrix_game_2_optimizations.py::test_nondefault_stream_returns_correct_results_without_device_sync \
	tests/synthesis/test_matrix_game_2_decode_overlap.py::test_decode_overlap_joins_the_nondefault_caller_and_preserves_recurrent_cache_and_lifetimes \
	tests/synthesis/test_matrix_game_2_decode_overlap.py::test_reset_drains_and_releases_only_the_owned_decode_stream \
	tests/synthesis/test_matrix_game_2_decode_overlap.py::test_failed_overlap_drains_enqueued_decode_and_invalidates_the_session \
	tests/synthesis/test_matrix_game_2_decode_overlap.py::test_overlap_profile_measures_each_stream_and_the_joined_wall_time \
	tests/core/attention/test_native_cudnn_fp8_cuda.py \
	tests/core/attention/test_triton_tma_cuda.py \
	tests/core/attention/test_block_kv_cuda_graph.py \
	tests/core/attention/test_cuda_range_attention.py \
	tests/core/test_cuda_gaussian_surfels.py \
	tests/base_models/test_cuda_sd2_marigold_components.py \
	tests/synthesis/test_cuda_dreamdojo_student.py \
	tests/core/utils/tensors/test_running_mean_std.py::test_running_mean_std_preserves_cuda_device_and_dtype \
	tests/core/utils/tensors/test_torch_utils.py::test_cuda_checkpoint_roundtrip_restores_the_weight_owner \
	tests/core/utils/tensors/test_torch_utils.py::test_multi_index_select_matches_cuda_indexing \
	tests/base_models/diffusion_model/test_wan_execution_preparation.py::test_cuda_prepared_wan_replays_new_conditions_without_retaining_prior_request \
	tests/base_models/diffusion_model/test_matrix_game4.py::test_cuda_bf16_x0_uses_released_sigma_grid_and_clean_commits \
	tests/base_models/diffusion_model/test_request_batching.py::test_cuda_batch_warmup_freezes_capture_and_preserves_output_ownership \
	tests/base_models/diffusion_model/test_image_request_batching.py::test_cuda_image_batch_warmup_replays_changed_images \
	tests/base_models/diffusion_model/test_image_request_batching.py::test_cuda_attention_preserves_request_shaped_split_kv_reductions \
	tests/base_models/diffusion_model/test_wan_execution_preparation.py::test_cuda_regional_compile_preserves_bfloat16_pointwise_rounding \
	tests/studio_visualization/test_realtime_delivery.py::test_device_frame_receipt_keeps_cuda_storage \
	tests/synthesis/test_lingbot_realtime_cache_cursors.py \
	tests/synthesis/test_lingbot_i2v_condition_reuse.py \
	tests/synthesis/test_lingbot_static_inputs.py
INFER_CUDA_REPORT ?= tmp/inference-cuda-contracts.xml
INFER_SOL_CUDA_CONTRACTS = \
	tests/core/test_sol_attention.py::test_sol_cuda_executes_with_receipt_and_validates_shape \
	tests/core/test_sol_attention.py::test_uncompressed_cuda_path_matches_upstream_sink_and_preserves_scale
INFER_SOL_CUDA_REPORT ?= tmp/inference-sol-cuda-contracts.xml
LIGHTVAE21_CHECKPOINT ?= $(WORLDFOUNDRY_LIGHTVAE21_CHECKPOINT)
LIGHTVAE21_CHECKPOINT_JUNIT ?= tmp/lightvae-checkpoint-contracts.xml
MG2_CHECKPOINT_ROOT ?=
MG2_CHECKPOINT_REPORT ?= tmp/mg2-checkpoint-contracts.json
MG2_CHECKPOINT_JUNIT ?= tmp/mg2-checkpoint-contracts.xml
MG2_TRAJECTORY_REPORT ?= tmp/mg2-trajectory-contracts.json
MG2_TRAJECTORY_JUNIT ?= tmp/mg2-trajectory-contracts.xml
MG2_CONDITIONED_TRAJECTORY_REPORT ?= tmp/mg2-conditioned-trajectory-contracts.json
MG2_CONDITIONED_TRAJECTORY_JUNIT ?= tmp/mg2-conditioned-trajectory-contracts.xml
GEOMETRY_MATRIX ?= tests/manual/geometry_regression_cases.json
GEOMETRY_REFERENCE ?=
GEOMETRY_CANDIDATE ?=
GEOMETRY_REPORT ?= tmp/3d-regression-gate.json
GEOMETRY_BASE ?=
GEOMETRY_DEPENDENCIES ?= tests/manual/geometry_regression_dependencies.json
GEOMETRY_PLAN ?= tmp/3d-regression-plan.json
GEOMETRY_PROFILE ?=
INFERENCE_BASE ?= $(GEOMETRY_BASE)
INFERENCE_PLAN ?= tmp/inference-regression-plan.json
INFERENCE_PROFILE ?= $(GEOMETRY_PROFILE)
INFERENCE_REPORT ?=
INFERENCE_COVERAGE ?= tmp/inference-regression-coverage.json
INFERENCE_GATE_OUTPUT ?= tmp/inference-regression-gate.json
RELEASE_HFD_ROOT ?= $(if $(WORLDFOUNDRY_HFD_ROOT),$(WORLDFOUNDRY_HFD_ROOT),$(HOME)/.cache/worldfoundry/checkpoints/hfd)
CANONICAL_DIFFUSION_SOURCES ?= \
	worldfoundry/base_models/diffusion_model/*.py \
	worldfoundry/base_models/diffusion_model/extensions \
	worldfoundry/base_models/diffusion_model/loaders \
	worldfoundry/base_models/diffusion_model/models \
	worldfoundry/base_models/diffusion_model/optimizations \
	worldfoundry/base_models/diffusion_model/recipes \
	worldfoundry/base_models/diffusion_model/runners \
	worldfoundry/base_models/diffusion_model/schedulers
# Ruff reads tracked Python sources from Git's index: a directory walk would also visit local
# license-gated runtimes and skip tracked sources below ignored names such as checkpoints/.
RUFF_SOURCES ?= worldfoundry scripts/model_zoo
RUFF_FORMAT_SOURCES ?= worldfoundry
ruff_tracked_files = git ls-files -z -- $(foreach path,$(1),'$(path)/*.py' '$(path)/*.pyi' '$(path)/*.ipynb')

help:
	@printf '%s\n' \
		'WorldFoundry development targets:' \
		'  make install-core      Install the editable core package.' \
		'  make install-dev       Install lightweight development dependencies.' \
		'  make test              Run the public CPU inference and packaging gate.' \
		'  make test-docs-contracts  Validate published CLI commands, catalog claims and documentation contracts; rejects skips.' \
		'  make test-infer        Alias for the public CPU gate.' \
		'  make test-infer-tensors  Test small checkpoint, geometry and execution tensors with CPU Torch and safetensors.' \
		'  make test-infer-video-tensors  Check real small video operators, sampler math and resident request isolation with CPU Torch and einops.' \
		'  make test-infer-contracts  Test checkpoint, geometry, serializer and execution contracts in a model environment.' \
		'  make test-infer-cuda-contracts  Run CUDA/FP8 graph, cache, attention and transport checks on SM90+; requires Torch, Triton, nvidia-cudnn-frontend and cuda-bindings; rejects skips.' \
		'  make test-infer-sol-cuda-contracts  Require real compressed and uncompressed Sol-Attn execution in its supported CUDA environment; rejects skips.' \
		'  make test-lightvae-checkpoint-contracts  Validate matched student encode/decode and causal caches with LIGHTVAE21_CHECKPOINT on CPU and CUDA; rejects skips.' \
		'  make test-mg2-checkpoint-contracts  Validate real MG2 block/cache, FP8 FFN and streaming VAE pixels using MG2_CHECKPOINT_ROOT; no generation quality or speed certification.' \
		'  make test-mg2-trajectory-contracts  Require bitwise parity of full MG2 denoise/refresh, rolling caches and 45 decoded frames for packed/split QKV and decode overlap.' \
		'  make test-mg2-conditioned-trajectory-contracts  Check real image VAE/CLIP conditioning and six 141-frame bitwise trajectories using local MG2 weights.' \
		'  make test-geometry     Audit real 3D replays against accepted references; requires GEOMETRY_REFERENCE and GEOMETRY_CANDIDATE.' \
		'  make plan-geometry     Select affected short 3D cases from GEOMETRY_BASE to HEAD without GPU/weights.' \
		'  make replay-geometry   Replay selected or all cases using a private GEOMETRY_PROFILE; optional GEOMETRY_REPLAY_PLAN.' \
		'  make plan-inference    Select affected 3D/video/world cases using INFERENCE_BASE.' \
		'  make replay-inference  Replay affected cases using INFERENCE_PROFILE and INFERENCE_PLAN.' \
		'  make coverage-inference  List short-case definitions and missing video/world variants.' \
		'  make verify-inference  Verify exact committed GPU evidence using INFERENCE_PLAN, INFERENCE_REPORT and INFERENCE_PROFILE.' \
		'  make test-eval-core    Run the extended evaluation contract suite.' \
		'  make docs-check        Verify checked-in generated documentation.' \
		'  make docs-dev-fast     Start docs using existing generated output.' \
		'  make docs-dev-ssd      Start docs with caches on local SSD.' \
		'  make docs-dev-local    Start docs from a local SSD mirror.' \
		'  make docs-build-fast   Build docs without the CI validation gates.' \
		'  make cli-entrypoint-check Validate documented CLI entrypoints.' \
		'  make lint              Run lightweight source and catalog checks.' \
		'  make preflight         Run the public runtime preflight.' \
		'  make pr-check          Check committed PR Python changes and DCO sign-offs; requires BASE_REF.' \
		'  make check-cuda-constraints  Verify CUDA-tier torch constraint stubs.' \
		'  make packaging-check   Audit package discovery and license-gated wheel content.'

install-core:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest "$(PUBLIC_TEST_MANIFEST)" --junit "$(PUBLIC_TEST_JUNIT)" --include-public-defaults $(foreach test,$(PUBLIC_GATE_CONTRACTS),--test $(test)) -- $(TEST_ARGS)

test-infer: test

test-docs-contracts:
	PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest "$(DOCS_TEST_MANIFEST)" --junit "$(DOCS_TEST_JUNIT)" --test tests/core/test_model_docs_codegen.py --test tests/eval_core/test_docs_quickstart.py --test tests/eval_core/test_benchmark_hub_docs.py -- $(TEST_ARGS)

.PHONY: test-realtime-client test-evaluation
test-realtime-client:
	node --test tests/studio_visualization/realtime_client.test.mjs tests/studio_visualization/realtime_udp.test.mjs

# This suite needs CPU Torch, safetensors and einops, with no weights or renderer.
test-evaluation:
	CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest tmp/evaluation-cpu-selection.json --junit tmp/evaluation-cpu-contracts.xml --marker 'not gpu' $(foreach test,$(EVALUATION_CPU_CONTRACTS),--test $(test)) -- $(TEST_ARGS)

test-infer-tensors:
	CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest "$(INFER_TENSOR_MANIFEST)" --junit "$(INFER_TENSOR_JUNIT)" --marker 'not gpu' $(foreach test,$(INFER_TENSOR_CONTRACTS),--test $(test)) -- $(TEST_ARGS)

test-infer-video-tensors:
	CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest tmp/inference-video-selection.json --junit tmp/inference-video-contracts.xml --marker 'not gpu' $(foreach test,$(VIDEO_TENSOR_CONTRACTS),--test $(test)) -- $(TEST_ARGS)

plan-inference:
	@test -n "$(INFERENCE_BASE)" || (echo 'Set INFERENCE_BASE to the comparison commit.'; exit 2)
	$(PYTHON) tests/manual/geometry_regression_impact.py --matrix "$(GEOMETRY_MATRIX)" --dependencies "$(GEOMETRY_DEPENDENCIES)" --base "$(INFERENCE_BASE)" --output "$(INFERENCE_PLAN)"

replay-inference:
	@test -n "$(INFERENCE_PROFILE)" || (echo 'Set INFERENCE_PROFILE to a private prepared model-host profile.'; exit 2)
	$(PYTHON) tests/manual/geometry_regression_suite.py --profile "$(INFERENCE_PROFILE)" --plan "$(INFERENCE_PLAN)"

coverage-inference:
	$(PYTHON) tests/manual/inference_regression_coverage.py --matrix "$(GEOMETRY_MATRIX)" $(if $(INFERENCE_BASE),--base "$(INFERENCE_BASE)",) --output "$(INFERENCE_COVERAGE)"

verify-inference:
	@test -n "$(INFERENCE_PROFILE)" -a -n "$(INFERENCE_REPORT)" || (echo 'Set INFERENCE_PROFILE and INFERENCE_REPORT.'; exit 2)
	$(PYTHON) tests/manual/inference_regression_gate.py --plan "$(INFERENCE_PLAN)" --report "$(INFERENCE_REPORT)" --profile "$(INFERENCE_PROFILE)" --output "$(INFERENCE_GATE_OUTPUT)"

# Run inside the 3D model environment; tensor and serializer tests need its dependencies.
test-infer-contracts:
	PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest tmp/inference-model-selection.json --junit tmp/inference-model-contracts.xml --marker 'not gpu' $(foreach test,$(INFER_TENSOR_CONTRACTS),--test $(test)) --test tests/pipelines/test_geometry_result_exports.py -- $(TEST_ARGS)

# Run manually on SM90+ with CUDA Torch, Triton, nvidia-cudnn-frontend and
# cuda-bindings installed; hosted CI has no GPU runner.
# An unavailable GPU, empty selection or skipped contract must fail this gate.
test-infer-cuda-contracts:
	$(PYTHON) -c 'import sys, torch; torch.cuda.is_available() or sys.exit("CUDA contract tests require an available GPU")'
	PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest tmp/inference-cuda-selection.json --junit "$(INFER_CUDA_REPORT)" $(foreach test,$(INFER_CUDA_CONTRACTS),--test $(test)) -- $(TEST_ARGS)

test-infer-sol-cuda-contracts:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c 'import sys; from worldfoundry.core.attention.backends.probe import probe_attention_backends; probe_attention_backends("cuda")["sol_attn"].usable or sys.exit("Sol-Attn contracts require its supported CUDA runtime and provider")'
	PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest tmp/inference-sol-cuda-selection.json --junit "$(INFER_SOL_CUDA_REPORT)" $(foreach test,$(INFER_SOL_CUDA_CONTRACTS),--test $(test)) -- $(TEST_ARGS)

test-lightvae-checkpoint-contracts:
	@test -f "$(LIGHTVAE21_CHECKPOINT)" || { printf '%s\n' 'Set LIGHTVAE21_CHECKPOINT to the matched Wan2.1 student checkpoint.' >&2; exit 2; }
	$(PYTHON) -c 'import sys, torch; torch.cuda.is_available() or sys.exit("LightVAE checkpoint contracts require CPU and CUDA execution")'
	OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WORLDFOUNDRY_LIGHTVAE21_CHECKPOINT="$(LIGHTVAE21_CHECKPOINT)" PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest tmp/lightvae-checkpoint-selection.json --junit "$(LIGHTVAE21_CHECKPOINT_JUNIT)" --test tests/base_models/diffusion_model/models/autoencoders/test_lightvae_wan21.py --test tests/base_models/diffusion_model/optimizations/test_lightvae_fp8.py -- $(TEST_ARGS)

# Local checkpoint evidence is opt-in and independent of the public small-tensor gate.
test-mg2-checkpoint-contracts:
	@test -n "$(MG2_CHECKPOINT_ROOT)" || { printf '%s\n' 'Set MG2_CHECKPOINT_ROOT to existing local Matrix-Game-2.0 weights.' >&2; exit 2; }
	$(PYTHON) -c 'import torch; assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9, "Checkpoint FP8 contracts require an SM90+ CUDA device"'
	WORLDFOUNDRY_MG2_CHECKPOINT_ROOT="$(MG2_CHECKPOINT_ROOT)" WORLDFOUNDRY_MG2_CHECKPOINT_REPORT="$(MG2_CHECKPOINT_REPORT)" PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest tmp/mg2-checkpoint-selection.json --junit "$(MG2_CHECKPOINT_JUNIT)" --test tests/synthesis/test_matrix_game_2_checkpoint_optimizations.py -- $(TEST_ARGS)
	$(PYTHON) -c 'import sys, xml.etree.ElementTree as ET; cases = list(ET.parse(sys.argv[1]).getroot().iter("testcase")); assert len(cases) == 3 and all(all(case.find(tag) is None for tag in ("skipped", "failure", "error")) for case in cases), "Every checkpoint operator contract must execute and pass"; print("Checkpoint gate: 3 operator contracts passed without skips")' "$(MG2_CHECKPOINT_JUNIT)"

test-mg2-trajectory-contracts:
	@test -n "$(MG2_CHECKPOINT_ROOT)" || { printf '%s\n' 'Set MG2_CHECKPOINT_ROOT to existing local Matrix-Game-2.0 weights.' >&2; exit 2; }
	$(PYTHON) -c 'import torch; assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9, "Full trajectory contracts require an SM90+ CUDA device"'
	WORLDFOUNDRY_MG2_CHECKPOINT_ROOT="$(MG2_CHECKPOINT_ROOT)" WORLDFOUNDRY_MG2_TRAJECTORY_REPORT="$(MG2_TRAJECTORY_REPORT)" PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest tmp/mg2-trajectory-selection.json --junit "$(MG2_TRAJECTORY_JUNIT)" --test tests/synthesis/test_matrix_game_2_checkpoint_trajectory.py -- $(TEST_ARGS)
	$(PYTHON) -c 'import sys, xml.etree.ElementTree as ET; cases = list(ET.parse(sys.argv[1]).getroot().iter("testcase")); assert len(cases) == 2 and all(all(case.find(tag) is None for tag in ("skipped", "failure", "error")) for case in cases), "Both full-checkpoint trajectories must execute and pass"; print("Trajectory gate: 2 bitwise contracts passed without skips")' "$(MG2_TRAJECTORY_JUNIT)"

test-mg2-conditioned-trajectory-contracts:
	@test -n "$(MG2_CHECKPOINT_ROOT)" || { printf '%s\n' 'Set MG2_CHECKPOINT_ROOT to existing local Matrix-Game-2.0 weights.' >&2; exit 2; }
	$(PYTHON) -c 'import torch; assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9, "Conditioned trajectory contracts require an SM90+ CUDA device"'
	OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WORLDFOUNDRY_MG2_CHECKPOINT_ROOT="$(MG2_CHECKPOINT_ROOT)" WORLDFOUNDRY_MG2_CONDITIONED_TRAJECTORY_REPORT="$(MG2_CONDITIONED_TRAJECTORY_REPORT)" WORLDFOUNDRY_MATRIX_REALTIME_CONDITION_BLOCKS=5 WORLDFOUNDRY_MATRIX_REALTIME_WARMUP_DECODER=1 PYTHONPATH=$(PYTHONPATH) $(STRICT_TEST_RUNNER) --manifest tmp/mg2-conditioned-selection.json --junit "$(MG2_CONDITIONED_TRAJECTORY_JUNIT)" --test tests/synthesis/test_matrix_game_2_checkpoint_conditioned_trajectory.py -- $(TEST_ARGS)
	$(PYTHON) -c 'import sys, xml.etree.ElementTree as ET; cases = list(ET.parse(sys.argv[1]).getroot().iter("testcase")); assert len(cases) == 6 and all(all(case.find(tag) is None for tag in ("skipped", "failure", "error")) for case in cases), "All six real-image checkpoint trajectories must execute and pass"; print("Conditioned trajectory gate: 6 bitwise contracts passed without skips")' "$(MG2_CONDITIONED_TRAJECTORY_JUNIT)"

test-geometry:
	@test -n "$(GEOMETRY_REFERENCE)" -a -n "$(GEOMETRY_CANDIDATE)" || { printf '%s\n' 'Set GEOMETRY_REFERENCE and GEOMETRY_CANDIDATE to completed real-checkpoint run directories.' >&2; exit 2; }
	$(PYTHON) tests/manual/geometry_regression.py audit --matrix "$(GEOMETRY_MATRIX)" --reference "$(GEOMETRY_REFERENCE)" --candidate "$(GEOMETRY_CANDIDATE)" --source-root "$(CURDIR)" --report "$(GEOMETRY_REPORT)"

.PHONY: plan-geometry replay-geometry
plan-geometry:
	@test -n "$(GEOMETRY_BASE)" || { printf '%s\n' 'Set GEOMETRY_BASE to the comparison commit or ref.' >&2; exit 2; }
	$(PYTHON) tests/manual/geometry_regression_impact.py --matrix "$(GEOMETRY_MATRIX)" --dependencies "$(GEOMETRY_DEPENDENCIES)" --source-root "$(CURDIR)" --base "$(GEOMETRY_BASE)" --output "$(GEOMETRY_PLAN)"

replay-geometry:
	@test -n "$(GEOMETRY_PROFILE)" || { printf '%s\n' 'Set GEOMETRY_PROFILE to the private host profile.' >&2; exit 2; }
	$(PYTHON) tests/manual/geometry_regression_suite.py --profile "$(GEOMETRY_PROFILE)" $(if $(GEOMETRY_REPLAY_PLAN),--plan "$(GEOMETRY_REPLAY_PLAN)",)

test-eval-core:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests/eval_core $(TEST_ARGS)

docs-check:
	npm --prefix docs/fumadocs run models:check
	npm --prefix docs/fumadocs run models:homes:check
	npm --prefix docs/fumadocs run benchmarks:check
	npm --prefix docs/fumadocs run coverage:check

docs-dev-fast:
	npm --prefix docs/fumadocs run dev:fast

docs-dev-ssd:
	npm --prefix docs/fumadocs run dev:ssd

docs-dev-local:
	npm --prefix docs/fumadocs run dev:local

docs-build-fast:
	npm --prefix docs/fumadocs run build:fast

cli-entrypoint-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli --help >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo models --json >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo benchmarks --json >/dev/null

lint: ruff-check format-check shell-check data-check runtime-registry-check workspace-registry-check

ruff-check:
	$(call ruff_tracked_files,$(RUFF_SOURCES)) | xargs -0 -r $(PYTHON) -m ruff check --force-exclude

format-check:
	$(call ruff_tracked_files,$(RUFF_FORMAT_SOURCES)) | xargs -0 -r $(PYTHON) -m ruff format --check --force-exclude
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q $(CANONICAL_DIFFUSION_SOURCES) worldfoundry/evals scripts

# BeeGFS: enumerate the bounded tracked paths from Git's index, not a recursive filesystem walk.
shell-check:
	@set -eu; \
	git ls-files 'scripts/setup/*.sh' 'scripts/dev/*.sh' 'docs/fumadocs/scripts/*.sh' | \
	while IFS= read -r script; do \
		if [ -f "$$script" ]; then bash -n "$$script"; fi; \
	done

data-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo models --json >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo benchmarks --json >/dev/null

runtime-registry-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c 'from worldfoundry.evals.models.validate import validate_runtime_registry; errors = [issue for issue in validate_runtime_registry() if issue.severity == "error"]; assert not errors, "\\n".join(f"[{issue.code}] {issue.message}" for issue in errors)'

workspace-registry-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c 'from worldfoundry.evals.tasks.catalog.integrity import catalog_runner_table_issues, video_catalog_dispatch_issues; from worldfoundry.evals.tasks.catalog.workspace_registry import validate_workspace_registry; issues = [*validate_workspace_registry(), *video_catalog_dispatch_issues(), *catalog_runner_table_issues()]; assert not issues, "\n".join(issues); print("workspace registry OK")'

check-cuda-constraints:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/setup/check_cuda_torch_constraints.py

# Set WHEEL=... and/or SDIST=... to audit built distribution contents.
packaging-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/setup/check_packaging_license_gate.py $(if $(WHEEL),--wheel $(WHEEL),) $(if $(SDIST),--sdist $(SDIST),)

compile-eval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q worldfoundry/evals scripts

cli-check:
	rm -rf $(CLI_CHECK_OUTPUT)
	mkdir -p $(CLI_CHECK_OUTPUT)/input
	printf '%s\n' '{"sample_id":"ci-0001","status":"success","artifacts":{"video":{"uri":"$(CLI_CHECK_OUTPUT)/input/demo.mp4","kind":"video"}}}' > $(CLI_CHECK_OUTPUT)/input/results.jsonl
	: > $(CLI_CHECK_OUTPUT)/input/demo.mp4
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli evaluate \
		--mode existing-results \
		--results-path $(CLI_CHECK_OUTPUT)/input/results.jsonl \
		--output-dir $(CLI_CHECK_OUTPUT)/run \
		--benchmark-id ci-existing-results \
		--model-id ci-package-check \
		--metric artifact_count \
		--required-artifact video \
		--json

pr-check:
	@test -n "$(BASE_REF)" || (echo 'Set BASE_REF to the PR comparison commit.'; exit 2)
	$(PYTHON) scripts/dev/check_python_changes.py --base "$(BASE_REF)" --head "$(HEAD_REF)"
	$(PYTHON) scripts/dev/check_dco.py --base "$(BASE_REF)" --head "$(HEAD_REF)"

precommit:
	$(PRE_COMMIT) run -a

precommit-install:
	$(PRE_COMMIT) install

preflight:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli preflight runtime \
		--profile $(PREFLIGHT_PROFILE) \
		--output-dir $(PREFLIGHT_OUTPUT) \
		--json
