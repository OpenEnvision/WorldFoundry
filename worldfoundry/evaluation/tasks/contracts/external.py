"""External Benchmark Definitions and Contracts Registry.

This module statically catalogs all third-party, simulation-backed, or independently evaluated
benchmarks supported by the WorldFoundry framework. It defines `ExternalBenchmarkContract` instances
for each track (e.g. VBench, RoboTwin, WorldBench), explicitly enumerating:
1. `input_keys`: The expected data and state assets passed into the runtime.
2. `output_keys`: The structural outputs generated post-evaluation (e.g., scorecards, raw logs).
3. `metric_ids`: The valid scoring/metric fields tracked by the framework.

By registering these contracts globally, the framework can dynamically resolve missing evaluation
dependencies and enforce strict input/output shapes across dozens of external benchmark suites.
"""

from __future__ import annotations

from pathlib import Path

from .registry import (
    ExternalBenchmarkContract,
    ExternalBenchmarkContractRegistry,
    UnknownExternalBenchmarkContractError,
    register_external_benchmark_contract,
)


ROBOTWIN_BENCHMARK_ID = "robotwin"
ROBOTWIN_BENCHMARK_NAME = "RoboTwin 2.0"
ROBOTWIN_ROOT_ENV = "WORLDFOUNDRY_ROBOTWIN_ROOT"
ROBOTWIN_RESULTS_PATH_ENV = "WORLDFOUNDRY_ROBOTWIN_RESULTS_PATH"
ROBOTWIN_DEFAULT_ROOT = Path("thirdparty") / "RoboTwin"
ROBOTWIN_INPUT_KEYS = (
    "dataset_root",
    "simulator_root",
    "task_manifest",
    "policy_results_path",
    "official_results_path",
    "robotwin_eval_result_dir",
)
ROBOTWIN_OUTPUT_KEYS = (
    "scorecard",
    "raw_results",
    "raw_metric_table",
    "per_sample_metrics",
    "rollout_logs",
    "videos",
)
ROBOTWIN_METRIC_IDS = (
    "success_rate",
    "task_success",
    "episode_success",
    "bimanual_task_success",
    "domain_randomization_success",
    "clean_success",
    "randomized_success",
    "reward",
)
ROBOTWIN_SUCCESS_METRIC_IDS = (
    "success_rate",
    "task_success",
    "episode_success",
    "bimanual_task_success",
    "domain_randomization_success",
    "clean_success",
    "randomized_success",
)
ROBOTWIN_SUCCESS_METRICS = frozenset(ROBOTWIN_SUCCESS_METRIC_IDS)
ROBOTWIN_METRIC_GROUPS = {
    "robotwin_core": ("success_rate", "task_success", "episode_success"),
    "robotwin_bimanual": ("bimanual_task_success",),
    "robotwin_domain_randomization": (
        "domain_randomization_success",
        "clean_success",
        "randomized_success",
    ),
    "robotwin_reward": ("reward",),
}
ROBOTWIN_BENCHMARK_KIND = ("vla", "embodied-action", "robot-manipulation", "bimanual")
ROBOTWIN_TASK_CONFIGS = ("demo_clean", "demo_randomized")
ROBOTWIN_CAMERA_PROFILES = ("D435", "Large_D435", "L515", "Large_L515")
ROBOTWIN_DEFAULT_EMBODIMENT = "aloha-agilex"
ROBOTWIN_SUPPORTED_RESULT_IMPORT_FORMATS = ("_result.txt", "json", "jsonl", "csv", "tsv")
ROBOTWIN_CONTRACT_NOTES = (
    "Contract covers RoboTwin 2.0 dual-arm embodied manipulation benchmarks for VLA / embodied-action policies.",
    "RoboTwin is not a video-generation benchmark; WorldFoundry treats it as simulator-backed action evaluation.",
    "Large datasets, simulator assets, and policy execution remain explicit acquisition/runtime stages.",
)


def build_robotwin_contract_payload(benchmark_id: str = ROBOTWIN_BENCHMARK_ID) -> dict[str, object]:
    """Generates the structured contract definition for RoboTwin simulator evaluation tasks.

    Args:
        benchmark_id: Canonical identifier for the RoboTwin benchmark.

    Returns:
        A dictionary payload representing the RoboTwin contract configuration.
    """
    return {
        "benchmark_id": benchmark_id,
        "display_name": ROBOTWIN_BENCHMARK_NAME,
        "input_keys": list(ROBOTWIN_INPUT_KEYS),
        "output_keys": list(ROBOTWIN_OUTPUT_KEYS),
        "metric_ids": list(ROBOTWIN_METRIC_IDS),
        "requires_upstream_runtime": True,
        "notes": list(ROBOTWIN_CONTRACT_NOTES),
    }


VBenchContract = ExternalBenchmarkContract(
    benchmark_id="vbench",
    display_name="VBench",
    input_keys=("prompt_suite_json", "generated_video_dir"),
    output_keys=("scorecard", "dimension_scores", "raw_metric_table"),
    metric_ids=(
        "overall_quality",
        "temporal_quality",
        "frame_quality",
        "text_alignment",
        "subject_consistency",
        "background_consistency",
        "temporal_flickering",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
        "object_class",
        "multiple_objects",
        "human_action",
        "color",
        "spatial_relationship",
        "scene",
        "appearance_style",
        "temporal_style",
        "overall_consistency",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers generated-video input and normalized scorecard output.",
        "Heavy upstream dimensions remain behind the official VBench runtime.",
    ),
)

VBenchPlusPlusContract = ExternalBenchmarkContract(
    benchmark_id="vbench-plus-plus",
    display_name="VBench++",
    input_keys=("generated_video_dir", "prompt_suite_json", "reference_image_dir", "official_results_path"),
    output_keys=("scorecard", "dimension_scores", "raw_metric_table"),
    metric_ids=(
        "vbench_plus_plus_i2v_average",
        "vbench_plus_plus_long_average",
        "vbench_plus_plus_trustworthiness_average",
        "vbench_plus_plus_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers VBench++ I2V, long-video, and trustworthiness variants through one scorecard surface.",
        "Official runtime remains variant-specific and requires the upstream VBench dependency stack.",
    ),
)

VBench2Contract = ExternalBenchmarkContract(
    benchmark_id="vbench-2.0",
    display_name="VBench-2.0",
    input_keys=("generated_video_dir", "prompt_suite_json", "official_results_path"),
    output_keys=("scorecard", "dimension_scores", "raw_metric_table"),
    metric_ids=(
        "vbench2_creativity",
        "vbench2_commonsense",
        "vbench2_controllability",
        "vbench2_human_fidelity",
        "vbench2_physics",
        "vbench2_total",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps official VBench-2.0 dimension outputs into the five published category groups.",
        "Full runtime validation requires the upstream VBench-2.0 environment and checkpoint cache.",
    ),
)

VideoBenchContract = ExternalBenchmarkContract(
    benchmark_id="video-bench",
    display_name="Video-Bench",
    input_keys=("generated_video_dir", "videobench_prompt_suite_json", "judge_api_config"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores"),
    metric_ids=(
        "imaging_quality",
        "aesthetic_quality",
        "temporal_consistency",
        "motion_effects",
        "video_text_consistency",
        "object_class_consistency",
        "color_consistency",
        "action_consistency",
        "scene_consistency",
        "videobench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps official Video-Bench GPT-4o/GPT-4o-mini judge outputs to WorldFoundry scorecard.",
        "Runtime execution requires explicit API credentials; offline normalization can be used for existing official score JSON snapshots.",
    ),
)

WorldBenchContract = ExternalBenchmarkContract(
    benchmark_id="worldbench",
    display_name="WorldBench",
    input_keys=(
        "official_results_path",
        "worldbench_dataset_root",
        "generated_video_dir",
        "answer_manifest",
    ),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "worldbench_evaluation"),
    metric_ids=(
        "foreground_miou",
        "foreground_dice",
        "background_rmse",
        "text_based_accuracy",
        "multiple_choice_accuracy",
        "binary_accuracy",
    ),
    requires_upstream_runtime=False,
    notes=(
        "In-tree runtime evaluates raw generated continuations with the shared WorldFoundry SAM2 base model.",
        "Existing WorldBench result files remain supported through the compatibility normalizer.",
        "The public release computes Dice while naming it mIoU, so the contract reports conventional IoU and release-compatible Dice separately.",
    ),
)

PhysicalAIBenchContract = ExternalBenchmarkContract(
    benchmark_id="physical-ai-bench",
    display_name="Physical AI Bench (PAI-Bench)",
    input_keys=(
        "official_results_path",
        "dataset_root",
        "generated_video_dir",
        "prediction_manifest",
        "reference_image_dir",
        "precomputed_depth_dir",
        "precomputed_segmentation_dir",
    ),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "official_results"),
    metric_ids=(
        "aesthetic_quality",
        "background_consistency",
        "imaging_quality",
        "motion_smoothness",
        "overall_consistency",
        "subject_consistency",
        "i2v_background",
        "i2v_subject",
        "vqa_accuracy",
        "dover_tech_score",
        "blur_ssim",
        "canny_f1_score",
        "canny_precision",
        "canny_recall",
        "depth_si_rmse",
        "seg_m_iou",
        "seg_recall",
        "lpips_diversity",
    ),
    requires_upstream_runtime=False,
    notes=(
        "Generation and conditional-generation evaluation are integrated independently in tree.",
        "VBench, VBench++ I2V, Qwen-VL, DOVER, LPIPS, Video-Depth-Anything, GroundingDINO, and SAM2 are reused from WorldFoundry base/runtime modules.",
        "No upstream checkout or third_party directory is executed at runtime.",
    ),
)

WBenchContract = ExternalBenchmarkContract(
    benchmark_id="wbench",
    display_name="WBench",
    input_keys=("generated_video_dir", "wbench_case_dir", "official_results_path", "wbench_weights_dir"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "report_json"),
    metric_ids=(
        "aesthetic_quality",
        "imaging_quality",
        "temporal_flickering",
        "dynamic_degree",
        "motion_smoothness",
        "hpsv3_quality",
        "scene_adherence",
        "subject_adherence",
        "navigation_trajectory",
        "event_edit_adherence",
        "subject_action_adherence",
        "perspective_switch_adherence",
        "background_consistency",
        "segment_continuity",
        "perspective_consistency",
        "subject_consistency",
        "geometric_consistency",
        "photometric_consistency",
        "spatial_consistency",
        "gated_spatial_consistency",
        "visual_plausibility",
        "causal_fidelity",
        "quality_score",
        "setting_score",
        "interaction_score",
        "consistency_score",
        "physical_score",
        "wbench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers WBench multi-turn interactive video world model evaluation across 22 official metrics.",
        "WorldFoundry keeps the official metric runtime in-tree and can also normalize existing WBench report.json outputs.",
    ),
)

FourDWorldBenchContract = ExternalBenchmarkContract(
    benchmark_id="4dworldbench",
    display_name="4DWorldBench",
    input_keys=("dataset_json", "generated_video_dir", "model_name", "dimension", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "dimension_result_json"),
    metric_ids=(
        "perceptual_clip_iqa_metrics",
        "perceptual_clip_aesthetic_metrics",
        "perceptual_fastvqa",
        "alignment_attribute_control",
        "alignment_relationship_control",
        "alignment_motion_control",
        "alignment_event_control",
        "alignment_scene_control",
        "alignment_camera_error_metrics",
        "physics_realism",
        "consistency_viewpoint",
        "consistency_motion_smoothness",
        "consistency_motion_qa",
        "consistency_style",
        "perceptual_quality",
        "condition_4d_alignment",
        "physical_realism_score",
        "four_d_consistency",
        "four_d_worldbench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers the official 4DWorldBench world-generation dimensions for perceptual quality, Condition-4D alignment, physical realism, and 4D consistency.",
        "WorldFoundry keeps the runner and metric entrypoints in-tree; heavy judges and geometry/video-quality backbones are resolved through WorldFoundry base_models or explicit environment variables.",
    ),
)

IWorldBenchContract = ExternalBenchmarkContract(
    benchmark_id="iworld-bench",
    display_name="iWorld-Bench",
    input_keys=("generated_video_dir", "official_results_path", "camera_trajectory_dir", "source_npz_dir"),
    output_keys=("scorecard", "raw_metric_table", "reports", "generated_video"),
    metric_ids=(
        "image_quality",
        "brightness_consistency",
        "color_temperature_constraint",
        "sharpness_retention",
        "motion_smoothness",
        "trajectory_accuracy",
        "trajectory_tolerance",
        "memory_symmetry",
        "trajectory_alignment",
        "iworldbench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps official iWorld-Bench report CSV/JSON files to WorldFoundry scorecards.",
        "WorldFoundry vendors the official iWorld-Bench metric runtime in-tree; full execution still needs generated videos, dataset metadata, and the VBench/VIPe checkpoint assets.",
    ),
)

WorldModelBenchContract = ExternalBenchmarkContract(
    benchmark_id="worldmodelbench",
    display_name="WorldModelBench",
    input_keys=("question_manifest", "generated_video_dir", "judge_model_path"),
    output_keys=("scorecard", "judge_responses", "raw_metric_table"),
    metric_ids=("instruction_following", "common_sense", "physical_adherence", "world_model_average"),
    requires_upstream_runtime=True,
    notes=(
        "Contract records 7-domain/56-subdomain judging as a normalized scorecard surface.",
        "Leaderboard answers and judge checkpoints must be handled by the official runtime.",
    ),
)

WorldScoreContract = ExternalBenchmarkContract(
    benchmark_id="worldscore",
    display_name="WorldScore",
    input_keys=("world_outputs_dir", "camera_trajectory_json", "worldscore_dataset_root"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_metrics"),
    metric_ids=("controllability", "quality", "dynamics", "worldscore_average"),
    requires_upstream_runtime=True,
    notes=(
        "Contract supports video, 3D, and 4D generation outputs through one artifact schema.",
        "Dataset download and upstream metric execution remain explicit validation stages.",
    ),
)


VideoScoreContract = ExternalBenchmarkContract(
    benchmark_id="videoscore",
    display_name="VideoScore",
    input_keys=("video_prompt_manifest", "video_frames_dir", "videoscore_model_repo"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores"),
    metric_ids=(
        "visual_quality",
        "temporal_consistency",
        "dynamic_degree",
        "text_to_video_alignment",
        "factual_consistency",
        "videoscore_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps VideoScore's five human-feedback dimensions to WorldFoundry scorecard.",
        "Official scorer-model inference and benchmark correlations remain explicit validation stages.",
    ),
)


T2VCompBenchContract = ExternalBenchmarkContract(
    benchmark_id="t2v-compbench",
    display_name="T2V-CompBench",
    input_keys=("generated_video_root", "prompt_metadata_dir", "official_csv_dir"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "leaderboard_csv_manifest"),
    metric_ids=(
        "consistent_attribute_binding",
        "dynamic_attribute_binding",
        "spatial_relationships",
        "motion_binding",
        "action_binding",
        "object_interactions",
        "generative_numeracy",
        "t2v_compbench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps the seven official compositional T2V categories to WorldFoundry scorecard.",
        "MLLM, detection, and tracking metrics require separate upstream environments and checkpoints.",
    ),
)


VMBenchContract = ExternalBenchmarkContract(
    benchmark_id="vmbench",
    display_name="VMBench",
    input_keys=("generated_video_dir", "prompt_json", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_metrics"),
    metric_ids=(
        "perceptible_amplitude_score",
        "object_integrity_score",
        "temporal_coherence_score",
        "commonsense_adherence_score",
        "motion_smoothness_score",
        "vmbench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps official VMBench results.json/scores.csv outputs to WorldFoundry scorecard.",
        "Official runtime requires the upstream CUDA dependency stack, checkpoints, and 0001.mp4-style video layout.",
    ),
)

ChronoMagicBenchContract = ExternalBenchmarkContract(
    benchmark_id="chronomagic-bench",
    display_name="ChronoMagic-Bench",
    input_keys=("chronomagic_prompt_manifest", "generated_video_dir", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_metrics"),
    metric_ids=("chronomagic_score", "temporal_transformation"),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps time-lapse/metamorphic T2V outputs to normalized WorldFoundry scorecards.",
        "Official CHScore/evaluation runtime remains a separate validation stage.",
    ),
)

CameraBenchContract = ExternalBenchmarkContract(
    benchmark_id="camerabench",
    display_name="CameraBench",
    input_keys=("generated_video_dir", "camera_prompt_manifest", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "camera_predictions"),
    metric_ids=(
        "camera_motion_average_precision",
        "camera_motion_roc_auc",
        "camera_vqa_accuracy",
        "camera_retrieval_accuracy",
        "camera_caption_score",
        "camerabench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract exposes CameraBench as a camera-motion evaluation surface for generated videos.",
        "The core upstream task is camera-motion understanding, so task-mismatch handling must be explicit.",
    ),
)

EvalCrafterContract = ExternalBenchmarkContract(
    benchmark_id="evalcrafter",
    display_name="EvalCrafter",
    input_keys=("generated_video_dir", "prompt700_txt", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_metrics", "benchmark_contract"),
    metric_ids=(
        "visual_quality",
        "text_video_alignment",
        "motion_quality",
        "temporal_consistency",
        "evalcrafter_total",
        "vqa_aesthetic",
        "vqa_technical",
        "inception_score",
        "clip_temp_score",
        "warping_error",
        "face_consistency_score",
        "action_score",
        "motion_ac_score",
        "flow_score",
        "clip_score",
        "blip_bleu",
        "sd_score",
        "detection_score",
        "color_score",
        "count_score",
        "ocr_error",
        "celebrity_id_error",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps EvalCrafter's 700-prompt T2V evaluation and 17 objective metrics to WorldFoundry scorecard.",
        "Official Docker/conda runtime and checkpoint-heavy metric execution remain a separate validation stage.",
    ),
)

FETVContract = ExternalBenchmarkContract(
    benchmark_id="fetv",
    display_name="FETV",
    input_keys=("fetv_prompt_manifest", "generated_video_dir", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "static_quality",
        "temporal_quality",
        "overall_alignment",
        "fine_grained_alignment",
        "clip_score",
        "blip_score",
        "fid",
        "fvd",
        "fetv_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers FETV fine-grained text-to-video prompts and human/automatic score families.",
        "Official FETV-EVAL runtime and UMT metric dependencies are not executed by the contract fixture.",
    ),
)

PhysVidBenchContract = ExternalBenchmarkContract(
    benchmark_id="physvidbench",
    display_name="PhysVidBench",
    input_keys=("physical_commonsense_prompt_manifest", "generated_video_dir", "caption_judge_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "physical_commonsense_accuracy",
        "affordance_understanding",
        "tool_use_consistency",
        "material_property_consistency",
        "temporal_dynamics_consistency",
        "physvidbench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract records the paper-described physical-commonsense T2V evaluation and result-normalization surface.",
        "Full official scoring still requires generated videos, AuroraCap caption extraction, and Gemini QA outputs.",
    ),
)

T2VWorldBenchContract = ExternalBenchmarkContract(
    benchmark_id="t2vworldbench",
    display_name="T2VWorldBench",
    input_keys=("world_knowledge_prompt_manifest", "generated_video_dir", "vlm_judge_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "physics_knowledge",
        "nature_knowledge",
        "activity_knowledge",
        "culture_knowledge",
        "causality_knowledge",
        "object_knowledge",
        "world_knowledge_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract records the paper-described 6-category world-knowledge T2V evaluation surface only.",
        "No public official code/data/runtime was confirmed, so this remains blocked until release evidence exists.",
    ),
)

VideoVerseContract = ExternalBenchmarkContract(
    benchmark_id="videoverse",
    display_name="VideoVerse",
    input_keys=("videoverse_prompt_manifest", "generated_video_dir", "vlm_judge_config", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "qa_accuracy",
        "event_coverage",
        "temporal_causality",
        "world_knowledge_consistency",
        "static_scene_consistency",
        "dynamic_event_consistency",
        "videoverse_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers VideoVerse prompt and QA-style world-model evaluation outputs.",
        "Official VLM judge execution and complete dataset materialization remain pending validation.",
    ),
)

PhyFPSBenchGenContract = ExternalBenchmarkContract(
    benchmark_id="phyfps-bench-gen",
    display_name="PhyFPS-Bench-Gen",
    input_keys=(
        "phyfps_prompt_manifest",
        "generated_video_dir",
        "visual_chronometer_root",
        "meta_fps",
        "meta_fps_manifest",
        "official_results_path",
    ),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract", "results_csv"),
    metric_ids=(
        "avg_error_fps",
        "pct_error",
        "inter_video_cv",
        "intra_video_cv",
        "phyfps_bench_gen_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers PhyFPS-Bench-Gen prompt suite, Visual Chronometer PhyFPS prediction, and Meta FPS alignment metrics.",
        "Alignment metrics require Meta FPS inputs from model documentation or a manifest.",
    ),
)

VisualChronometerContract = ExternalBenchmarkContract(
    benchmark_id="visual-chronometer",
    display_name="Visual Chronometer",
    input_keys=(
        "generated_video_dir",
        "visual_chronometer_root",
        "official_results_path",
    ),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract", "results_csv"),
    metric_ids=(
        "mean_phyfps",
        "inter_video_cv",
        "intra_video_cv",
        "visual_chronometer_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers in-tree Visual Chronometer PhyFPS prediction over arbitrary generated videos.",
        "Official-run uses the checked-in runtime wrapper over the Visual Chronometer inference checkout.",
    ),
)

AIGCBenchContract = ExternalBenchmarkContract(
    benchmark_id="aigcbench",
    display_name="AIGCBench",
    input_keys=("reference_image_dir", "prompt_suite_json", "generated_video_dir", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "mse_first",
        "ssim_first",
        "image_genvideo_clip",
        "genvideo_text_clip",
        "genvideo_refvideo_clip_keyframes",
        "flow_square_mean",
        "genvideo_refvideo_clip_corresponding_frames",
        "genvideo_clip_adjacent_frames",
        "frame_count",
        "dover",
        "genvideo_refvideo_ssim",
        "aigcbench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps AIGCBench's 11 paper metrics for image-to-video evaluation to WorldFoundry scorecard.",
        "Official metric execution requires reference assets, CLIP/DOVER/optical-flow dependencies, and upstream scorer outputs.",
    ),
)

MiraBenchContract = ExternalBenchmarkContract(
    benchmark_id="mirabench",
    display_name="MiraBench",
    input_keys=("generated_video_dir", "mira_meta_csv", "ground_truth_frame_dir", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "dynamic_degree",
        "tracking_strength",
        "dino_temporal_consistency",
        "clip_temporal_consistency",
        "temporal_motion_smoothness",
        "mean_absolute_error",
        "root_mean_square_error",
        "aesthetic_quality",
        "imaging_quality",
        "camera_alignment",
        "main_object_alignment",
        "background_alignment",
        "style_alignment",
        "overall_alignment",
        "fvd",
        "fid",
        "kid",
        "mirabench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers MiraBench's 17 paper metrics across motion, temporal, 3D, quality, alignment, and distribution families.",
        "Official evaluation requires MiraBench prompt metadata, generated videos, model checkpoints/scorers, and normalized result files.",
    ),
)

MemoBenchContract = ExternalBenchmarkContract(
    benchmark_id="memobench",
    display_name="MemoBench",
    input_keys=(
        "generated_synthetic_dir",
        "generated_real_dir",
        "memobench_data_root",
        "official_results_path",
        "ors_results_path",
        "vqa_results_dir",
    ),
    output_keys=(
        "scorecard",
        "raw_metric_table",
        "per_sample_scores",
        "eval_csv",
        "ors_scores",
        "vqa_scores",
        "leaderboard_csv",
    ),
    metric_ids=(
        "visual_quality",
        "motion_smoothness",
        "object_identity_consistency",
        "geo3d_consistency",
        "camera_controllability",
        "image_reward_score",
        "object_revisit_score",
        "gt_all_psnr",
        "gt_all_ssim",
        "gt_all_lpips",
        "instruction_following",
        "object_background",
        "continuity_of_memory",
        "physics_adherence",
        "memobench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps MemoBench visual-memory metrics across automated, object revisit, pixel fidelity, and VQA stages.",
        "WorldFoundry normalizes caller-provided MemoBench CSV/JSON outputs and can execute the vendored Step 1 automated metric runner.",
        "Full leaderboard parity requires generated frame artifacts, official dataset assets, SAM-3 for ORS, and Gemini/VLM credentials for VQA.",
    ),
)

DevilDynamicsContract = ExternalBenchmarkContract(
    benchmark_id="devil-dynamics",
    display_name="DEVIL Dynamics",
    input_keys=("generated_video_dir", "prompt_suite_json", "judge_api_config", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "dynamics_range",
        "dynamics_controllability",
        "dynamics_quality",
        "devil_dynamics_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps DEVIL dynamics-focused T2V metrics to WorldFoundry scorecard.",
        "Naturalness judging can require Gemini/API configuration in the upstream runtime.",
    ),
)

GenAIBenchContract = ExternalBenchmarkContract(
    benchmark_id="genai-bench",
    display_name="GenAI-Bench",
    input_keys=("generated_artifact_manifest", "human_preference_data", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "pairwise_preferences", "benchmark_contract"),
    metric_ids=(
        "pairwise_accuracy",
        "image_generation_preference_accuracy",
        "image_editing_preference_accuracy",
        "video_preference_accuracy",
        "genai_bench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract records GenAI-Bench pairwise judge accuracy against human preference labels.",
        "Full inference requires MLLM judge outputs; existing human labels and model predictions can be normalized locally.",
    ),
)

PhyGenBenchContract = ExternalBenchmarkContract(
    benchmark_id="phygenbench",
    display_name="PhyGenBench",
    input_keys=("physics_prompt_manifest", "generated_video_dir", "phygen_eval_config", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "physical_commonsense",
        "physical_law_adherence",
        "semantic_adherence",
        "phygenbench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps PhyGenBench/PhyGenEval physical commonsense outputs to WorldFoundry scorecard.",
        "Official execution requires the upstream PhyGenEval judge stack and generated videos.",
    ),
)

VideoPhyContract = ExternalBenchmarkContract(
    benchmark_id="videophy",
    display_name="VideoPhy",
    input_keys=("videophy_prompt_manifest", "generated_video_dir", "autoeval_checkpoint", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "semantic_adherence",
        "physical_commonsense",
        "joint_score",
        "videophy_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers VideoPhy semantic adherence and physical commonsense scoring.",
        "Official auto-rater and data materialization remain explicit upstream runtime stages.",
    ),
)

VideoPhy2Contract = ExternalBenchmarkContract(
    benchmark_id="videophy2",
    display_name="VideoPhy2",
    input_keys=("videophy2_prompt_manifest", "generated_video_dir", "autoeval_checkpoint", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "semantic_adherence",
        "physical_commonsense",
        "joint_score",
        "rule_followed_rate",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers VideoPhy2 action-centric physical commonsense evaluation.",
        "Official auto-evaluator checkpoint and HF test data are external acquisition stages.",
    ),
)

PhysicsIQContract = ExternalBenchmarkContract(
    benchmark_id="physics-iq",
    display_name="Physics-IQ Original",
    input_keys=("reference_video_or_frames", "generated_video_dir", "physics_iq_metadata", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "physics_iq_score",
        "physics_iq_stable_score",
        "physics_iq_spatiotemporal",
        "physics_iq_spatial",
        "physics_iq_weighted_spatial",
        "physics_iq_mse_penalty",
    ),
    requires_upstream_runtime=False,
    notes=(
        "Contract maps the original Physics-IQ protocol to the in-tree raw-video evaluator.",
        "Official dataset media remain external assets; no upstream source checkout is used at runtime.",
    ),
)

PhysicsIQVerifiedContract = ExternalBenchmarkContract(
    benchmark_id="physics-iq-verified",
    display_name="Physics-IQ Verified",
    input_keys=("reference_video_or_frames", "generated_video_dir", "physics_iq_metadata", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "physics_iq_verified_score",
        "physics_iq_verified_spatiotemporal",
        "physics_iq_verified_spatial",
        "physics_iq_verified_weighted_spatial",
        "physics_iq_verified_mse",
    ),
    requires_upstream_runtime=False,
    notes=(
        "Contract maps Physics-IQ Verified to the shared in-tree raw-video evaluator and verified score protocol.",
        "Official verified dataset media remain external assets; no upstream source checkout is used at runtime.",
    ),
)

T2VSafetyBenchContract = ExternalBenchmarkContract(
    benchmark_id="t2v-safety-bench",
    display_name="T2VSafetyBench",
    input_keys=("safety_prompt_manifest", "generated_video_dir", "judge_config", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "pornography_nsfw_rate",
        "borderline_pornography_nsfw_rate",
        "violence_nsfw_rate",
        "gore_nsfw_rate",
        "public_figures_nsfw_rate",
        "discrimination_nsfw_rate",
        "political_sensitivity_nsfw_rate",
        "illegal_activities_nsfw_rate",
        "disturbing_content_nsfw_rate",
        "misinformation_falsehoods_nsfw_rate",
        "copyright_trademark_nsfw_rate",
        "temporal_risk_nsfw_rate",
        "nsfw_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract exposes T2VSafetyBench as a safety-critical text-to-video evaluation surface.",
        "Full scoring requires the upstream safety judge configuration and generated videos.",
    ),
)

IPVBenchContract = ExternalBenchmarkContract(
    benchmark_id="ipv-bench",
    display_name="IPV-Bench",
    input_keys=("impossible_prompt_manifest", "generated_video_dir", "understanding_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "visual_quality",
        "prompt_following",
        "impossible_video_score",
        "judgement_accuracy",
        "mcqa_accuracy",
        "open_qa_score",
        "ipv_bench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract records IPV-Bench impossible-video generation and understanding metrics.",
        "Human annotation or official judge outputs remain required for leaderboard-grade scores.",
    ),
)

WorldReasonBenchContract = ExternalBenchmarkContract(
    benchmark_id="worldreasonbench",
    display_name="WorldReasonBench",
    input_keys=("qa_manifest", "generated_video_dir", "reward_pairs", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "protocol_results", "benchmark_contract"),
    metric_ids=(
        "score_pr",
        "qa_accuracy",
        "state_score",
        "process_score",
        "fidelity_score",
        "mechanism_score",
        "static_outcome_score",
        "dynamic_reasoning_score",
        "reasoning_gap",
        "pointwise_score",
        "reasoning_correctness",
        "content_fidelity",
        "visual_aesthetics",
        "pointwise_spearman",
        "induced_pairwise_accuracy",
        "pairwise_accuracy_with_ties",
        "pairwise_accuracy_without_ties",
    ),
    requires_upstream_runtime=True,
    notes=(
        "All three official protocol families are normalized by an in-tree runner.",
        "Official restricted data and judge prompts remain caller-provided external assets.",
    ),
)

VideoScienceBenchContract = ExternalBenchmarkContract(
    benchmark_id="videoscience-bench",
    display_name="VideoScience-Bench",
    input_keys=("science_prompt_manifest", "generated_video_dir", "judge_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "prompt_consistency",
        "phenomenon_congruency",
        "correct_dynamism",
        "immutability",
        "spatio_temporal_coherence",
        "videoscience_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps VideoScience-Bench scientific reasoning evaluation to WorldFoundry scorecard.",
        "Official data and evaluation code remain upstream runtime dependencies.",
    ),
)

PhyEduVideoContract = ExternalBenchmarkContract(
    benchmark_id="phyeduvideo",
    display_name="PhyEduVideo",
    input_keys=("physics_education_prompt_manifest", "generated_video_dir", "judge_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "semantic_adherence",
        "physics_commonsense",
        "motion_smoothness",
        "temporal_flickering",
        "phyeduvideo_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers PhyEduVideo physics-education text-to-video evaluation.",
        "Official evaluation requires the upstream codebase and any judge/runtime assets.",
    ),
)

WorldArenaContract = ExternalBenchmarkContract(
    benchmark_id="worldarena",
    display_name="WorldArena",
    input_keys=("generated_video_dir", "worldarena_task_manifest", "embodied_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "visual_quality",
        "motion_quality",
        "content_consistency",
        "physics_adherence",
        "three_d_accuracy",
        "controllability",
        "data_engine_success",
        "policy_evaluator_correlation",
        "action_planner_success",
        "human_quality",
        "ewm_score",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract records WorldArena perceptual and functional embodied-world-model evaluation outputs.",
        "Full validation requires WorldArena/RoboTwin assets and official task execution.",
    ),
)

WorldInWorldContract = ExternalBenchmarkContract(
    benchmark_id="world-in-world",
    display_name="World-in-World",
    input_keys=("closed_loop_task_manifest", "world_model_rollouts_dir", "planner_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "rollout_logs", "benchmark_contract"),
    metric_ids=(
        "active_recognition_success_rate",
        "image_goal_navigation_success_rate",
        "image_goal_navigation_spl",
        "active_embodied_qa_score",
        "active_embodied_qa_spl",
        "robotic_manipulation_success_rate",
        "interaction_trace_consistency",
        "world_in_world_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers closed-loop visual-world-model utility metrics.",
        "Official evaluation remains simulator/planner heavy and is kept as an upstream runtime stage.",
    ),
)

PhyGroundContract = ExternalBenchmarkContract(
    benchmark_id="phyground",
    display_name="PhyGround",
    input_keys=("phyground_prompt_manifest", "first_frame_dir", "generated_video_dir", "judge_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "semantic_adherence",
        "physical_temporal_validity",
        "persistence",
        "solid_body_score",
        "fluid_dynamics_score",
        "optics_score",
        "phyground_overall",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract maps PhyGround physical reasoning scores and per-law diagnostics to WorldFoundry scorecard.",
        "Official scoring requires the released prompt/image assets and judge model or API backend.",
    ),
)

EWMBenchContract = ExternalBenchmarkContract(
    benchmark_id="ewmbench",
    display_name="EWMBench",
    input_keys=("generated_frame_dir", "ewmbench_task_manifest", "official_results_path"),
    output_keys=("scorecard", "raw_metric_table", "per_sample_scores", "benchmark_contract"),
    metric_ids=(
        "scene_consistency",
        "motion_correctness",
        "semantic_alignment",
        "diversity",
        "ewmbench_average",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers EWMBench scene, motion, and semantic quality evaluation for embodied world models.",
        "Official execution requires the upstream evaluator, generated frames/videos, and benchmark assets.",
    ),
)

WRBenchContract = ExternalBenchmarkContract(
    benchmark_id="wrbench",
    display_name="WRBench",
    input_keys=(
        "natural25_manifest",
        "generated_video_dir",
        "video_manifest",
        "camera_sidecars",
        "official_results_path",
    ),
    output_keys=(
        "scorecard",
        "benchmark_contract",
        "raw_metric_table",
        "per_sample_scores",
        "main_table",
    ),
    metric_ids=(
        "d1_cam_prec",
        "d1_cam_align_common_yaw",
        "d1_cam_align_static_hold",
        "d2_visual_integrity",
        "d3_visible_spatial_consistency",
        "d4_visible_state_consistency",
        "d5_reobservation_spatial_consistency",
        "d6_reobservation_state_consistency",
        "wrbench_average",
    ),
    requires_upstream_runtime=False,
    notes=(
        "The complete WRBench Python package and Natural-25 assets are vendored in-tree at revision 629595dc60ec08a29711af0377280c4ac9dd40bc.",
        "D1-CamPrec and D1-CamAlign are separate metrics; D5/D6 use the official re-observation applicability gate.",
        "Full scoring requires explicitly configured VGGT-Omega, DINOv2, Qwen3.5, and Qwen3-VL model assets.",
    ),
)

LIBEROContract = ExternalBenchmarkContract(
    benchmark_id="libero",
    display_name="LIBERO",
    input_keys=("dataset_root", "episode_manifest", "policy_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs"),
    metric_ids=("success_rate", "task_success", "episode_success", "action_accuracy"),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers LIBERO policy rollouts and official result normalization for VLA models.",
        "WorldFoundry can normalize official result files directly; simulator runtime remains an explicit upstream stage.",
    ),
)

LIBEROParaContract = ExternalBenchmarkContract(
    benchmark_id="libero-para",
    display_name="LIBERO-Para",
    input_keys=(
        "dataset_root",
        "paraphrase_manifest",
        "episode_manifest",
        "policy_results_path",
        "official_results_path",
    ),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs"),
    metric_ids=(
        "success_rate",
        "paraphrase_success_rate",
        "language_generalization_success",
        "task_success",
        "action_accuracy",
    ),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers LIBERO-Para paraphrased language-instruction manipulation evaluation for VLA policies.",
        "WorldFoundry normalizes official-shaped result files; upstream LIBERO-Para runtime execution remains external evidence.",
    ),
)

SimplerEnvContract = ExternalBenchmarkContract(
    benchmark_id="simpler-env",
    display_name="SimplerEnv",
    input_keys=("simulator_root", "episode_manifest", "policy_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs", "videos"),
    metric_ids=("success_rate", "task_success", "episode_success", "reward"),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers ManiSkill2/Google-Robot style SimplerEnv VLA evaluation.",
        "Official simulator execution and policy wrappers are handled by the upstream runtime or subprocess runner.",
    ),
)

RoboCasaContract = ExternalBenchmarkContract(
    benchmark_id="robocasa",
    display_name="RoboCasa",
    input_keys=("dataset_root", "simulator_root", "task_manifest", "policy_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs", "videos"),
    metric_ids=("success_rate", "task_success", "episode_success", "reward"),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers RoboCasa long-horizon manipulation tasks for embodied VLA policies.",
        "WorldFoundry treats generated rollouts and official result JSON/CSV files as the stable bridge.",
    ),
)

CALVINContract = ExternalBenchmarkContract(
    benchmark_id="calvin",
    display_name="CALVIN",
    input_keys=("dataset_root", "task_manifest", "policy_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs"),
    metric_ids=("success_rate", "sequence_success", "task_success", "completion"),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers CALVIN long-horizon language-conditioned manipulation.",
        "Upstream runtime is kept separate from WorldFoundry scorecard normalization.",
    ),
)

ManiSkillContract = ExternalBenchmarkContract(
    benchmark_id="maniskill",
    display_name="ManiSkill",
    input_keys=("simulator_root", "task_manifest", "policy_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs", "videos"),
    metric_ids=("success_rate", "task_success", "reward", "normalized_return"),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers ManiSkill simulated manipulation and planning benchmarks.",
        "WorldFoundry provides the common VLA/VA/WAM request/result surface around upstream runtime execution.",
    ),
)

RLBenchContract = ExternalBenchmarkContract(
    benchmark_id="rlbench",
    display_name="RLBench",
    input_keys=("simulator_root", "task_manifest", "policy_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs", "videos"),
    metric_ids=("success_rate", "task_success", "episode_success", "reward", "normalized_return"),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers RLBench vision-guided manipulation rollouts for robot policy evaluation.",
        "CoppeliaSim/PyRep execution remains an explicit upstream stage; WorldFoundry normalizes result files.",
    ),
)

MetaWorldContract = ExternalBenchmarkContract(
    benchmark_id="metaworld",
    display_name="Meta-World",
    input_keys=("simulator_root", "task_manifest", "policy_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs"),
    metric_ids=("success_rate", "task_success", "reward", "normalized_return"),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers Meta-World multi-task and meta-RL manipulation suites.",
        "WorldFoundry treats continuous-control policy outputs as action traces rather than generated videos.",
    ),
)

BridgeDataV2Contract = ExternalBenchmarkContract(
    benchmark_id="bridgedata-v2",
    display_name="BridgeData V2",
    input_keys=("dataset_root", "episode_manifest", "policy_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs"),
    metric_ids=("success_rate", "task_success", "goal_success", "action_accuracy", "rollout_success"),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers BridgeData V2 real-robot dataset and policy evaluation result normalization.",
        "Dataset conversion, checkpoint selection, and robot/runtime execution stay outside the core scorecard path.",
    ),
)

AI2THORContract = ExternalBenchmarkContract(
    benchmark_id="ai2thor",
    display_name="AI2-THOR",
    input_keys=("scene", "language_instruction", "observation_image", "policy_action", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs"),
    metric_ids=("success_rate", "task_success", "episode_success", "reward"),
    requires_upstream_runtime=True,
    notes=(
        "Contract covers AI2-THOR native embodied closed-loop scene rollouts.",
        "Unity simulator assets, display/CloudRendering backend, and policy server remain explicit runtime inputs.",
    ),
)

_EMBODIED_ACTION_INPUT_KEYS = (
    "dataset_root",
    "simulator_root",
    "task_manifest",
    "policy_results_path",
    "official_results_path",
)
_EMBODIED_ACTION_OUTPUT_KEYS = (
    "scorecard",
    "raw_results",
    "per_sample_metrics",
    "rollout_logs",
    "videos",
)
_EMBODIED_ACTION_METRIC_IDS = (
    "success_rate",
    "task_success",
    "episode_success",
    "reward",
    "normalized_return",
)


def _embodied_action_contract(
    benchmark_id: str,
    display_name: str,
    *,
    input_keys: tuple[str, ...] = _EMBODIED_ACTION_INPUT_KEYS,
    output_keys: tuple[str, ...] = _EMBODIED_ACTION_OUTPUT_KEYS,
    metric_ids: tuple[str, ...] = _EMBODIED_ACTION_METRIC_IDS,
    notes: tuple[str, ...] = (),
) -> ExternalBenchmarkContract:
    """Helper factory to construct an ExternalBenchmarkContract for an embodied action task.

    Args:
        benchmark_id: Canonical identifier for the benchmark.
        display_name: Human-readable display name.
        input_keys: Explicit list of required input data/assets.
        output_keys: Explicit list of guaranteed output artifacts.
        metric_ids: Sequence of official metric scalar keys this benchmark produces.
        notes: Explanatory annotations and developer cautions.

    Returns:
        A fully constructed ExternalBenchmarkContract instance.
    """
    return ExternalBenchmarkContract(
        benchmark_id=benchmark_id,
        display_name=display_name,
        input_keys=input_keys,
        output_keys=output_keys,
        metric_ids=metric_ids,
        requires_upstream_runtime=True,
        notes=(
            f"Contract covers {display_name} embodied policy rollouts and official result normalization.",
            "WorldFoundry exposes the native request/result surface; simulator assets, policy checkpoints, and full official rollout evidence remain explicit runtime inputs.",
            *notes,
        ),
    )


Behavior1KContract = _embodied_action_contract(
    "behavior1k",
    "BEHAVIOR-1K",
    input_keys=("behavior_dataset_root", "simulator_root", "task_manifest", "policy_results_path", "official_results_path"),
    metric_ids=("success_rate", "task_success", "episode_success", "subgoal_success", "reward"),
    notes=("BEHAVIOR-1K requires OmniGibson/Isaac Sim assets and license-gated simulator setup.",),
)
KinetixContract = _embodied_action_contract(
    "kinetix",
    "Kinetix",
    metric_ids=("success_rate", "task_success", "episode_success", "reward", "normalized_return", "real_time_success_rate"),
    notes=("Kinetix uses a JAX 2D physics runtime; keep it in a simulator-specific environment when the unified PyTorch environment is insufficient.",),
)
LIBEROMemContract = _embodied_action_contract(
    "libero-mem",
    "LIBERO-Mem",
    input_keys=("dataset_root", "episode_manifest", "memory_manifest", "policy_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs"),
    metric_ids=("success_rate", "memory_success_rate", "task_success", "episode_success", "action_accuracy"),
)
LIBEROPlusContract = _embodied_action_contract(
    "libero-plus",
    "LIBERO-Plus",
    input_keys=("dataset_root", "episode_manifest", "policy_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs"),
    metric_ids=("success_rate", "task_success", "episode_success", "action_accuracy"),
)
LIBEROProContract = _embodied_action_contract(
    "libero-pro",
    "LIBERO-Pro",
    input_keys=("dataset_root", "episode_manifest", "policy_results_path", "official_results_path"),
    output_keys=("scorecard", "raw_results", "per_sample_metrics", "rollout_logs"),
    metric_ids=("success_rate", "task_success", "episode_success", "action_accuracy"),
)
ManiSkill2Contract = _embodied_action_contract(
    "maniskill2",
    "ManiSkill2",
    input_keys=ManiSkillContract.input_keys,
    output_keys=ManiSkillContract.output_keys,
    metric_ids=ManiSkillContract.metric_ids,
    notes=("The legacy WorldFoundry benchmark id `maniskill` remains available for older manifests.",),
)
MIKASAContract = _embodied_action_contract(
    "mikasa",
    "MiKASA-Robo",
    metric_ids=("success_rate", "task_success", "episode_success", "reward", "normalized_return"),
)
MolmoSpacesContract = _embodied_action_contract(
    "molmospaces",
    "MolmoSpaces-Bench",
    input_keys=("molmospaces_asset_root", "simulator_root", "task_manifest", "policy_results_path", "official_results_path"),
    metric_ids=("success_rate", "pick_success_rate", "task_success", "episode_success", "reward"),
    notes=("MolmoSpaces assets are large; stage only the benchmark subset required by the selected suite.",),
)
RoboCerebraContract = _embodied_action_contract(
    "robocerebra",
    "RoboCerebra",
    metric_ids=("success_rate", "task_success", "episode_success", "reward"),
)
RoboMMEContract = _embodied_action_contract(
    "robomme",
    "RoboMME",
    input_keys=("dataset_root", "simulator_root", "memory_manifest", "policy_results_path", "official_results_path"),
    metric_ids=("success_rate", "counting_success_rate", "permanence_success_rate", "reference_success_rate", "imitation_success_rate"),
)
VLABenchContract = _embodied_action_contract(
    "vlabench",
    "VLABench",
    metric_ids=("success_rate", "task_success", "episode_success", "reward", "normalized_return"),
)

LARYBenchContract = ExternalBenchmarkContract(
    benchmark_id="larybench",
    display_name="LARYBench",
    input_keys=(
        "dataset_root",
        "metadata_root",
        "latent_action_root",
        "model_root",
        "official_results_path",
    ),
    output_keys=("scorecard", "raw_metric_table", "benchmark_contract", "runtime_logs"),
    metric_ids=(
        "extraction_coverage",
        "extracted_samples",
        "input_samples",
        "top1_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "sample_count",
        "train_loss",
        "val_seen_loss",
        "val_seen_mse",
        "val_seen_mse_position",
        "val_seen_mse_orientation",
        "val_seen_mse_gripper",
        "val_unseen_loss",
        "val_unseen_mse",
        "val_unseen_mse_position",
        "val_unseen_mse_orientation",
        "val_unseen_mse_gripper",
    ),
    requires_upstream_runtime=False,
    notes=(
        "The LARYBench extraction, classification, and regression runtime is integrated in tree.",
        "Datasets and model checkpoints remain explicit assets; no external source checkout is imported or executed.",
        "Per-action-dimension MSE fields may be added to scorecards according to the selected robot embodiment.",
    ),
)

RoboTwinContract = ExternalBenchmarkContract(
    benchmark_id=ROBOTWIN_BENCHMARK_ID,
    display_name=ROBOTWIN_BENCHMARK_NAME,
    input_keys=ROBOTWIN_INPUT_KEYS,
    output_keys=ROBOTWIN_OUTPUT_KEYS,
    metric_ids=ROBOTWIN_METRIC_IDS,
    requires_upstream_runtime=True,
    notes=ROBOTWIN_CONTRACT_NOTES,
)

_BUILTIN_CONTRACT_ITEMS = (
    AI2THORContract,
    AIGCBenchContract,
    Behavior1KContract,
    BridgeDataV2Contract,
    CameraBenchContract,
    CALVINContract,
    ChronoMagicBenchContract,
    DevilDynamicsContract,
    EWMBenchContract,
    WRBenchContract,
    EvalCrafterContract,
    FETVContract,
    FourDWorldBenchContract,
    GenAIBenchContract,
    IPVBenchContract,
    IWorldBenchContract,
    KinetixContract,
    LARYBenchContract,
    LIBEROContract,
    LIBEROMemContract,
    LIBEROParaContract,
    LIBEROPlusContract,
    LIBEROProContract,
    ManiSkillContract,
    ManiSkill2Contract,
    MetaWorldContract,
    MIKASAContract,
    MemoBenchContract,
    MiraBenchContract,
    MolmoSpacesContract,
    PhyEduVideoContract,
    PhysVidBenchContract,
    PhyGenBenchContract,
    PhyGroundContract,
    PhysicsIQContract,
    PhysicsIQVerifiedContract,
    PhysicalAIBenchContract,
    RoboCasaContract,
    RoboCerebraContract,
    RoboMMEContract,
    RoboTwinContract,
    RLBenchContract,
    SimplerEnvContract,
    T2VSafetyBenchContract,
    T2VWorldBenchContract,
    T2VCompBenchContract,
    VideoBenchContract,
    VideoPhyContract,
    VideoPhy2Contract,
    VideoVerseContract,
    PhyFPSBenchGenContract,
    VisualChronometerContract,
    VideoScienceBenchContract,
    VideoScoreContract,
    VBench2Contract,
    VBenchContract,
    VBenchPlusPlusContract,
    VLABenchContract,
    VMBenchContract,
    WBenchContract,
    WorldArenaContract,
    WorldBenchContract,
    WorldInWorldContract,
    WorldModelBenchContract,
    WorldReasonBenchContract,
    WorldScoreContract,
)
for _contract in _BUILTIN_CONTRACT_ITEMS:
    register_external_benchmark_contract(_contract, source="builtin")

_CONTRACT_REGISTRY = ExternalBenchmarkContractRegistry(include_registered=True)
del _contract


def get_external_benchmark_contract(benchmark_id: str) -> ExternalBenchmarkContract:
    """Retrieves an explicit evaluation contract bound to the given benchmark ID.

    Resolution order:

    1. Built-in contracts registered in :mod:`contracts.external`.
    2. Catalog ``runner_target`` contract referenced by the benchmark-zoo manifest.
    """
    try:
        return _CONTRACT_REGISTRY.get(benchmark_id)
    except UnknownExternalBenchmarkContractError:
        from worldfoundry.evaluation.tasks.catalog.specs import external_benchmark_contract_for_id

        contract = external_benchmark_contract_for_id(benchmark_id)
        if contract is not None:
            return contract
        raise


def list_external_benchmark_contracts() -> tuple[ExternalBenchmarkContract, ...]:
    """Retrieves all statically registered external benchmark contracts across the system.

    Returns:
        A sorted tuple of all registered ExternalBenchmarkContract objects.
    """
    return _CONTRACT_REGISTRY.list()


__all__ = sorted(
    name
    for name in globals()
    if not name.startswith("_") and name not in {"Path", "annotations"}
)
