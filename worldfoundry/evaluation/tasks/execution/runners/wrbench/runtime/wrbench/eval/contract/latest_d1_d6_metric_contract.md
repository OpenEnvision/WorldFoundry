# Latest D1-D6 Metric Contract

- schema: `wrbench_latest_d1_d6_metrics_v3`
- fallback score fields allowed: `false`
- pending evidence dimensions: `none`

| dimension | output column | source field | source role | promotion status | current run tag | forbidden fallback fields |
|---|---|---|---|---|---|---|
| D1 | `D1_camera_pose` | `d1_camera_accuracy` | `d1_requested_control_rows_jsonl` | `paper_facing_current` | `wrbench_paper_v1` | `none` |
| D1-CamAlign | `D1_camalign` | `d1_camalign_score` | `d1_camalign_rows_jsonl` | `paper_facing_current` | `wrbench_paper_v1` | `none` |
| D2 | `D2_visual_integrity` | `d2_selected_visual_integrity_score` | `d2_selected_visual_integrity_scores_json` | `paper_facing_current` | `d2_lg_v2_candidate_e_as_selected_visual_integrity` | `d2_dinov2_temporal_consistency` |
| D3 | `D3_spatial_in` | `vlm_spatial_fidelity` | `runtime_v2_score_probe_or_gate_masked_export` | `benchmark_default_current` | `wrbench_paper_v1` | `none` |
| D4 | `D4_state_in` | `vlm_state_fidelity` | `runtime_v2_score_probe_or_gate_masked_export` | `benchmark_default_current` | `wrbench_paper_v1` | `none` |
| D5 | `D5_spatial_oov` | `vlm_spatial_reasoning` | `runtime_v2_gate_masked_export` | `benchmark_default_current` | `wrbench_paper_v1` | `none` |
| D6 | `D6_state_oov` | `vlm_state_reasoning` | `runtime_v2_gate_masked_export` | `benchmark_default_current` | `wrbench_paper_v1` | `none` |

## D1 Reporting Exclusions

- D1-CamPrec excludes `100` `hunyuan_game_craft` `static` rows from the reporting denominator (`unsupported_static_hold_upstream_gamecraft`); raw row scoring remains auditable.

## Method Notes

- D2 uses the fixed full-FoV DINOv2 v2 visual-integrity score: `d2_selected_visual_integrity_score = d2_lg_v2_candidate_e = min(cls_first_last_cosine, local_patch_token_low_p20)`. Preprocessing is `full_fov_resize_pad_no_center_crop`; sampling is `time_fps_3_max24`. The legacy display label `D2_dinov2_temporal_consistency` is forbidden as a fallback.
