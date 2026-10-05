"""Hand-tuned prose for high-traffic model homes.

Facts only — every sentence is grounded in model-recipes-data.json
(overview, architecture, usage, limitations, variants, checkpoints, commands).
Do not invent scores, leaderboards, or paper claims.

WorldFoundry pages are inference contracts, not training write-ups.
Prefer variant + ``worldfoundry-eval run``, then knobs, then env / prepare /
license+VRAM. Omit pretraining hours, datasets, optimizer, and “how the
authors trained it” unless one clause names a checkpoint.
Do not add a "What this page does not claim" / "Limits" / "本页不声称" section.
Never emit raw manifest dialect (``profile_resolves_*``,
``official_demo_parity_pending``, ``in_tree_checkpoint_runtime_static_verified``).
"""

from __future__ import annotations

from typing import Any

# description / use_cases / contract_extra / run_extra
# Each locale list is 1–2 short paragraphs. Inference-first.
EXEMPLAR_PROSE: dict[str, dict[str, Any]] = {
    "being-h05": {
        "description": {
            "en": "BeingBeyond 2B cross-embodiment VLA; start on the GPU-validated LIBERO line.",
            "zh": "BeingBeyond 的 2B 跨本体 VLA；从已 GPU 验证的 LIBERO 线起步。",
        },
        "use_cases": {
            "en": [
                "Start on `being-h05-2b-libero`. Launch `worldfoundry-eval run being-h05-2b`. Swap `being-h05-2b-robocasa` or the combined LIBERO+RoboCasa checkpoint only when comparing embodiments — the base `being-h05-2b` card is not the validated path.",
            ],
            "zh": [
                "从 `being-h05-2b-libero` 开始。启动 `worldfoundry-eval run being-h05-2b`。比较跨本体时再换 `being-h05-2b-robocasa` 或 LIBERO+RoboCasa 组合权重；基座 `being-h05-2b` 不是已验证路径。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `image`, `instruction`, `prompt`, and `state`. Optional `video`. `actions`: `robot_action`, `action_unified`, `proprio`, `action_space`. Default profile pins `data_config_name=libero_nonorm`, `dataset_name=libero_posttrain`, `embodiment_tag=libero`. Artifact: `being_h05_action_trace.json`. Paper eval on this 2B backbone uses RGB-only 224×224.",
            ],
            "zh": [
                "必填 `image`、`instruction`、`prompt`、`state`。可选 `video`。`actions`：`robot_action`、`action_unified`、`proprio`、`action_space`。默认 profile 固定 `data_config_name=libero_nonorm`、`dataset_name=libero_posttrain`、`embodiment_tag=libero`。产物：`being_h05_action_trace.json`。论文评测用 RGB-only 224×224。",
            ],
        },
        "run_extra": {
            "en": [
                "Dedicated `worldfoundry-being-h05` (Python 3.10, CUDA 11.8 or newer). Stage `$WORLDFOUNDRY_HFD_ROOT/BeingBeyond--Being-H05-2B_libero`. Weights are not downloaded at runtime. Confirm the Hugging Face card before production use. Status: Verified on the LIBERO specialist.",
            ],
            "zh": [
                "独立环境 `worldfoundry-being-h05`（Python 3.10，CUDA 11.8 或更新）。就位 `$WORLDFOUNDRY_HFD_ROOT/BeingBeyond--Being-H05-2B_libero`。运行时不下载权重。生产前确认 Hugging Face 卡片。状态：LIBERO 专线已验证。",
            ],
        },
    },
    "being-h07": {
        "description": {
            "en": "BeingBeyond H0.7 latent world-action model; launch the VLA profile. GPU artifact not recorded.",
            "zh": "BeingBeyond H0.7 潜空间世界-动作模型；跑 VLA profile。本卡片未记录 GPU 产物。",
        },
        "use_cases": {
            "en": [
                "Start on `being-h07`. Launch `worldfoundry-eval run being-h07 --pipeline.task-profile vla`. Need a GPU-validated BeingBeyond route? Use [Being-H0.5](/docs/guides/supported-models/being-h05) instead.",
            ],
            "zh": [
                "从 `being-h07` 开始。启动 `worldfoundry-eval run being-h07 --pipeline.task-profile vla`。需要 GPU 已验证的 BeingBeyond 路径时改走 [Being-H0.5](/zh/docs/guides/supported-models/being-h05)。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `image`, `prompt`, and `state`. Artifacts: `action_trace`, `being-h07_action_trace.json`.",
            ],
            "zh": [
                "必填 `image`、`prompt`、`state`。产物：`action_trace`、`being-h07_action_trace.json`。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128` (Python 3.11, CUDA 12.8). Stage `BeingBeyond/Being-H07` when released — missing weights become a runtime plan, not a catalog blocker. No license is recorded. Status: Runner parity pending.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`（Python 3.11，CUDA 12.8）。权重 `BeingBeyond/Being-H07` 就位前运行只出 runtime plan，不是 catalog 阻塞。未记录 license。状态：Runner 一致性待确认。",
            ],
        },
    },
    "easyanimate": {
        "description": {
            "en": "Alibaba PAI DiT video family; WorldFoundry wraps the V5.1 I2V runtime. Runner parity pending.",
            "zh": "阿里 PAI 的 DiT 视频系列；WorldFoundry 封装 V5.1 图生视频运行时。Runner 一致性待确认。",
        },
        "use_cases": {
            "en": [
                "Start on the V5.1 I2V line (`easyanimate_i2v` / `easyanimate-i2v`). Launch `worldfoundry-eval run easyanimate`. Pick `*-InP` weights for image-conditioned demos; EasyAnimateV5.1-7b-zh is the text-to-video weight — do not mix the two.",
            ],
            "zh": [
                "从 V5.1 图生视频线（`easyanimate_i2v` / `easyanimate-i2v`）开始。启动 `worldfoundry-eval run easyanimate`。图像条件用 `*-InP` 权重；EasyAnimateV5.1-7b-zh 才是文生视频权重，不要混用。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `prompt` and `image`. Artifact: `easyanimate-i2v.mp4`.",
            ],
            "zh": [
                "必填 `prompt`、`image`。产物：`easyanimate-i2v.mp4`。",
            ],
        },
        "run_extra": {
            "en": [
                "Independent EasyAnimate runtime; WRBench camera-control reuses it. Status: Runner parity pending.",
            ],
            "zh": [
                "独立 EasyAnimate 运行时；WRBench 相机控制复用同一套。状态：Runner 一致性待确认。",
            ],
        },
    },
    "pi0": {
        "description": {
            "en": "Physical Intelligence flow-matching VLA on the in-tree OpenPI runtime; GCS weights only.",
            "zh": "Physical Intelligence 的 flow-matching VLA，走树内 OpenPI 运行时；权重只在 GCS。",
        },
        "use_cases": {
            "en": [
                "Start on `pi0`. Launch `worldfoundry-eval run pi0`. This is the concrete π0 algorithm route, not a LeRobot zoo policy. LIBERO evidence sits on `pi05_libero` under openpi, not this card.",
            ],
            "zh": [
                "从 `pi0` 开始。启动 `worldfoundry-eval run pi0`。这是具体的 π0 算法路由，不是 LeRobot policy-zoo。LIBERO 证据在 openpi 家族的 `pi05_libero`，不在本卡。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `image`, `instruction`, `prompt`. Optional `video`. `actions`: `robot_action`, `proprio`, `action_space`. Default config `pi0_droid`. Artifact: `pi0_action_trace.json`.",
            ],
            "zh": [
                "必填 `image`、`instruction`、`prompt`。可选 `video`。`actions`：`robot_action`、`proprio`、`action_space`。默认配置 `pi0_droid`。产物：`pi0_action_trace.json`。",
            ],
        },
        "run_extra": {
            "en": [
                "Dedicated `worldfoundry-openpi-cu12` (Python 3.11, CUDA 12.4 or newer). Stage `gs://openpi-assets` with gsutil — no Hugging Face checkpoint is recorded. Status: Runner parity pending; numerical parity is gated on those GCS assets.",
            ],
            "zh": [
                "独立环境 `worldfoundry-openpi-cu12`（Python 3.11，CUDA 12.4 或更新）。用 gsutil 就位 `gs://openpi-assets`——本条目没有 Hugging Face checkpoint。状态：Runner 一致性待确认；数值一致性只取决于 GCS 资产。",
            ],
        },
    },
    "pi0-fast": {
        "description": {
            "en": "Autoregressive π0 variant: FAST action tokens instead of flow matching; default `pi0_fast_droid`.",
            "zh": "π0 的自回归变体：用 FAST 动作 token 代替 flow matching；默认 `pi0_fast_droid`。",
        },
        "use_cases": {
            "en": [
                "Start on `pi0-fast` when you want discrete FAST tokens on the same PaliGemma backbone. Launch `worldfoundry-eval run pi0-fast`. Do not route this through LeRobot policy-zoo.",
            ],
            "zh": [
                "需要同一 PaliGemma 骨干上的离散 FAST token 时选 `pi0-fast`。启动 `worldfoundry-eval run pi0-fast`。不要走 LeRobot policy-zoo。",
            ],
        },
        "contract_extra": {
            "en": [
                "Default configuration `pi0_fast_droid`. Artifact: `pi0_fast_action_trace.json`. Faster decode, less smooth chunks than flow-matching π0.",
            ],
            "zh": [
                "默认配置 `pi0_fast_droid`。产物：`pi0_fast_action_trace.json`。解码比 flow-matching π0 更快，动作块更不平滑。",
            ],
        },
        "run_extra": {
            "en": [
                "Same OpenPI env as π0. Stage GCS checkpoints with gsutil. Status: Runner parity pending until assets are present.",
            ],
            "zh": [
                "与 π0 同一 OpenPI 环境。用 gsutil 就位 GCS checkpoint。状态：资产未就位前为 Runner 一致性待确认。",
            ],
        },
    },
    "pi05": {
        "description": {
            "en": "π0.5 open-world upgrade of π0's flow-matching VLA; `pi05_libero` is the in-tree reference.",
            "zh": "π0.5 是 π0 flow-matching VLA 的开放世界升级；树内参考路径为 `pi05_libero`。",
        },
        "use_cases": {
            "en": [
                "Start on `pi05` for flow-matching decode on the open-world checkpoint. Launch `worldfoundry-eval run pi05`. The validated in-tree reference is `pi05_libero`.",
            ],
            "zh": [
                "从 `pi05` 开始，用开放世界 checkpoint 跑 flow-matching 解码。启动 `worldfoundry-eval run pi05`。树内已验证参考是 `pi05_libero`。",
            ],
        },
        "contract_extra": {
            "en": [
                "Artifact: `pi05_action_trace.json`. Keep the official OpenPI runtime with the pi05-specific config.",
            ],
            "zh": [
                "产物：`pi05_action_trace.json`。使用官方 OpenPI 运行时与 pi05 专用配置。",
            ],
        },
        "run_extra": {
            "en": [
                "Stage `gs://openpi-assets` with gsutil/gcloud. Status: Verified on the `pi05_libero` path.",
            ],
            "zh": [
                "用 gsutil/gcloud 就位 `gs://openpi-assets`。状态：`pi05_libero` 路径已验证。",
            ],
        },
    },
    "openpi": {
        "description": {
            "en": "Physical Intelligence OpenPI runtime (Apache-2.0): shared infra for π0 / π0-FAST / π0.5; GCS weights.",
            "zh": "Physical Intelligence 的 OpenPI 运行时（Apache-2.0）：π0 / π0-FAST / π0.5 的共享基础设施；权重在 GCS。",
        },
        "use_cases": {
            "en": [
                "Use this page for shared OpenPI infra — variants `pi0-base`, `pi05-libero`, `pi0-fast-base`, `pi0-fast-droid`. Jump to the π0 / π0-FAST / π0.5 cards for a specific algorithm. Checkpoint-backed `pi05_libero` is the validated path.",
            ],
            "zh": [
                "本页是共享 OpenPI 基础设施——variant 包括 `pi0-base`、`pi05-libero`、`pi0-fast-base`、`pi0-fast-droid`。具体算法转到 π0 / π0-FAST / π0.5。已验证路径是带 checkpoint 的 `pi05_libero`。",
            ],
        },
        "contract_extra": {
            "en": [
                "Action schema is emitted via the OpenPI `create_trained_policy_infer` helper. Runtime lives under `worldfoundry/synthesis/action_generation/openpi`.",
            ],
            "zh": [
                "动作 schema 经 OpenPI 的 `create_trained_policy_infer` 写出。运行时在 `worldfoundry/synthesis/action_generation/openpi`。",
            ],
        },
        "run_extra": {
            "en": [
                "Stage `gs://openpi-assets` with gsutil. License Apache-2.0. Status: Verified on shared infra / `pi05_libero`.",
            ],
            "zh": [
                "用 gsutil 就位 `gs://openpi-assets`。License Apache-2.0。状态：共享基础设施 / `pi05_libero` 已验证。",
            ],
        },
    },
    "openvla": {
        "description": {
            "en": "Open 7B VLA: Prismatic VLM emits 7-DoF action tokens; base route verified with `unnorm_key=bridge_orig`.",
            "zh": "开源 7B VLA：Prismatic VLM 输出 7-DoF 动作 token；基座路由以 `unnorm_key=bridge_orig` 已验证。",
        },
        "use_cases": {
            "en": [
                "Start on `openvla-7b`. Launch `worldfoundry-eval run openvla-7b`. Switch to `openvla-libero-10` and `unnorm_key=libero_10` only for LIBERO. Continuous / chunked OpenVLA is the [OpenVLA-OFT](/docs/guides/supported-models/openvla-oft) sibling.",
            ],
            "zh": [
                "从 `openvla-7b` 开始。启动 `worldfoundry-eval run openvla-7b`。只有打 LIBERO 时才换 `openvla-libero-10` 与 `unnorm_key=libero_10`。连续/分块线是姊妹卡 [OpenVLA-OFT](/zh/docs/guides/supported-models/openvla-oft)。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `image`, `instruction`, `prompt`. Optional `video` (unused on the recorded predict). Default `unnorm_key=bridge_orig`. Artifacts: `openvla_action_trace.json`, `actions`, `rollout_metrics`. Vision is SigLIP 224px + DINOv2; 256 bins on [-1, 1] then 7-DoF de-norm. Host runs fp16.",
            ],
            "zh": [
                "必填 `image`、`instruction`、`prompt`。可选 `video`（记录的 predict 不用）。默认 `unnorm_key=bridge_orig`。产物：`openvla_action_trace.json`、`actions`、`rollout_metrics`。视觉为 SigLIP 224px + DINOv2；[-1, 1] 上 256 bins 再反归一化到 7-DoF。主机跑 fp16。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128` (Python 3.11, CUDA 11.3). Stage `openvla/openvla-7b` and the LIBERO-10 fine-tune locally. License MIT. Status: Verified on the base card.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`（Python 3.11，CUDA 11.3）。就位 `openvla/openvla-7b` 与 LIBERO-10 微调。License MIT。状态：基座卡已验证。",
            ],
        },
    },
    "cosmos-predict-2": {
        "description": {
            "en": "Previous-generation NVIDIA world foundation model (2B/14B video2world); T5 + Rectified Flow AB2; gated.",
            "zh": "上一代 NVIDIA 世界基础模型（2B/14B video2world）；T5 + Rectified Flow AB2；权重 gated。",
        },
        "use_cases": {
            "en": [
                "Start on `cosmos-predict2-2b-video2world` unless you need the 14B variant. Launch `worldfoundry-eval run cosmos-predict-2`. Prefer [Cosmos Predict 2.5](/docs/guides/supported-models/cosmos-predict-2.5) unless you specifically need this v2 conditioner.",
            ],
            "zh": [
                "从 `cosmos-predict2-2b-video2world` 开始，除非需要 14B。启动 `worldfoundry-eval run cosmos-predict-2`。除非明确需要 v2 conditioner，否则优先 [Cosmos Predict 2.5](/zh/docs/guides/supported-models/cosmos-predict-2.5)。",
            ],
        },
        "contract_extra": {
            "en": [
                "Inputs: image + video. Artifact: `cosmos-predict2-2b-video2world.mp4`.",
            ],
            "zh": [
                "输入：image + video。产物：`cosmos-predict2-2b-video2world.mp4`。",
            ],
        },
        "run_extra": {
            "en": [
                "Stage gated `nvidia/Cosmos-Predict2-2B-Video2World` (and 14B). NVIDIA Open Model License. Status: Runner parity pending.",
            ],
            "zh": [
                "就位 gated 仓库 `nvidia/Cosmos-Predict2-2B-Video2World`（及 14B）。NVIDIA Open Model License。状态：Runner 一致性待确认。",
            ],
        },
    },
    "cosmos-predict-2.5": {
        "description": {
            "en": "NVIDIA world foundation model: text/image/video in, world video out. 2B text-to-world is the verified route.",
            "zh": "NVIDIA 世界基础模型：文本/图像/视频入，世界视频出。已验证路径是 2B 文生世界。",
        },
        "use_cases": {
            "en": [
                "Start on `cosmos-predict-2.5-2b`. Launch `worldfoundry-eval run cosmos-predict-2.5-2b --pipeline.task-profile text-to-world-validation`. Use `cosmos-predict-2.5-14b` only when you accept that line still needs GPU output validation.",
            ],
            "zh": [
                "从 `cosmos-predict-2.5-2b` 开始。启动 `worldfoundry-eval run cosmos-predict-2.5-2b --pipeline.task-profile text-to-world-validation`。只有接受 14B 仍需 GPU 输出验证时才换 `cosmos-predict-2.5-14b`。",
            ],
        },
        "contract_extra": {
            "en": [
                "2B text-to-world: 1280×704, 93 frames, 35 steps, 16 fps. Default artifact: `cosmos-predict2.5-2b.mp4` (14B writes `cosmos-predict2.5-14b.mp4`).",
            ],
            "zh": [
                "2B 文生世界：1280×704、93 帧、35 步、16 fps。默认产物：`cosmos-predict2.5-2b.mp4`（14B 写 `cosmos-predict2.5-14b.mp4`）。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128`. Stage gated `nvidia/Cosmos-Predict2.5-2B`. Status: Verified on 2B text-to-world; 14B is Runner parity pending.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`。就位 gated `nvidia/Cosmos-Predict2.5-2B`。状态：2B 文生世界已验证；14B 为 Runner 一致性待确认。",
            ],
        },
    },
    "cosmos3": {
        "description": {
            "en": "Newest Cosmos generation: joint image/video/sound plus policy and dynamics heads; Nano/Super.",
            "zh": "目前记录的最新 Cosmos 一代：联合图像/视频/声音，外加策略与动力学头；Nano/Super。",
        },
        "use_cases": {
            "en": [
                "Start on `cosmos3-nano`. Launch `worldfoundry-eval run cosmos3-nano`. Switch to `cosmos3-super` only when you need that scale. This infer-only recipe excludes reasoner, training, and vLLM serving.",
            ],
            "zh": [
                "从 `cosmos3-nano` 开始。启动 `worldfoundry-eval run cosmos3-nano`。需要更大规格再换 `cosmos3-super`。本条目是仅推理配方，不含 reasoner、训练与 vLLM serving。",
            ],
        },
        "contract_extra": {
            "en": [
                "Joint stack: text-to-image/video, image/video-to-video, synchronized sound, and structured action inference (policy, forward, inverse dynamics). Nano structural check: 814 transformer / 196 VAE / 182 AVAE tensors.",
            ],
            "zh": [
                "联合栈：文生图/视频、图/视频到视频、同步声音，以及结构化动作推理（策略、前向/逆向动力学）。Nano 结构检查：814 transformer / 196 VAE / 182 AVAE tensor。",
            ],
        },
        "run_extra": {
            "en": [
                "Dedicated `worldfoundry-cosmos3-cu128`. Status: Runner parity pending — end-to-end GPU output must be rerun for the native path.",
            ],
            "zh": [
                "独立环境 `worldfoundry-cosmos3-cu128`。状态：Runner 一致性待确认——原生路径的端到端 GPU 输出必须重跑。",
            ],
        },
    },
    "cosmos-transfer-2.5": {
        "description": {
            "en": "Cosmos sim-to-real arm: edge/control video in, photoreal world video out. 2B edge route is Verified.",
            "zh": "Cosmos 的 sim-to-real 分支：边缘/控制视频入，照片级世界视频出。2B 边缘路径已验证。",
        },
        "use_cases": {
            "en": [
                "Start on the 2B edge-controlled variant. Launch `worldfoundry-eval run cosmos-transfer-2.5-2b` with a raw RGB control video.",
            ],
            "zh": [
                "从 2B 边缘控制 variant 开始。启动 `worldfoundry-eval run cosmos-transfer-2.5-2b`，并提供原始 RGB 控制视频。",
            ],
        },
        "contract_extra": {
            "en": [
                "Native route extracts Canny edges automatically; set `control_is_preprocessed` only when supplying edge maps. Recorded run: 35 denoising steps, 93 decoded frames, 1280×704.",
            ],
            "zh": [
                "原生路径会自动提取 Canny 边缘；只有直接提供边缘图时才设 `control_is_preprocessed`。记录运行：35 步去噪、93 帧解码、1280×704。",
            ],
        },
        "run_extra": {
            "en": [
                "Stage the official 2B edge checkpoint plus the Predict2.5 VAE and Reason1 text encoder. Only the 2B edge variant is Verified.",
            ],
            "zh": [
                "就位官方 2B 边缘 checkpoint，以及 Predict2.5 VAE 与 Reason1 文本编码器。仅 2B 边缘 variant 为已验证。",
            ],
        },
    },
    "helios": {
        "description": {
            "en": "PKU-YuanGroup long-video family in 33-frame chunks; Distilled/Mid/Base in a dedicated Helios env.",
            "zh": "北大 Yuan Group 的长视频系列，按 33 帧分块；Distilled/Mid/Base，独立 Helios 环境。",
        },
        "use_cases": {
            "en": [
                "Start on `helios-distilled` for iteration. Launch `worldfoundry-eval run helios-distilled`. Switch to `helios-base` for quality (much longer runtime) or `helios-mid` in between. Helios does not expose keyboard or camera controls.",
            ],
            "zh": [
                "迭代从 `helios-distilled` 开始。启动 `worldfoundry-eval run helios-distilled`。要质量再换 `helios-base`（更慢），中间档是 `helios-mid`。Helios 不提供键盘或相机控制。",
            ],
        },
        "contract_extra": {
            "en": [
                "33-frame chunks; Distilled sessions keep a native 9-latent / 33-frame history and accept `prompt_update` between segments. Recorded Base A100 smoke used a 3-step [1,1,1] pyramid, not the official [20,20,20] recipe.",
            ],
            "zh": [
                "按 33 帧分块；Distilled 会话保留原生 9 latent / 33 帧历史，分段之间接受 `prompt_update`。记录的 Base A100 smoke 用 3 步 [1,1,1] 金字塔，不是官方 [20,20,20] 配方。",
            ],
        },
        "run_extra": {
            "en": [
                "Dedicated `worldfoundry-helios-cu128` plus GPU for full 240-request-frame visual parity. Status: Runner parity pending on the full-horizon path.",
            ],
            "zh": [
                "完整 240 请求帧视觉一致性需要独立环境 `worldfoundry-helios-cu128` 加 GPU。状态：全时程路径为 Runner 一致性待确认。",
            ],
        },
    },
    "cogvideox": {
        "description": {
            "en": "Zhipu THUDM open T2V/I2V baseline; 2B T2V, 5B T2V, and 5B I2V routes are Verified.",
            "zh": "智谱 THUDM 的开源文生/图生视频基线；2B T2V、5B T2V 与 5B I2V 路径均已验证。",
        },
        "use_cases": {
            "en": [
                "Start on `cogvideox-2b-t2v` for a fast prompt. Launch `worldfoundry-eval run cogvideox-2b-t2v`. Move to `cogvideox-5b-t2v` for quality or `cogvideox-5b-i2v` to animate a still.",
            ],
            "zh": [
                "快速试 prompt 从 `cogvideox-2b-t2v` 开始。启动 `worldfoundry-eval run cogvideox-2b-t2v`。要质量换 `cogvideox-5b-t2v`，给静帧加运动用 `cogvideox-5b-i2v`。",
            ],
        },
        "contract_extra": {
            "en": [
                "T2V exposes `num_inference_steps`, `num_frames`, `height`, `width`, and `fps`. I2V adds `guidance_scale`. Weights from THUDM Hugging Face repos or a local `model_path`. Offline batch only.",
            ],
            "zh": [
                "T2V 暴露 `num_inference_steps`、`num_frames`、`height`、`width`、`fps`。I2V 另有 `guidance_scale`。权重来自 THUDM Hugging Face 或本地 `model_path`。仅离线批生成。",
            ],
        },
        "run_extra": {
            "en": [
                "License metadata is Apache-2.0 on 2B and `other` on several 5B / 1.5 cards — review before commercial use. Status: Verified.",
            ],
            "zh": [
                "2B 的 license 元数据为 Apache-2.0，若干 5B / 1.5 卡片为 `other`——商用前需审阅。状态：已验证。",
            ],
        },
    },
    "ltx-video": {
        "description": {
            "en": "Lightricks LTX-Video 0.9.8 distilled I2V on the shared LTX native stack. Family T2V labels are not a local route.",
            "zh": "Lightricks LTX-Video 0.9.8 蒸馏图生视频，走共享 LTX 原生栈。系列级文生视频标签不是本地路径。",
        },
        "use_cases": {
            "en": [
                "Start on `ltx-video-i2v`. Launch `worldfoundry-eval run ltx-video-i2v`. Family-level T2V labels are upstream metadata, not a current local path.",
            ],
            "zh": [
                "从 `ltx-video-i2v` 开始。启动 `worldfoundry-eval run ltx-video-i2v`。系列级文生视频标签是上游能力元数据，不是当前本地路径。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `prompt` and `image`. Artifact: `ltx-video-i2v.mp4`. Two-stage denoise inserts the official spatial upscaler automatically. T5 loads through NativeModuleLoader; Diffusers is not the inference backend.",
            ],
            "zh": [
                "必填 `prompt`、`image`。产物：`ltx-video-i2v.mp4`。两阶段去噪会自动插入官方空间超分。T5 经 NativeModuleLoader 加载，Diffusers 不是推理后端。",
            ],
        },
        "run_extra": {
            "en": [
                "Official HF cards report license `other`. Status: Runner parity pending.",
            ],
            "zh": [
                "官方 HF 卡片记录 license 为 `other`。状态：Runner 一致性待确认。",
            ],
        },
    },
    "wan2.1": {
        "description": {
            "en": "Alibaba Tongyi Lab open video family. The 1.3B T2V 832×480 / 81-frame route is Verified.",
            "zh": "阿里通义开源视频系列。已验证路径是 1.3B 文生视频 832×480 / 81 帧。",
        },
        "use_cases": {
            "en": [
                "Start on `wan2.1-t2v-1.3b`. Launch `worldfoundry-eval run wan2.1-t2v-1.3b`. Use `wan2.1-t2v-14b` or I2V 480p/720p only when you accept Runner parity pending on those lines.",
            ],
            "zh": [
                "从 `wan2.1-t2v-1.3b` 开始。启动 `worldfoundry-eval run wan2.1-t2v-1.3b`。只有接受那几条仍为 Runner 一致性待确认时，才换 `wan2.1-t2v-14b` 或 I2V 480p/720p。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `prompt`. Optional `image` / `video`. Verified 1.3B T2V: 832×480, 81 frames, 50 steps, guidance 6.0, 16 fps, shift 8.0. Artifact: `wan2.1-t2v-1.3b-832x480-81f.mp4`. 14B T2V default 1280×720 shift 5.0; I2V 480p 832×480 / 40 steps / guidance 5.0.",
            ],
            "zh": [
                "必填 `prompt`。可选 `image` / `video`。已验证 1.3B T2V：832×480、81 帧、50 步、guidance 6.0、16 fps、shift 8.0。产物：`wan2.1-t2v-1.3b-832x480-81f.mp4`。14B T2V 默认 1280×720 shift 5.0；I2V 480p 为 832×480 / 40 步 / guidance 5.0。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128` (Python 3.11, CUDA 12.8). Stage Wan-AI checkpoints locally. License Apache-2.0. Catalog VRAM 8 GB on the 1.3B line. Status: Verified on 1.3B T2V.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`（Python 3.11，CUDA 12.8）。就位 Wan-AI checkpoint。License Apache-2.0。1.3B 线 catalog 显存 8 GB。状态：1.3B T2V 已验证。",
            ],
        },
    },
    "wan2.1-vace": {
        "description": {
            "en": "Wan2.1 all-in-one create/edit member: reference images, V2V, and mask/control on the native Wan assembly.",
            "zh": "Wan2.1 的一体创建/编辑成员：参考图、视频到视频与 mask/控制，走原生 Wan 组装。",
        },
        "use_cases": {
            "en": [
                "Start on `wan2.1-vace`. Launch `worldfoundry-eval run wan2.1-vace`. Use this card for reference-image create/edit, not the plain Wan2.1 T2V backbone.",
            ],
            "zh": [
                "从 `wan2.1-vace` 开始。启动 `worldfoundry-eval run wan2.1-vace`。参考图创建/编辑走本卡，不要当成普通 Wan2.1 文生视频骨干。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `prompt`. Optional `image` / `video`. Native defaults: 1280×720, 81 frames, 50 steps, guidance 5.0. Artifact: `wan2.1-vace.mp4`.",
            ],
            "zh": [
                "必填 `prompt`。可选 `image` / `video`。原生默认：1280×720、81 帧、50 步、guidance 5.0。产物：`wan2.1-vace.mp4`。",
            ],
        },
        "run_extra": {
            "en": [
                "Stage `Wan-AI/Wan2.1-VACE-14B` or the 1.3B diffusers variant. Status: Runner parity pending for full checkpoint visual parity.",
            ],
            "zh": [
                "就位 `Wan-AI/Wan2.1-VACE-14B` 或 1.3B diffusers variant。状态：完整 checkpoint 视觉一致性为 Runner 一致性待确认。",
            ],
        },
    },
    "wan2.2": {
        "description": {
            "en": "Alibaba Wan2.2: verified TI2V 5B at 1280×704 / 121 frames / 24 fps; A14B T2V and I2V are pending.",
            "zh": "阿里 Wan2.2：已验证 TI2V 5B（1280×704 / 121 帧 / 24 fps）；A14B 文生/图生仍待确认。",
        },
        "use_cases": {
            "en": [
                "Start on `wan2.2-ti2v-5b`. Launch `worldfoundry-eval run wan2.2-ti2v-5b` — that variant uses the recorded 5B contract. Use `wan2.2-t2v-a14b` / `wan2.2-i2v-a14b` only when you accept Runner parity pending. No V2V on this card.",
            ],
            "zh": [
                "从 `wan2.2-ti2v-5b` 开始。启动 `worldfoundry-eval run wan2.2-ti2v-5b`——该 variant 使用已记录的 5B 契约。只有接受 Runner 一致性待确认时才换 `wan2.2-t2v-a14b` / `wan2.2-i2v-a14b`。本卡没有 V2V。",
            ],
        },
        "contract_extra": {
            "en": [
                "5B: 1280×704 (or 704×1280), 121 frames, 24 fps, 50 UniPC steps, guidance 5.0, flow-shift 5.0. Artifact: `wan2.2-ti2v-5b-1280x704-121f.mp4`. Prompt required; an image switches the same 5B weights to I2V. A14B T2V: 832×480, 81 frames, 16 fps, 40 steps.",
            ],
            "zh": [
                "5B：1280×704（或 704×1280）、121 帧、24 fps、50 步 UniPC、guidance 5.0、flow-shift 5.0。产物：`wan2.2-ti2v-5b-1280x704-121f.mp4`。必填 prompt；加图则同一 5B 权重切到 I2V。A14B T2V：832×480、81 帧、16 fps、40 步。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128`. Stage `Wan-AI/Wan2.2-TI2V-5B` (Apache-2.0, public). Catalog VRAM 24 GB on 5B, 80 GB floor on A14B. Status: Verified on TI2V 5B.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`。就位 `Wan-AI/Wan2.2-TI2V-5B`（Apache-2.0，公开）。5B catalog 显存 24 GB，A14B 地板 80 GB。状态：TI2V 5B 已验证。",
            ],
        },
    },
    "wan-2p7": {
        "description": {
            "en": "DashScope-hosted Wan 2.7 image-to-video API. No local checkpoint on this card.",
            "zh": "DashScope 托管的 Wan 2.7 图生视频 API。本卡没有本地 checkpoint。",
        },
        "use_cases": {
            "en": [
                "Start on `wan-2p7` when local Wan weights are not an option. Launch `worldfoundry-eval run wan-2p7`. For a local verified clip use [Wan2.2](/docs/guides/supported-models/wan2.2) TI2V 5B.",
            ],
            "zh": [
                "本地 Wan 权重不可用时从 `wan-2p7` 开始。启动 `worldfoundry-eval run wan-2p7`。本地已验证片段请用 [Wan2.2](/zh/docs/guides/supported-models/wan2.2) 的 TI2V 5B。",
            ],
        },
        "contract_extra": {
            "en": [
                "Hosted image-to-video. No in-catalog VRAM, artifacts, or weight repos are recorded on this row.",
            ],
            "zh": [
                "托管图生视频。本行未记录 catalog 显存、产物或权重仓库。",
            ],
        },
        "run_extra": {
            "en": [
                "Requires DashScope credentials. Status: Planned as a hosted API route — runner is not applicable locally.",
            ],
            "zh": [
                "需要 DashScope 凭证。状态：托管 API 路径为计划中——本地 runner 不适用。",
            ],
        },
    },
    "hunyuanvideo": {
        "description": {
            "en": "Tencent open T2V/I2V DiT: prompt (and optional image) in, video out. Runner parity pending.",
            "zh": "腾讯开源文生/图生视频 DiT：prompt（及可选 image）入，视频出。Runner 一致性待确认。",
        },
        "use_cases": {
            "en": [
                "Start on `hunyuanvideo-t2v`. Launch `worldfoundry-eval run hunyuanvideo-t2v`. Use `hunyuanvideo-i2v` when a still is the conditioner. The 1.5 revision is a sibling card.",
            ],
            "zh": [
                "从 `hunyuanvideo-t2v` 开始。启动 `worldfoundry-eval run hunyuanvideo-t2v`。有静帧条件时用 `hunyuanvideo-i2v`。1.5 修订版是姊妹卡。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `prompt`. Optional `image` / `video`. Artifact: `hunyuanvideo-t2v.mp4`. Native recipes record 720×1280, 129 frames.",
            ],
            "zh": [
                "必填 `prompt`。可选 `image` / `video`。产物：`hunyuanvideo-t2v.mp4`。原生配方记录 720×1280、129 帧。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128`. Stage `tencent/HunyuanVideo` (license `other`). Status: Runner parity pending.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`。就位 `tencent/HunyuanVideo`（license 为 `other`）。状态：Runner 一致性待确认。",
            ],
        },
    },
    "hunyuanvideo-1.5": {
        "description": {
            "en": "Tencent HunyuanVideo 1.5 T2V/I2V on shared native infra; I2V needs gated FLUX.1-Redux-dev.",
            "zh": "腾讯混元视频 1.5 的文生/图生视频，走共享原生基础设施；图生视频需要 gated 的 FLUX.1-Redux-dev。",
        },
        "use_cases": {
            "en": [
                "Start on `hunyuanvideo-1.5-t2v`. Launch `worldfoundry-eval run hunyuanvideo-1.5-t2v`. Use `hunyuanvideo-1.5-i2v` only after accepting the gated FLUX.1-Redux-dev terms.",
            ],
            "zh": [
                "从 `hunyuanvideo-1.5-t2v` 开始。启动 `worldfoundry-eval run hunyuanvideo-1.5-t2v`。只有接受 gated FLUX.1-Redux-dev 条款后才跑 `hunyuanvideo-1.5-i2v`。",
            ],
        },
        "contract_extra": {
            "en": [
                "Default artifact `hunyuanvideo-1.5-t2v.mp4`. I2V conditioner declares `black-forest-labs/FLUX.1-Redux-dev` as an independent gated vision checkpoint.",
            ],
            "zh": [
                "默认产物 `hunyuanvideo-1.5-t2v.mp4`。图生视频 conditioner 把 `black-forest-labs/FLUX.1-Redux-dev` 声明为独立 gated 视觉 checkpoint。",
            ],
        },
        "run_extra": {
            "en": [
                "Main weights `tencent/HunyuanVideo-1.5` carry `other` license metadata. Status: Runner parity pending for native checkpoint-backed CUDA.",
            ],
            "zh": [
                "主权重 `tencent/HunyuanVideo-1.5` 的 license 元数据为 `other`。状态：原生带 checkpoint 的 CUDA 为 Runner 一致性待确认。",
            ],
        },
    },
    "gr00t": {
        "description": {
            "en": "NVIDIA Isaac GR00T humanoid/manipulation policy (N1–N1.7); N1.7 LIBERO predict is Verified.",
            "zh": "NVIDIA Isaac GR00T 人形/操作策略（N1–N1.7）；N1.7 LIBERO predict 已验证。",
        },
        "use_cases": {
            "en": [
                "Start on `gr00t-n1.7-libero` when you want the recorded CUDA predict. Launch `worldfoundry-eval run gr00t`. Do not start from N1-2B for that evidence.",
            ],
            "zh": [
                "要跑已记录的 CUDA predict 时从 `gr00t-n1.7-libero` 开始。启动 `worldfoundry-eval run gr00t`。不要从 N1-2B 起步去对那条证据。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `image`, `instruction`, `prompt`. Artifact: `gr00t_action_trace.json`. Verified predict stacks local Cosmos-Reason2-2B with `nvidia/GR00T-N1.7-LIBERO/libero_10` on CUDA 11.8.",
            ],
            "zh": [
                "必填 `image`、`instruction`、`prompt`。产物：`gr00t_action_trace.json`。已验证 predict 在 CUDA 11.8 上把本地 Cosmos-Reason2-2B 与 `nvidia/GR00T-N1.7-LIBERO/libero_10` 叠在一起。",
            ],
        },
        "run_extra": {
            "en": [
                "Dedicated `worldfoundry-gr00t` (Python 3.10, CUDA 11.8 or newer). Authenticate for gated NVIDIA repos. NVIDIA Open Model License. Status: Verified on N1.7 LIBERO predict.",
            ],
            "zh": [
                "独立环境 `worldfoundry-gr00t`（Python 3.10，CUDA 11.8 或更新）。gated NVIDIA 仓库需先认证。NVIDIA Open Model License。状态：N1.7 LIBERO predict 已验证。",
            ],
        },
    },
    "diffusion-policy": {
        "description": {
            "en": "Stanford REAL Lab visuomotor diffusion policy; in-tree low-dim PushT only. Dill checkpoints rejected.",
            "zh": "斯坦福 REAL Lab 的视觉运动扩散策略；树内仅低维 PushT。dill checkpoint 会被拒绝。",
        },
        "use_cases": {
            "en": [
                "Start on `diffusion-policy` for the low-dimensional PushT baseline. Launch `worldfoundry-eval run diffusion-policy`. Image-based Diffusion Policy variants are not in-tree.",
            ],
            "zh": [
                "低维 PushT 基线从 `diffusion-policy` 开始。启动 `worldfoundry-eval run diffusion-policy`。图像版 Diffusion Policy variant 不在树内。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `state` (`state_dim` 20, `observation_history` 2). Artifact: `diffusion_policy_action_trace.json`. Runtime rejects executable dill and restores tensor-only state dictionaries under weights-only reload.",
            ],
            "zh": [
                "必填 `state`（`state_dim` 20、`observation_history` 2）。产物：`diffusion_policy_action_trace.json`。运行时拒绝可执行 dill，并以 weights-only 重载恢复纯 tensor state dict。",
            ],
        },
        "run_extra": {
            "en": [
                "Dedicated `worldfoundry-diffusion-policy` (Python 3.10, CUDA 11.8 or newer). Convert the official dill workspace checkpoint once offline. Status: Verified on the converted weights-only probe.",
            ],
            "zh": [
                "独立环境 `worldfoundry-diffusion-policy`（Python 3.10，CUDA 11.8 或更新）。在可信离线环境把官方 dill workspace checkpoint 转换一次。状态：转换后的 weights-only probe 已验证。",
            ],
        },
    },
    "causal-forcing": {
        "description": {
            "en": "Tsinghua causal video student distilled from Wan2.1. Studio default is chunk-wise T2V at 832×480 / 81 frames / 4 steps.",
            "zh": "清华从 Wan2.1 蒸馏的因果视频学生。Studio 默认是 chunk-wise 文生视频：832×480 / 81 帧 / 4 步。",
        },
        "use_cases": {
            "en": [
                "Start on the chunk-wise student for text-to-video. Launch `worldfoundry-eval run causal-forcing`. Image-to-video is frame-wise only — switch `config_path` / `checkpoint_path` before passing a still. This is not Self-Forcing or Rolling-Forcing.",
            ],
            "zh": [
                "文生视频从 chunk-wise 学生开始。启动 `worldfoundry-eval run causal-forcing`。图生视频只支持 frame-wise——传静帧前先改 `config_path` / `checkpoint_path`。这不是 Self-Forcing 或 Rolling-Forcing。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `prompt`. Catalog also marks `image`. Size 832×480, 81 pixel frames (21 latents), 16 fps, guidance 3.0, 4 steps (`[1000, 750, 500, 250]`). Artifact: `causal-forcing.mp4`. `actions` is `seed`. I2V resizes the still to 480×832 and samples 21−1 latents.",
            ],
            "zh": [
                "必填 `prompt`。catalog 也标记 `image`。尺寸 832×480、81 像素帧（21 latent）、16 fps、guidance 3.0、4 步（`[1000, 750, 500, 250]`）。产物：`causal-forcing.mp4`。`actions` 为 `seed`。I2V 把静帧缩到 480×832，采样 21−1 个 latent。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128` (Python 3.11, CUDA 12.8). Stage the Causal-Forcing pair plus a local Wan2.1 layout. Catalog license unknown; catalog VRAM unset. Status: Runner parity pending.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`（Python 3.11，CUDA 12.8）。就位所选 Causal-Forcing 配对以及本地 Wan2.1 布局。catalog license 未知；显存未记录。状态：Runner 一致性待确认。",
            ],
        },
    },
    "dust3r": {
        "description": {
            "en": "NAVER Labs two-view pointmaps, no calibration. Runner Verified on the bundled multi-view case. CC-BY-NC-SA-4.0.",
            "zh": "NAVER Labs 两视图点图，无需标定。捆绑多视图用例上 runner 已验证。CC-BY-NC-SA-4.0。",
        },
        "use_cases": {
            "en": [
                "Start on `dust3r` for uncalibrated two-view or multi-view reconstruction. Launch `worldfoundry-eval run dust3r`. Streaming metric state is [CUT3R](/docs/guides/supported-models/cut3r), not this pairwise card.",
            ],
            "zh": [
                "无标定两视图/多视图重建从 `dust3r` 开始。启动 `worldfoundry-eval run dust3r`。流式米制状态是 [CUT3R](/zh/docs/guides/supported-models/cut3r)，不是这张成对卡片。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `image`. Optional `prompt`, `video`. `actions`: `multi_view_reconstruction`. Artifact: `dust3r_point_cloud.ply`. Checkpoint `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` (512 DPT).",
            ],
            "zh": [
                "必填 `image`。可选 `prompt`、`video`。`actions`：`multi_view_reconstruction`。产物：`dust3r_point_cloud.ply`。权重 `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`（512 DPT）。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128`. Stage the NAVER `512_dpt` file first. License CC-BY-NC-SA-4.0. Catalog VRAM unset. Status: Verified.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`。先就位 NAVER `512_dpt` 文件。License CC-BY-NC-SA-4.0。catalog 显存未记录。状态：已验证。",
            ],
        },
    },
    "go1": {
        "description": {
            "en": "AgiBot + OpenDriveLab generalist VLA. Static load only; GPU numerical rollout pending. CC-BY-NC-SA-4.0.",
            "zh": "AgiBot + OpenDriveLab 通才 VLA。仅静态加载；GPU 数值 rollout 待确认。CC-BY-NC-SA-4.0。",
        },
        "use_cases": {
            "en": [
                "Start on `go1`. Launch `worldfoundry-eval run go1`. This is not [Genie Envisioner](/docs/guides/supported-models/genie-envisioner) and not [Galaxea G0Plus](/docs/guides/supported-models/galaxea-vla).",
            ],
            "zh": [
                "从 `go1` 开始。启动 `worldfoundry-eval run go1`。这不是 [Genie Envisioner](/zh/docs/guides/supported-models/genie-envisioner)，也不是 [Galaxea G0Plus](/zh/docs/guides/supported-models/galaxea-vla)。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `image`, `prompt`, `state`, `control_frequency`. Cameras `cam_high`, `cam_right_wrist`, `cam_left_wrist`. Default `agibot_pretrain`: 3 cameras, frequency 30, action chunk 30. Artifact: `go1_action_trace.json`.",
            ],
            "zh": [
                "必填 `image`、`prompt`、`state`、`control_frequency`。相机 `cam_high`、`cam_right_wrist`、`cam_left_wrist`。默认 `agibot_pretrain`：3 路相机、频率 30、动作块 30。产物：`go1_action_trace.json`。",
            ],
        },
        "run_extra": {
            "en": [
                "Dedicated `worldfoundry-vla-go1` (Python 3.10, CUDA 12.4 or newer). Stage `agibot-world/GO-1`. License CC-BY-NC-SA-4.0. Catalog VRAM unset. Status: Static load only.",
            ],
            "zh": [
                "独立环境 `worldfoundry-vla-go1`（Python 3.10，CUDA 12.4 或更新）。就位 `agibot-world/GO-1`。License CC-BY-NC-SA-4.0。catalog 显存未记录。状态：仅静态加载。",
            ],
        },
    },
    "xvla": {
        "description": {
            "en": "Tsinghua AIR cross-embodiment VLA on Florence-2. All nine released checkpoints GPU-validated. Apache-2.0.",
            "zh": "清华 AIR 的跨本体 VLA，Florence-2 骨干。九个已发布 checkpoint 均已 GPU 验证。Apache-2.0。",
        },
        "use_cases": {
            "en": [
                "Start on `xvla`. Launch `worldfoundry-eval run xvla`. Always pass the `domain_id` that matches the embodiment/checkpoint.",
            ],
            "zh": [
                "从 `xvla` 开始。启动 `worldfoundry-eval run xvla`。必须传入与本体/checkpoint 匹配的 `domain_id`。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `image`, `prompt`, `state`, `domain_id`, `denoising_steps`. Artifacts: `action_trace`, `xvla_action_trace.json`. Official float32 is the default; bf16/fp16 are explicit options. Recorded A100 probe used 10 denoising steps and finite 30×20 actions.",
            ],
            "zh": [
                "必填 `image`、`prompt`、`state`、`domain_id`、`denoising_steps`。产物：`action_trace`、`xvla_action_trace.json`。默认官方 float32；bf16/fp16 是显式选项。记录的 A100 probe 用 10 步去噪，输出有限 30×20 动作。",
            ],
        },
        "run_extra": {
            "en": [
                "Dedicated `worldfoundry-vla-xvla` (Python 3.10, CUDA 12.1 or newer). Stage `2toINF/X-VLA-*` (WidowX, Google-Robot, Libero, …). Runtime forces `trust_remote_code=false`. Status: Verified.",
            ],
            "zh": [
                "独立环境 `worldfoundry-vla-xvla`（Python 3.10，CUDA 12.1 或更新）。就位 `2toINF/X-VLA-*`（WidowX、Google-Robot、Libero 等）。运行时强制 `trust_remote_code=false`。状态：已验证。",
            ],
        },
    },
    "open-sora": {
        "description": {
            "en": "HPC-AI Tech Open-Sora v1.2 T2V via STDiT-v3. 480p / 51 frames / 24 fps / 30 steps. Runner parity pending.",
            "zh": "HPC-AI Tech Open-Sora v1.2 文生视频（STDiT-v3）。480p / 51 帧 / 24 fps / 30 步。Runner 一致性待确认。",
        },
        "use_cases": {
            "en": [
                "Start on `open-sora` for text-to-video only. Launch `worldfoundry-eval run open-sora`. There is no local I2V or V2V on this card. Not [Open-Sora-Plan](/docs/guides/supported-models/open-sora-plan).",
            ],
            "zh": [
                "只作文生视频时从 `open-sora` 开始。启动 `worldfoundry-eval run open-sora`。本卡没有本地 I2V 或 V2V。不是 [Open-Sora-Plan](/zh/docs/guides/supported-models/open-sora-plan)。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `prompt`. Artifact: `open-sora.mp4`. Launch knobs: 480p, 9:16, 51 frames, 24 fps, rectified flow, 30 steps, guidance 7.0, aesthetic 6.5, seed 1024.",
            ],
            "zh": [
                "必填 `prompt`。产物：`open-sora.mp4`。启动旋钮：480p、9:16、51 帧、24 fps、rectified flow、30 步、guidance 7.0、aesthetic 6.5、seed 1024。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128`. Stage `hpcai-tech/OpenSora-STDiT-v3`, `hpcai-tech/OpenSora-VAE-v1.2`, and `DeepFloyd/t5-v1_1-xxl` (all Apache-2.0). Catalog VRAM unset. Status: Runner parity pending.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`。就位 `hpcai-tech/OpenSora-STDiT-v3`、`hpcai-tech/OpenSora-VAE-v1.2` 与 `DeepFloyd/t5-v1_1-xxl`（均为 Apache-2.0）。catalog 显存未记录。状态：Runner 一致性待确认。",
            ],
        },
    },
    "matrix-game-1": {
        "description": {
            "en": "Skywork Matrix-Game 1.0: image + prompt in, `world.mp4` out. Prefer 2.0/3.0 for verified runners.",
            "zh": "Skywork Matrix-Game 1.0：image + prompt 入，`world.mp4` 出。要已验证 runner 请优先 2.0/3.0。",
        },
        "use_cases": {
            "en": [
                "Start on `matrix-game-1` only when you need the first-generation card. Launch `worldfoundry-eval run matrix-game-1`. Prefer [Matrix-Game 2](/docs/guides/supported-models/matrix-game-2) or [Matrix-Game 3](/docs/guides/supported-models/matrix-game-3) for verified runners.",
            ],
            "zh": [
                "只有需要第一代卡片时从 `matrix-game-1` 开始。启动 `worldfoundry-eval run matrix-game-1`。要已验证 runner 请优先 [Matrix-Game 2](/zh/docs/guides/supported-models/matrix-game-2) 或 [Matrix-Game 3](/zh/docs/guides/supported-models/matrix-game-3)。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `image` and `prompt`. `actions`: keyboard, mouse, navigation_token. Artifact: `world.mp4`.",
            ],
            "zh": [
                "必填 `image`、`prompt`。`actions`：keyboard、mouse、navigation_token。产物：`world.mp4`。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128` (Python 3.11, CUDA 12.8). Stage `Skywork/Matrix-Game` locally. Status: Runner parity pending.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`（Python 3.11，CUDA 12.8）。就位 `Skywork/Matrix-Game`。状态：Runner 一致性待确认。",
            ],
        },
    },
    "matrix-game-2": {
        "description": {
            "en": "Skywork Matrix-Game 2.0 interactive world video. Official universal-action route is Verified.",
            "zh": "Skywork Matrix-Game 2.0 交互世界视频。官方 universal-action 路径已验证。",
        },
        "use_cases": {
            "en": [
                "Start on `matrix-game-2-universal-action-validation`. Launch `worldfoundry-eval run matrix-game-2-universal-action-validation --pipeline.task-profile official-universal-image`. Sibling cards: Matrix-Game 1 and 3.",
            ],
            "zh": [
                "从 `matrix-game-2-universal-action-validation` 开始。启动 `worldfoundry-eval run matrix-game-2-universal-action-validation --pipeline.task-profile official-universal-image`。姊妹卡：Matrix-Game 1 与 3。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `image` and `official_bench_actions`. Recorded official defaults: 150 frames, 12 fps, seed 42, size 352×640. Artifacts: `matrix-game-2-universal-ref-image-seed42-15f.mp4`, `matrix_game2_official_universal_seed42_15f.mp4`.",
            ],
            "zh": [
                "必填 `image`、`official_bench_actions`。记录的官方默认：150 帧、12 fps、seed 42、尺寸 352×640。产物：`matrix-game-2-universal-ref-image-seed42-15f.mp4`、`matrix_game2_official_universal_seed42_15f.mp4`。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128`. Stage `Skywork/Matrix-Game-2.0` (MIT). Catalog VRAM 24 GB. Status: Verified.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`。就位 `Skywork/Matrix-Game-2.0`（MIT）。catalog 显存 24 GB。状态：已验证。",
            ],
        },
    },
    "matrix-game-3": {
        "description": {
            "en": "Skywork Matrix-Game 3.0 higher-resolution interactive world video. Official cityscape route is Verified.",
            "zh": "Skywork Matrix-Game 3.0 更高分辨率交互世界视频。官方 cityscape 路径已验证。",
        },
        "use_cases": {
            "en": [
                "Start on `matrix-game-3`. Launch `worldfoundry-eval run matrix-game-3 --pipeline.task-profile official-cityscape-image`.",
            ],
            "zh": [
                "从 `matrix-game-3` 开始。启动 `worldfoundry-eval run matrix-game-3 --pipeline.task-profile official-cityscape-image`。",
            ],
        },
        "contract_extra": {
            "en": [
                "Required `prompt` and `image`. Recorded cityscape knobs: `num_iterations=12`, `num_inference_steps=3`, size 704×1280, 17 fps, seed 42. Artifact: `matrix-game-3.mp4`.",
            ],
            "zh": [
                "必填 `prompt`、`image`。记录的 cityscape 旋钮：`num_iterations=12`、`num_inference_steps=3`、尺寸 704×1280、17 fps、seed 42。产物：`matrix-game-3.mp4`。",
            ],
        },
        "run_extra": {
            "en": [
                "Unified `worldfoundry-unified-cu128`. Stage `Skywork/Matrix-Game-3.0`. Status: Verified.",
            ],
            "zh": [
                "统一环境 `worldfoundry-unified-cu128`。就位 `Skywork/Matrix-Game-3.0`。状态：已验证。",
            ],
        },
    },
}
