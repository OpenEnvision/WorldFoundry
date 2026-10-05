"""Shared paths and captions for official paper teaser / overview figures.

Body figures are never ``paper.png`` (that file is the title-adjacent thumbnail).
``CAPTIONS`` are one-sentence explanations shown under each figure, not short labels.
"""

from __future__ import annotations

from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MODELS = DOCS_ROOT / "public" / "models"

TEASER_NAMES = ("teaser.png", "teaser.jpg", "teaser.webp", "fig1.png")
OVERVIEW_NAMES = (
    "overview.png",
    "overview.jpg",
    "overview.webp",
    "overall.png",
    "architecture.png",
    "arch.png",
)

DEFAULT_TEASER = ("Official paper teaser.", "论文官方 Teaser。")
DEFAULT_OVERVIEW = ("Official paper method overview.", "论文方法总览。")

# Models that reuse a sibling's already-curated assets (same official paper).
SHARE_FROM: dict[str, str] = {
    "pi0-fast": "pi0",
    "being-h07": "being-h05",
    "openpi": "pi05",
    "bernini-r-1.3b": "bernini",
    "bernini-r-14b": "bernini",
    "cosmos-transfer-2.5": "cosmos-predict-2.5",
    "lingbot-world-act": "lingbot-world",
    "longvie-1": "longvie-2",
    "hyworld-worldgen": "hy-world-2.0",
    "depth-anything-v3": "depth-anything-v3-prior",
    "db-cogact": "dm0",
    "joyai-echo-longvideo": "joyai-echo-wm",
    "ltx-2.5": "ltx-2.x",
}

CAPTIONS: dict[str, dict[str, tuple[str, str]]] = {
    "a1": {
        "overview": (
            "Method overview: a 7B VLM plus a flow-matching or MLP action head, with budget-aware dynamic-exit acceleration.",
            "论文方法总览：7B VLM 加上 flow-matching 或 MLP 动作头，并用预算感知的动态退出加速推理。",
        ),
    },
    "abot-m0": {
        "teaser": (
            "Official paper teaser: ABot-M0.5 as a granularity-aligned world-action model for mobile manipulation.",
            "论文官方 Teaser：ABot-M0.5 作为粒度对齐的移动操作世界-动作模型。",
        ),
        "overview": (
            "Method overview: a dual-level Mixture-of-Transformers that jointly predicts video latents and disentangled mobile/manipulation actions.",
            "论文方法总览：双层混合 Transformer，联合预测视频潜变量并拆分移动与操作动作。",
        ),
    },
    "abot-world-0-5b-lf": {
        "teaser": (
            "Official paper teaser: ABot-World-0 as a real-time interactive world simulator on a single RTX 5090.",
            "论文官方 Teaser：ABot-World-0 在单张 RTX 5090 上做实时交互世界模拟。",
        ),
        "overview": (
            "Method overview: a bidirectional action-conditioned teacher is distilled into a causal LongForcing student for low-latency rollout.",
            "论文方法总览：先把双向动作条件教师蒸馏成因果 LongForcing 学生，再做低延迟滚动。",
        ),
    },
    "ac3d": {
        "teaser": (
            "Official paper teaser: AC3D camera-controlled video samples with consistent wide-angle motion.",
            "论文官方 Teaser：AC3D 的相机可控视频样例，广角运动保持一致。",
        ),
        "overview": (
            "Method overview: frozen DiT-XL video blocks with a lightweight trainable camera DiT-XS ControlNet stream.",
            "论文方法总览：冻结的 DiT-XL 视频模块，加上可训练的轻量相机 DiT-XS ControlNet 支路。",
        ),
    },
    "act": {
        "overview": (
            "Method overview: Action Chunking Transformer’s architecture from the paper.",
            "论文方法总览：Action Chunking Transformer 论文中的方法架构。",
        ),
    },
    "adaworld": {
        "teaser": (
            "Official paper teaser: AdaWorld contrasts action-agnostic pretraining with latent-action pretraining that transfers actions and adapts with few interactions.",
            "论文官方 Teaser：AdaWorld 用潜动作预训练对照无动作预训练，从而少交互即可迁移动作并适配新环境。",
        ),
        "overview": (
            "Method overview: a latent-action encoder conditions an autoregressive diffusion world model for next-frame prediction and rollout.",
            "论文方法总览：潜动作编码器条件化自回归扩散世界模型，用于下一帧预测与滚动生成。",
        ),
    },
    "ahawam": {
        "teaser": (
            "Official paper teaser: Overview of AHA-WAM. AHA-WAM connects past observations, future-oriented world planning, and fast closed-loop action execution: a slow world planner maintains reusable memory and planning context, while a fast action.",
            "论文官方 Teaser：AHA-WAM 的方法论文配图。",
        ),
        "overview": (
            "Method overview: AHA-WAM architecture and attention mask. AHA-WAM decouples world planning and action execution into a slow video-DiT planner and a fast action-DiT executor. The video branch is trained with a fully causal mask to.",
            "论文方法总览：AHA-WAM 的视频生成架构。",
        ),
    },
    "alayaworld": {
        "teaser": (
            "Official paper teaser: Interactive world simulation across diverse scenes. AlayaWorld synthesizes explorable worlds that span first- and third-person viewpoints, real-world, game, and synthetic domains, and both indoor and outdoor.",
            "论文官方 Teaser：AlayaWorld 的交互世界模型论文配图。",
        ),
    },
    "allegro": {
        "teaser": (
            "Official paper teaser: Allegro is capable of generating high-quality, dynamic videos from various descriptive text inputs.",
            "论文官方 Teaser：Allegro 的视频生成论文配图。",
        ),
    },
    "animatediff": {
        "overview": (
            "Method overview: AnimateDiff inserts a temporal motion module into a frozen text-to-image U-Net, then optionally adapts it with MotionLoRA.",
            "论文方法总览：AnimateDiff 在冻结的文生图 U-Net 中插入时间运动模块，必要时再用 MotionLoRA 适配新运动模式。",
        ),
    },
    "astra": {
        "overview": (
            "Method overview: Astra generates future video chunk by chunk from an image, actions, and optional memory.",
            "论文方法总览：Astra 根据初始图像、动作与可选记忆，按块自回归生成未来视频。",
        ),
    },
    "being-h05": {
        "teaser": (
            "Official paper teaser: UniHand 2.0 at a glance — the human, robot, and vision-text mix that Being-H0.5 trains on.",
            "论文官方 Teaser：UniHand 2.0 一览——Being-H0.5 所训练的人类演示、机器人操作与视觉-文本数据。",
        ),
        "overview": (
            "Method overview: a Mixture-of-Transformers with shared attention between understanding and action experts, plus a unified action space.",
            "论文方法总览：混合 Transformer，理解专家与动作专家共享注意力，并使用统一动作空间。",
        ),
    },
    "bernini": {
        "overview": (
            "Method overview: an MLLM semantic planner plus a DiT renderer with segment-aware 3D RoPE on a unified multimodal sequence.",
            "论文方法总览：MLLM 语义规划器加上 DiT 渲染器，并在统一多模态序列上使用片段感知 3D RoPE。",
        ),
    },
    "cameractrl": {
        "teaser": (
            "Official paper teaser: CameraCtrl steers camera trajectories for general and personalized text-to-video.",
            "论文官方 Teaser：CameraCtrl 为通用与个性化文生视频控制相机轨迹。",
        ),
        "overview": (
            "Method overview: a camera encoder injects trajectory features into a pretrained video diffusion backbone.",
            "论文方法总览：相机编码器把轨迹特征注入预训练视频扩散骨干。",
        ),
    },
    "causal-forcing": {
        "teaser": (
            "Official paper teaser: Limitations of existing methods. While distilling from the same bidirectional base model, SOTA autoregressive diffusion distillation methods like Self-Forcing still lag significantly behind standard DMD, which.",
            "论文官方 Teaser：Causal-Forcing 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: Qualitative comparisons with existing methods. Our method achieves substantially higher dynamics and better visual quality than existing distilled autoregressive video models (Causvid and Self Forcing), while.",
            "论文方法总览：Causal-Forcing 的视频生成架构。",
        ),
    },
    "causal-rcm": {
        "teaser": (
            "Official paper teaser: 5 random video samples from 4-step sCM, DMD2, SiD, and rCM on Wan2.1 1.3B. rCM resolves the quality issues of sCM while showing clear superiority to DMD2/SiD in generation diversity, exhibiting highly similar object.",
            "论文官方 Teaser：Causal-rCM 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: Illustration of rCM. Left: the forward consistency objective of sCM propagates error from small to large times; Right: reverse-divergence minimization serves as a long-skip regularizer.",
            "论文方法总览：Causal-rCM 的方法架构。",
        ),
    },
    "cogact": {
        "overview": (
            "Method overview: Overview of our architecture. Our model is componentized into three parts: 1) a vision module encoding information from the current image observation into visual tokens; 2) a language module that integrates the.",
            "论文方法总览：CogACT 的方法架构。",
        ),
    },
    "cogvideox": {
        "teaser": (
            "Official paper teaser: a collage of CogVideoX-generated video frames from the paper.",
            "论文官方 Teaser：CogVideoX 论文中的生成视频拼图。",
        ),
        "overview": (
            "Method overview: an expert Transformer denoises 3D-causal-VAE video latents, with separate text and vision AdaLN paths.",
            "论文方法总览：专家 Transformer 在 3D 因果 VAE 视频潜空间中去噪，文本与视觉各走一路 AdaLN。",
        ),
    },
    "cosmos-predict-2": {
        "overview": (
            "Method overview: Cosmos-Predict takes a current state and a control signal and predicts the next world state.",
            "论文方法总览：Cosmos-Predict 根据当前状态与控制信号预测下一时刻的世界状态。",
        ),
    },
    "cosmos-predict-2.5": {
        "teaser": (
            "Official paper teaser: Cosmos Predict 2.5 world-foundation-model samples from the paper.",
            "论文官方 Teaser：Cosmos Predict 2.5 论文中的世界基础模型样例。",
        ),
        "overview": (
            "Method overview: Cosmos Predict 2.5’s world-model architecture from the paper.",
            "论文方法总览：Cosmos Predict 2.5 论文中的世界模型架构。",
        ),
    },
    "cosmos3": {
        "teaser": (
            "Official paper teaser: Cosmos3 world-model samples from the paper.",
            "论文官方 Teaser：Cosmos3 论文中的世界模型样例。",
        ),
        "overview": (
            "Method overview: Cosmos3’s world-model architecture from the paper.",
            "论文方法总览：Cosmos3 论文中的世界模型架构。",
        ),
    },
    "cut3r": {
        "overview": (
            "Method overview: Continuous 3D Perception. Given a stream of RGB images as input, our approach enables dense 3D reconstruction in an online, continuous manner, estimating both camera parameters and dense 3D geometry with each.",
            "论文方法总览：CUT3R 的几何与深度架构。",
        ),
    },
    "dap": {
        "teaser": (
            "Official paper teaser: 360° RGB panoramas paired with predicted metric depth, plus a radar comparison against other panoramic depth models.",
            "论文官方 Teaser：360° RGB 全景与预测度量深度成对展示，并与其他全景深度模型作雷达图对比。",
        ),
        "overview": (
            "Method overview: DAP’s three-stage pipeline — scene-invariant labeling, realism filtering, then training on pseudo-labeled panoramas.",
            "论文方法总览：DAP 的三阶段流程——场景不变标注、真实感过滤，再在伪标注全景数据上训练。",
        ),
    },
    "depth-anything-v1": {
        "teaser": (
            "Official paper teaser: Depth Anything zero-shot metric-style depth on in-the-wild photos.",
            "论文官方 Teaser：Depth Anything 在野外照片上的零样本深度。",
        ),
    },
    "depth-anything-v2-prior": {
        "teaser": (
            "Official paper teaser: Depth Anything V2 recovers finer structure than V1 and SD-based depth models.",
            "论文官方 Teaser：Depth Anything V2 比 V1 与基于 SD 的深度模型恢复更细的结构。",
        ),
    },
    "depth-anything-v3-prior": {
        "overview": (
            "Method overview: a vanilla DINO transformer with within-view and cross-view attention, then a dual-DPT head for depth, rays, and points.",
            "论文方法总览：原版 DINO Transformer 做视角内与跨视角注意力，再用双 DPT 头预测深度、射线与点。",
        ),
    },
    "dexora-1b": {
        "teaser": (
            "Official paper teaser: Fig. 1: Dexora overview . (a) Motivation : Three illustrative contrasts highlight the need for dual-arm, dual-hand dexterous VLA: piston insertion (requires two arms), book retrieval from a packed shelf (hands with.",
            "论文官方 Teaser：Dexora 1B 的机器人操作论文配图。",
        ),
    },
    "diamond": {
        "overview": (
            "Method overview: a 3D U-Net diffusion world model that conditions on actions by frame-stacking or cross-attention.",
            "论文方法总览：3D U-Net 扩散世界模型，用帧堆叠或交叉注意力注入动作。",
        ),
    },
    "diffusion-policy": {
        "teaser": (
            "Official paper teaser: how a diffusion policy iteratively denoises actions, versus an explicit or implicit visuomotor policy.",
            "论文官方 Teaser：扩散策略如何迭代去噪得到动作，以及它与显式 / 隐式视觉运动策略的对比。",
        ),
    },
    "dino-wm": {
        "teaser": (
            "Official paper teaser: We present DINO-WM , a method for training visual models by using pretrained DINOv2 embeddings of image frames (a). Once trained, given a target observation o T o_{T} , we can directly optimize agent behavior by.",
            "论文官方 Teaser：DINO-WM 的方法论文配图。",
        ),
        "overview": (
            "Method overview: Architecture of DINO-WM . Given observations o t − k : t o_{t-k:t} , we optimize the sequence of actions a t : T − 1 a_{t:T-1} to minimize the predicted loss to the desired goal o g o_{g} . All forward computation is.",
            "论文方法总览：DINO-WM 的方法架构。",
        ),
    },
    "dm0": {
        "overview": (
            "Method overview: The overall architecture of Dexbotic. It introduces the Dexdata format to unify different embodiments. In Model Layer, Dexbotic integrates the open-source vision encoder, LLM and action expert through a unified.",
            "论文方法总览：DM0 的机器人操作架构。",
        ),
    },
    "dreamdojo": {
        "overview": (
            "Method overview: DreamDojo overview. DreamDojo acquires comprehensive physical knowledge from large-scale human datasets by utilizing latent actions as unified labels. After post-training and distillation on the target robots, our.",
            "论文方法总览：DreamDojo 的机器人操作架构。",
        ),
    },
    "dreamx-world-5b": {
        "teaser": (
            "Official paper teaser: DreamX-World 1.0 generates interactive videos with precise camera and event control across photorealistic, game-style, and stylized visual domains.",
            "论文官方 Teaser：DreamX-World 5B 的相机控制论文配图。",
        ),
        "overview": (
            "Method overview: System overview of DreamX-World 1.0. The pipeline integrates camera-accurate data, efficient control, autoregressive distillation, long-range memory, interaction alignment, and optimized serving.",
            "论文方法总览：DreamX-World 5B 的相机控制架构。",
        ),
    },
    "dreamzero": {
        "teaser": (
            "Official paper teaser: Overview . By jointly predicting video and action, World Action Models (WAMs) inherit world physics priors that enable 1) effective learning from diverse, non-repetitive data, 2) open-world generalization, 3).",
            "论文官方 Teaser：NVIDIA DreamZero 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: Model Architecture of DreamZero . The model takes three inputs: visual context (encoded via a VAE), language instructions (via a text encoder), and proprioceptive state (via a state encoder). These are processed by.",
            "论文方法总览：NVIDIA DreamZero 的方法架构。",
        ),
    },
    "dualcamctrl": {
        "overview": (
            "Method overview: DualCamCtrl’s dual-branch framework jointly generates aligned RGB and depth video latents.",
            "论文方法总览：DualCamCtrl 的双分支框架联合生成对齐的 RGB 与深度视频潜变量。",
        ),
    },
    "dust3r": {
        "teaser": (
            "Official paper teaser: DUSt3R reconstructs a kitchen pointmap and cameras from four uncalibrated frames.",
            "论文官方 Teaser：DUSt3R 用四张未标定帧重建厨房点图与相机。",
        ),
        "overview": (
            "Method overview: shared ViT encoders and interacting decoders predict dense pointmaps in camera-1 coordinates.",
            "论文方法总览：共享 ViT 编码器与交互解码器，在相机 1 坐标系下预测密集点图。",
        ),
    },
    "dvlt": {
        "teaser": (
            "Official paper teaser: DéjàView. Given multiple input views (top-left) , DéjàView reconstructs camera poses and consistent depth by repeatedly applying the same transformer block, with the number of refinement steps K K exposed as an.",
            "论文官方 Teaser：Déjà View Looping Transformer 的几何与深度论文配图。",
        ),
        "overview": (
            "Method overview: Method overview. V V input images are encoded by a shared DINOv2 ( Oquab et al., 2024 ) backbone. A single looped transformer block with frame-wise and global attention sub-blocks is then applied recurrently to the.",
            "论文方法总览：Déjà View Looping Transformer 的方法架构。",
        ),
    },
    "dynamicrafter": {
        "overview": (
            "Method overview: Flowchart of the proposed DynamiCrafter . During training, we randomly select a video frame as the image condition of the denoising process through the proposed dual-stream image injection mechanism to inherit visual.",
            "论文方法总览：DynamiCrafter 的视频生成架构。",
        ),
    },
    "easyanimate": {
        "teaser": (
            "Official paper teaser: EasyAnimate generation samples from the paper.",
            "论文官方 Teaser：EasyAnimate 论文中的生成样例。",
        ),
        "overview": (
            "Method overview: EasyAnimate’s pipeline — caption cleanup, VAE latents, a Qwen2-VL-conditioned DiT, then reward post-training.",
            "论文方法总览：EasyAnimate 流程——清洗字幕、VAE 潜变量、Qwen2-VL 条件 DiT，再做奖励后训练。",
        ),
    },
    "echo-infinity": {
        "teaser": (
            "Official paper teaser: Echo-Infinity at a glance. Left: Echo-Infinity can generate extremely long videos in real-time over 24 hours ( > 1.3 M >{1.3 M} frames) , while LongLive, constrained by the absolute RoPE, degrades dramatically for.",
            "论文官方 Teaser：Echo-Infinity 的视频生成论文配图。",
        ),
    },
    "echo-memory-context-k1": {
        "teaser": (
            "Official paper teaser: Abstract teaser and workflow of Echo-Memory. Given a text description, historical observations, and the camera/action state, an action world model must generate chunk-wise video while carrying memory across revisits..",
            "论文官方 Teaser：Echo-Memory 的相机控制论文配图。",
        ),
    },
    "eventvla": {
        "overview": (
            "Method overview: Overview of EventVLA. EventVLA tackles long-horizon, memory-requiring manipulation tasks by storing sparse, task-critical visual evidence. The figure illustrates the (a) non-Markovian challenge, (b) our proposed and.",
            "论文方法总览：EventVLA 的机器人操作架构。",
        ),
    },
    "evoke": {
        "teaser": (
            "Official paper teaser: Two hours of uninterrupted generation. Representative two-hour rollouts under continuous camera control, generated in three steps per chunk without classifier-free guidance. Each row shows six frames sampled.",
            "论文官方 Teaser：Evoke 的相机控制论文配图。",
        ),
    },
    "fantasyworld": {
        "teaser": (
            "Official paper teaser: FantasyWorld overview. Given multimodal inputs (image, text, and camera trajectory), the model generates photorealistic videos along the specified views while constructing an implicit 3D representation for consistent.",
            "论文官方 Teaser：FantasyWorld 的相机控制论文配图。",
        ),
        "overview": (
            "Method overview: Overview.",
            "论文方法总览：FantasyWorld 的方法架构。",
        ),
    },
    "fastvideo-causal-wan2.2": {
        "teaser": (
            "Official paper teaser: Closed-loop simulation workflow. A policy model (here, Alpamayo 1 ( NVIDIA, 2026a ) ) or user sends an action to the AlpaSim simulation runtime. AlpaSim updates the simulation state and forwards the context to.",
            "论文官方 Teaser：FastVideo CausalWan2.2 的视频生成论文配图。",
        ),
    },
    "fastwam": {
        "teaser": (
            "Official paper teaser: Three representative WAM paradigms. (A) Joint-modeling WAMs denoise future video and action tokens together. (B) Causal WAMs first generate future observations and then condition action prediction on the generated.",
            "论文官方 Teaser：FastWAM 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: (a) Fast-WAM model architecture.",
            "论文方法总览：FastWAM 的方法架构。",
        ),
    },
    "flashworld": {
        "teaser": (
            "Official paper teaser: FlashWorld generates high-quality 3D scenes in seconds across diverse environments.",
            "论文官方 Teaser：FlashWorld 在多样场景中于数秒内生成高质量三维场景。",
        ),
        "overview": (
            "Method overview: a dual-mode multi-view latent diffusion model distilled across generation modes.",
            "论文方法总览：双模式多视角潜空间扩散模型，并在生成模式之间做蒸馏。",
        ),
    },
    "framepack": {
        "teaser": (
            "Official paper teaser: Anti-drifting sampling and training methods. We present sampling approaches to generate frames in different temporal orders. The shadowed squares are the generated frames in each iteration, whereas the white squares.",
            "论文官方 Teaser：FramePack 的方法论文配图。",
        ),
    },
    "galaxea-vla": {
        "teaser": (
            "Official paper teaser: Structured action tokenization. Heterogeneous robot actions are decomposed into semantically aligned motion parts, encoded with a residual vector quantizer, and serialized as part-specific action tokens. This.",
            "论文官方 Teaser：Galaxea G0Plus 的机器人操作论文配图。",
        ),
    },
    "gamma-world": {
        "teaser": (
            "Official paper teaser: We propose γ -World, a novel generative multi-agent world model from virtual games to real-world environments. More results and video demos are available on our project page.",
            "论文官方 Teaser：Gamma-World 的交互世界模型论文配图。",
        ),
        "overview": (
            "Method overview: Method overview. γ -World takes synchronized observations and actions from multiple agents as input, tokenizes each agent stream with shared visual and action encoders, and generates future multi-agent rollouts with.",
            "论文方法总览：Gamma-World 的方法架构。",
        ),
    },
    "gen3c": {
        "teaser": (
            "Official paper teaser: Motivation: Our model can generate consistent videos when the camera covers the same region multiple times, while previous work produces severe artifacts due to the lack of explicit modeling of the history.",
            "论文官方 Teaser：GEN3C 的相机控制论文配图。",
        ),
        "overview": (
            "Method overview: Overview of Gen3C . With the user input, which can be a single-view image, multi-view images, or dynamic video(s) , we first build a spatiotemporal 3D cache (Sec. 4.1 ) by predicting the depth for each image and.",
            "论文方法总览：GEN3C 的几何与深度架构。",
        ),
    },
    "genie-envisioner": {
        "teaser": (
            "Official paper teaser: Overview of the Genie Envisioner World Foundation Platform. Genie Envisioner is a unified world foundation platform that integrates manipulation policy learning and evaluation within a single video-generative.",
            "论文官方 Teaser：Genie Envisioner 的机器人操作论文配图。",
        ),
        "overview": (
            "Method overview: Overview.",
            "论文方法总览：Genie Envisioner 的方法架构。",
        ),
    },
    "geocalib-prior": {
        "teaser": (
            "Official paper teaser: GeoCalib estimates camera calibration from a single image by combining learning with geometry.",
            "论文官方 Teaser：GeoCalib 把学习与几何结合起来，从单张图估计相机标定。",
        ),
        "overview": (
            "Method overview: a network predicts a Perspective Field, then Levenberg–Marquardt fits camera parameters.",
            "论文方法总览：网络先预测 Perspective Field，再用 Levenberg–Marquardt 拟合相机参数。",
        ),
    },
    "giga-brain-0": {
        "overview": (
            "Method overview: GigaBrain-0.7 pairs a VLA action system with a VLM planner and a world-model evaluator.",
            "论文方法总览：GigaBrain-0.7 把 VLA 动作系统与 VLM 规划器、世界模型评估器配对。",
        ),
    },
    "giga-world-0": {
        "overview": (
            "Method overview: The framework of GigaWorld-0-Video-Dreamer.",
            "论文方法总览：GigaWorld-0 的视频生成架构。",
        ),
    },
    "giga-world-policy-0.5": {
        "overview": (
            "Method overview: Overview of GigaWorld-Policy-0.5 , an MoT-based action-centered World Action Model. The model consists of a visual expert and an action expert: the visual expert specializes in processing video tokens, while the.",
            "论文方法总览：GigaWorld-Policy-0.5 的机器人操作架构。",
        ),
    },
    "go1": {
        "teaser": (
            "Official paper teaser: Fig. 2 : Data collection pipeline. We embrace a human-in-the-loop framework to ensure high quality, enriched with detailed annotations and error recovery behaviors. Human feedback plays a critical role not only in.",
            "论文官方 Teaser：GO-1 的方法论文配图。",
        ),
    },
    "gr00t": {
        "teaser": (
            "Official paper teaser: GR00T controlling different robot embodiments from vision and language.",
            "论文官方 Teaser：GR00T 根据视觉与语言控制不同机器人本体。",
        ),
        "overview": (
            "Method overview: GR00T’s dual-system stack — a vision-language planner paired with a real-time action expert.",
            "论文方法总览：GR00T 的双系统栈——视觉语言规划器与实时动作专家配对。",
        ),
    },
    "helios": {
        "teaser": (
            "Official paper teaser: Helios generating long video in real time.",
            "论文官方 Teaser：Helios 实时生成长视频的样例。",
        ),
        "overview": (
            "Method overview: an autoregressive video DiT with multi-term memory and a pyramid predictor-corrector over a frozen VAE.",
            "论文方法总览：自回归视频 DiT，带多段记忆，并在冻结 VAE 上做金字塔式预测-校正。",
        ),
    },
    "hma": {
        "overview": (
            "Method overview: Action-Video Dynamics Model from Heterogeneous Robot Interactions. HMA utilizes heterogeneous datasets comprising over 3 million trajectories (videos) from 40 distinct embodiments to pre-train a full dynamics model.",
            "论文方法总览：HMA 的机器人操作架构。",
        ),
    },
    "hunyuan-game-craft": {
        "overview": (
            "Method overview: action-conditioned Double/Single-stream DiT blocks with WASD continuous control and history masks.",
            "论文方法总览：动作条件的双流/单流 DiT，配合 WASD 连续控制与历史掩码。",
        ),
    },
    "hunyuanvideo": {
        "overview": (
            "Method overview: a causal 3D VAE compresses video; an LLM-conditioned diffusion backbone denoises the latents.",
            "论文方法总览：因果 3D VAE 压缩视频，大语言模型条件的扩散主干在潜空间去噪。",
        ),
    },
    "hunyuanvideo-1.5": {
        "teaser": (
            "Official paper teaser: HunyuanVideo 1.5 generation samples from the paper.",
            "论文官方 Teaser：HunyuanVideo 1.5 论文中的生成样例。",
        ),
        "overview": (
            "Method overview: HunyuanVideo 1.5’s Diffusion Transformer for latent video generation.",
            "论文方法总览：HunyuanVideo 1.5 用于潜空间视频生成的 Diffusion Transformer。",
        ),
    },
    "hunyuanworld-1": {
        "teaser": (
            "Official paper teaser: HunyuanWorld 1.0 samples from the paper.",
            "论文官方 Teaser：HunyuanWorld 1.0 论文中的样例。",
        ),
        "overview": (
            "Method overview: An overview of HunyuanWorld 1.0 architecture for 3D world generation. Given a conditioned scene image or textual description, HunyuanWorld 1.0 generates layer-wise 3D worlds in mesh through a staged generative.",
            "论文方法总览：HunyuanWorld 1.0 的视频生成架构。",
        ),
    },
    "hunyuanworld-mirror": {
        "overview": (
            "Method overview: Overview of WorldMirror . Our framework employs Multi-modal Tokenization to encode all inputs (images, optional priors including intrinsics, camera poses, and depth maps) into a unified token sequence. The merged.",
            "论文方法总览：HunyuanWorld-Mirror 的几何与深度架构。",
        ),
    },
    "hunyuanworld-voyager": {
        "teaser": (
            "Official paper teaser: Loading Balancing for SP-Ring.",
            "论文官方 Teaser：HunyuanWorld-Voyager 的方法论文配图。",
        ),
    },
    "hy-embodied-vla": {
        "teaser": (
            "Official paper teaser: Overview of Hy-Embodied-0.5-VLA. An end-to-end VLA system that pairs the Hy-Embodied-0.5-MoT backbone with a flow-matching action expert under a delta-chunk action representation, pre-trained on a 10 10 K-hour.",
            "论文官方 Teaser：Hy-Embodied-0.5-VLA 的机器人操作论文配图。",
        ),
        "overview": (
            "Method overview: Architectural overview of HyVLA-0 . 5 . The framework adopts a MoT architecture to facilitate cross-modal interactions via a shared joint-attention mechanism. To effectively process K K -frame multi-view RGB.",
            "论文方法总览：Hy-Embodied-0.5-VLA 的机器人操作架构。",
        ),
    },
    "hy-world-2.0": {
        "teaser": (
            "Official paper teaser: HY-World 2.0 samples from the paper.",
            "论文官方 Teaser：HY-World 2.0 论文中的样例。",
        ),
    },
    "hy-worldplay": {
        "teaser": (
            "Official paper teaser: Overview of WorldCompass . 1) Starting from environmental prompts and action sequences, we generate shared prefix video clips. At the n n -th target clip, we perform clip-level rollouts to generate a set of candidate.",
            "论文官方 Teaser：HY-WorldPlay 的视频生成论文配图。",
        ),
    },
    "hydra": {
        "overview": (
            "Method overview: Model architecture.",
            "论文方法总览：HyDRA 的方法架构。",
        ),
    },
    "i2vgen-xl": {
        "overview": (
            "Method overview: The overall framework of I2VGen-XL. In the base stage , two hierarchical encoders are employed to simultaneously capture high-level semantics and low-level details of input images, ensuring more realistic dynamics.",
            "论文方法总览：I2VGen-XL 的视频生成架构。",
        ),
    },
    "infinite-vggt": {
        "overview": (
            "Method overview: Framework of StreamVGGT. Our model consists of three main components: an image encoder, a spatio-temporal decoder, and multi-task prediction heads. During training, we utilize full-sequence inputs to provide the.",
            "论文方法总览：Infinite VGGT 的几何与深度架构。",
        ),
    },
    "infinite-world": {
        "overview": (
            "Method overview: Overview of Infinite-World architecture. (a) Hierarchical Pose-free Memory Compressor: The Hierarchical Pose-free Memory Compressor (HPMC) recursively compresses raw historical latents into a fixed memory budget via.",
            "论文方法总览：Infinite-World 的方法架构。",
        ),
    },
    "inspatio-world": {
        "overview": (
            "Method overview: InSpatio-World : Toward a Versatile 4D World Simulator. Top: Our framework enables the synthesis of diverse dynamic scenes from a single video, supporting real-time, high-DoF interactive 4D roaming experiences..",
            "论文方法总览：InSpatio-World 的交互世界模型架构。",
        ),
    },
    "internvla-a1": {
        "teaser": (
            "Official paper teaser: Overview of InternVLA-A1.5. InternVLA-A1.5 unifies understanding, latent foresight, and action by attaching a lightweight expert to a pretrained VLM backbone. It is co-trained on vision-language and robot.",
            "论文官方 Teaser：InternVLA-A1-3B 的机器人操作论文配图。",
        ),
        "overview": (
            "Method overview: Framework of InternVLA-A1.5 . The architecture adopts a Mixture-of-Transformers design comprising a pretrained VLM for multimodal perception and a lightweight unified expert that shares full attention layers with the.",
            "论文方法总览：InternVLA-A1-3B 的机器人操作架构。",
        ),
    },
    "irasim": {
        "overview": (
            "Method overview: Network Architecture of IRASim . (a) shows the general diffusion transformer architecture of IRASim. The input to IRASim includes the historical frames and the given trajectory. (b) Video-level adaptation.",
            "论文方法总览：IRASim 的视频生成架构。",
        ),
    },
    "kairos-sensenova": {
        "teaser": (
            "Official paper teaser: Motivation of Kairos. Existing world models have advanced along representational, generative, interactive, and unified world-action directions, providing useful capabilities such as abstract reasoning, high-fidelity.",
            "论文官方 Teaser：Kairos Sensenova 的交互世界模型论文配图。",
        ),
        "overview": (
            "Method overview: Framework of Kairos. World Understanding extracts a control-sufficient state Z t Z_{t} ; World Generation regularizes physical consistency of Z t Z_{t} through future imagination; World Prediction uses Z t Z_{t} for.",
            "论文方法总览：Kairos Sensenova 的方法架构。",
        ),
    },
    "lagernvs": {
        "overview": (
            "Method overview: Method. The model takes any number of images and, optionally, their camera parameters as input. A large network initialized from a reconstruction model [ 70 ] outputs an intermediate feature representation with.",
            "论文方法总览：LagrNVS 的相机控制架构。",
        ),
    },
    "lapa": {
        "teaser": (
            "Official paper teaser: Problem Formulation. We investigate building a generalist robotic foundation model from human motion videos without action labels.",
            "论文官方 Teaser：Latent Action Pretraining 的机器人操作论文配图。",
        ),
        "overview": (
            "Method overview: Model architecture of our Latent Action Quantization Model.",
            "论文方法总览：Latent Action Pretraining 的方法架构。",
        ),
    },
    "last-r1": {
        "overview": (
            "Method overview: The Overview of LaST-R1. (a) Unlike vanilla RL baselines that strictly optimize actions, (b) our approach utilizes LAPO to jointly optimize an adaptive latent CoT alongside physical execution. By bridging cognitive.",
            "论文方法总览：LaST-R1 的方法架构。",
        ),
    },
    "lda-1b": {
        "teaser": (
            "Official paper teaser: Data generation pipeline: We first curated over 10,680 object meshes from Objaverse [ 63 ] that are suitable for tabletop grasping and randomly selected and placed these objects on the table (left). Next, we used.",
            "论文官方 Teaser：LDA-1B 的方法论文配图。",
        ),
        "overview": (
            "Method overview: GraspVLA consists of an autoregressive vision-language backbone and a flow-matching based action expert. It exploits the synergy between Internet grounding data and synthetic action data with a Progressive Action.",
            "论文方法总览：LDA-1B 的机器人操作架构。",
        ),
    },
    "leworldmodel": {
        "teaser": (
            "Official paper teaser: LeWorldModel Training Pipeline. Given frame observations 𝒐 1 : T {{o}}_{1:T} and actions 𝒂 1 : T {{a}}_{1:T} , the encoder maps frames into low-dimensional latent representations 𝒛 1 : T {{z}}_{1:T} . The predictor.",
            "论文官方 Teaser：LeWorldModel 的方法论文配图。",
        ),
    },
    "lingbot-map": {
        "overview": (
            "Method overview: Pipeline of the proposed LingBot-Map . The framework processes the current view T i T_{i} relative to an initialization set [ T 1 , T n ) [T_{1},T_{n}) . A DINO backbone extracts image features, which are then.",
            "论文方法总览：LingBot-Map 的方法架构。",
        ),
    },
    "lingbot-va": {
        "teaser": (
            "Official paper teaser: LingBot-VA : An Autoregressive World Model for Robotic Manipulation. (1) Pretraining: LingBot-VA is pretrained on diverse in-the-wild videos and robot action data, enabling strong generalization across scenes and.",
            "论文官方 Teaser：LingBot-VA 的机器人操作论文配图。",
        ),
        "overview": (
            "Method overview: Framework overview : LingBot-VA is conditioned by autoregressive diffusion for unified video-action world modeling . We leverage a dual-stream Mixture-of-Transformers (MoT) architecture that interleaves video and.",
            "论文方法总览：LingBot-VA 的交互世界模型架构。",
        ),
    },
    "lingbot-video": {
        "teaser": (
            "Official paper teaser: Samples of Text-to-Image and Text-to-Video tasks generated by LingBot-Video . LingBot-Video can produce images and videos with high visual fidelity, rich details, and strong text-prompt alignment across diverse.",
            "论文官方 Teaser：LingBot-Video 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: Overview of the task-unified single-stream diffusion transformer. Unified inputs are processed by stacked transformer blocks, where timestep modulation controls the attention and Sparse MoE branches; the attention.",
            "论文方法总览：LingBot-Video 的视频生成架构。",
        ),
    },
    "lingbot-vla": {
        "overview": (
            "Method overview: Composite architecture for omni-modal LLMs . The architecture consists of three fully decoupled modules: encoder, foundation model, and decoder.",
            "论文方法总览：LingBot-VLA 的机器人操作架构。",
        ),
    },
    "lingbot-vla-v2": {
        "overview": (
            "Method overview: Overview of LingBot-VLA 2.0 . We revamp the data processing pipeline and curate 60,000 hours of pretraining data, including 50,000 hours of robot trajectories across 20 robot configurations and 10,000 hours of.",
            "论文方法总览：LingBot-VLA 2.0 的机器人操作架构。",
        ),
    },
    "lingbot-world": {
        "teaser": (
            "Official paper teaser: Interactive world simulation across diverse environments. The figure showcases selected samples generated by LingBot-World , demonstrating its capability to synthesize high-fidelity videos in various domains,.",
            "论文官方 Teaser：LingBot-World 的交互世界模型论文配图。",
        ),
        "overview": (
            "Method overview: Pipeline of LingBot-World. The left part shows the pipeline of LingBot-World video generation. LingBot-World uses an image or a video, noisy latents, and user-defined action signals as inputs to generate video.",
            "论文方法总览：LingBot-World 的视频生成架构。",
        ),
    },
    "lingbot-world-v2": {
        "teaser": (
            "Official paper teaser: LingBot-World-Infinity generates infinite worlds in real time, featuring versatile interactions.",
            "论文官方 Teaser：LingBot-World-V2 的方法论文配图。",
        ),
        "overview": (
            "Method overview: Overview of the proposed data engine. Heterogeneous raw videos are temporally segmented, filtered, and routed to category-specific annotation pipelines, producing optimized chunk-wise captions.",
            "论文方法总览：LingBot-World-V2 的视频生成架构。",
        ),
    },
    "loger": {
        "teaser": (
            "Official paper teaser: Overview of a single block of our hybrid memory module. We process the input sequence in consecutive chunks of frames. While each block utilizes frame and bidirectional attention from prior work, we introduce new.",
            "论文官方 Teaser：LoGeR 的方法论文配图。",
        ),
        "overview": (
            "Method overview: Comparison of different methods across varying sequence lengths and scene scales. Although FastVGGT is able to process a larger number of frames during inference, it fails completely on large-scale scenes,.",
            "论文方法总览：LoGeR 的几何与深度架构。",
        ),
    },
    "longcat-video": {
        "teaser": (
            "Official paper teaser: Examples on Text-to-Video , Image-to-Video and Video-Continuation tasks. Video-Continuation supports long video generation as well as interactive generation with multiple instructions. We unify these tasks with a.",
            "论文官方 Teaser：LongCat-Video 的交互世界模型论文配图。",
        ),
        "overview": (
            "Method overview: Overview of training process.",
            "论文方法总览：LongCat-Video 的视频生成架构。",
        ),
    },
    "longvie-2": {
        "overview": (
            "Method overview: Framework of LongVie 2. LongVie 2 serves as a controllable video world model that integrates both dense and sparse control signals to provide world-level guidance for enhanced controllability. A degradation-aware.",
            "论文方法总览：LongVie 2 的交互世界模型架构。",
        ),
    },
    "ltx-video": {
        "teaser": (
            "Official paper teaser: LTX-Video realtime latent-diffusion samples from the paper.",
            "论文官方 Teaser：LTX-Video 论文中的实时潜空间扩散样例。",
        ),
        "overview": (
            "Method overview: LTX-Video’s Video-VAE plus a 3D transformer that denoises in latent space.",
            "论文方法总览：LTX-Video 的 Video-VAE 与在潜空间去噪的 3D Transformer。",
        ),
    },
    "magi-1": {
        "teaser": (
            "Official paper teaser: (Left) Magi-1 performs chunk-wise autoregressive denoising. The video is generated in chunks of 24 frames, where each chunk attends to all previously denoised chunks. Once a chunk reaches a certain denoising level,.",
            "论文官方 Teaser：MAGI-1 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: Model Architecture of Transformer-based VAE.",
            "论文方法总览：MAGI-1 的方法架构。",
        ),
    },
    "matrix-game-2": {
        "overview": (
            "Method overview: Pipelines of Matrix-Game 2.0.",
            "论文方法总览：Matrix-Game 2.0 的交互世界模型架构。",
        ),
    },
    "matrix-game-3": {
        "teaser": (
            "Official paper teaser: Matrix-Game 3.0 introduces precise action control and long-horizon memory retrieval, enabling an interactive world model with long-term memory and real-time performance of up to 40 FPS.",
            "论文官方 Teaser：Matrix-Game 3.0 的交互世界模型论文配图。",
        ),
        "overview": (
            "Method overview: Overview of Matrix-Game 3.0 . Our framework unifies Unreal Engine–based data generation, memory-augmented DiT training with an error buffer, and accelerated real-time deployment. It generates long-horizon training.",
            "论文方法总览：Matrix-Game 3.0 的交互世界模型架构。",
        ),
    },
    "mem-0": {
        "teaser": (
            "Official paper teaser: RMBench Tasks. We illustrate the nine memory-dependent tasks in RMBench along with their key execution steps. Tasks detailed description are shown in Appendix. A.",
            "论文官方 Teaser：Mem-0 的方法论文配图。",
        ),
    },
    "mineworld": {
        "overview": (
            "Method overview: Illustrations of MineWorld model architecture. Visual and action tokenizers convert game states and actions into discrete tokens, which are concatenated and fed into a Transformer decoder as the input. The.",
            "论文方法总览：MineWorld 的交互世界模型架构。",
        ),
    },
    "minwm-hy-action2v": {
        "teaser": (
            "Official paper teaser: Overview of minWM. minWM is a full-stack pipeline that converts T2V/TI2V foundation models into camera-controllable few-step autoregressive world models, covering data construction, controllable fine-tuning, AR.",
            "论文官方 Teaser：minWM 的相机控制论文配图。",
        ),
    },
    "mmaudio": {
        "teaser": (
            "Official paper teaser: In addition to training on audio-visual-(text) datasets, we perform multimodal joint training with high-quality, abundant audio-text data which enables effective data scaling. At inference, MMAudio generates.",
            "论文官方 Teaser：MMAudio 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: We visualize the spectrograms of generated audio (by prior works and our method) and the ground-truth. Note our method generates the audio effects most closely aligned to the ground-truth, while other methods often.",
            "论文方法总览：MMAudio 的方法架构。",
        ),
    },
    "mme-vla": {
        "overview": (
            "Method overview: Framework of MME-VLA Suite. The top part illustrates three memory representations, each with two instantiations: (1) Symbolic Memory summarizes past interactions as high-level abstractions via language-based.",
            "论文方法总览：MME-VLA 的机器人操作架构。",
        ),
    },
    "modelscope-t2v": {
        "teaser": (
            "Official paper teaser: ModelScopeT2V text-to-video samples from the paper.",
            "论文官方 Teaser：ModelScopeT2V 论文中的文生视频样例。",
        ),
        "overview": (
            "Method overview: a VQGAN latent plus a text-conditioned denoising U-Net with temporal attention.",
            "论文方法总览：VQGAN 潜空间加上带时间注意力、文本条件的去噪 U-Net。",
        ),
    },
    "molmoact2": {
        "teaser": (
            "Official paper teaser: Overview of MolmoAct2 . MolmoAct2 is a fully open action reasoning model for real-world deployment. From a suite of high-quality robot datasets that we collect, filter, and curate at scale across three platforms.",
            "论文官方 Teaser：MolmoAct2 的机器人操作论文配图。",
        ),
    },
    "molmobot": {
        "teaser": (
            "Official paper teaser: MolmoBot leverages diverse simulation data to achieve zero-shot sim-to-real transfer on multiple robotic tasks such as pick-and-place and door opening. This unlocks the ability to dramatically scale up the training.",
            "论文官方 Teaser：MolmoBot 的机器人操作论文配图。",
        ),
    },
    "monst3r": {
        "teaser": (
            "Official paper teaser: Watch the video.",
            "论文官方 Teaser：MonST3R 的视频生成论文配图。",
        ),
    },
    "mosaicmem": {
        "teaser": (
            "Official paper teaser: Left: Memory mechanism comparison and visualization. MoscaicMem is a hybrid approach, unifies the strengths of explicit and implicit memory. Right: (A) MosaicMem achieves more accurate camera motion than implicit.",
            "论文官方 Teaser：MosaicMem 的相机控制论文配图。",
        ),
        "overview": (
            "Method overview: Method overview. Left: MosaicMem lifts patches into 3D, then gathers and stitches them in the target view like a mosaic. Middle: Architecture overview. Camera motion is controlled jointly by MosaicMem retrieval and.",
            "论文方法总览：MosaicMem 的相机控制架构。",
        ),
    },
    "motionbricks": {
        "teaser": (
            "Official paper teaser: Figure 1. MotionBricks enables real-time motion control across animation and robotics. All motions are generated by our unified latent neural backbone using the smart primitive interface. Top: We showcase.",
            "论文官方 Teaser：MotionBricks 的机器人操作论文配图。",
        ),
    },
    "motionctrl": {
        "teaser": (
            "Official paper teaser: Figure 1. Control Results of MotionCtrl . MotionCtrl is capable of controlling both camera motion and object motion in videos produced by a video generation model. It can also simultaneously control both types of.",
            "论文官方 Teaser：MotionCtrl 的相机控制论文配图。",
        ),
        "overview": (
            "Method overview: Figure 2. MotionCtrl Framework . MotionCtrl extends the Denoising U-Net structure of LVDM with a Camera Motion Control Module (CMCM) and an Object Motion Control Module (OMCM). As illustrated in (b), the CMCM.",
            "论文方法总览：MotionCtrl 的相机控制架构。",
        ),
    },
    "moverse": {
        "overview": (
            "Method overview: MoVerse pipeline overview. From a single narrow-field-of-view input image, Stage I synthesizes a gravity-aligned 360 ∘ panorama, Stage II lifts the panorama into a persistent 3D Gaussian scaffold, and Stage III.",
            "论文方法总览：MoVerse 的方法架构。",
        ),
    },
    "mvdiffusion": {
        "teaser": (
            "Official paper teaser: MVDiffusion synthesizes consistent multi-view images. Top: generating perspective crops which can be stitched into panorama; Bottom: generating coherent multi-view images from depths.",
            "论文官方 Teaser：MVDiffusion 的几何与深度论文配图。",
        ),
    },
    "neoverse": {
        "overview": (
            "Method overview: Framework of NeoVerse. In the reconstruction part, we propose a pose-free feed-forward 4DGS reconstruction model ( Sec. 3.1 ) with bidirectional motion modeling. The degraded renderings in novel viewpoints from 4DGS.",
            "论文方法总览：NeoVerse 的方法架构。",
        ),
    },
    "oasis-500m": {
        "overview": (
            "Method overview: a ViT-VAE plus an action-conditioned DiT that predicts the next Minecraft frames.",
            "论文方法总览：ViT-VAE 加上动作条件 DiT，预测下一帧 Minecraft 画面。",
        ),
    },
    "octo": {
        "teaser": (
            "Official paper teaser: Octo as a modular generalist policy — flexible goals, observations, and action spaces across robots.",
            "论文官方 Teaser：Octo 作为模块化通用策略——目标、观测与动作空间可换，并覆盖多种机器人。",
        ),
    },
    "omniforcing": {
        "overview": (
            "Method overview: OmniForcing breaks the latency barrier for joint audio-visual generation. Top: Our framework achieves real-time streaming at ∼ 25 FPS with an ultra-low Time-To-First-Chunk (TTFC) of ∼ 0.7s. Bottom: The bidirectional.",
            "论文方法总览：OmniForcing 的方法架构。",
        ),
    },
    "open-dreamer": {
        "teaser": (
            "Official paper teaser: Dreamer 4 learns to solve complex control tasks by reinforcement learning inside of its world model. We decode the imagined training sequences for visualization, showing that the world model has learned to simulate a.",
            "论文官方 Teaser：Open Dreamer 的交互世界模型论文配图。",
        ),
    },
    "open-magvit2": {
        "overview": (
            "Method overview: Overview of Open-MAGVIT2. There are two crucial stages in Open-MAGVIT2. In Stage I {I} : the image is first encoded by MAGVIT-v2 Encoder and subsequently transformed into bits format by Lookup-Free Quantizer (LFQ)..",
            "论文方法总览：Open-MAGVIT2 的方法架构。",
        ),
    },
    "open-sora": {
        "teaser": (
            "Official paper teaser: Open-Sora video generation samples from the paper.",
            "论文官方 Teaser：Open-Sora 论文中的视频生成样例。",
        ),
        "overview": (
            "Method overview: Open-Sora’s video generation stack from the paper.",
            "论文方法总览：Open-Sora 论文中的视频生成架构。",
        ),
    },
    "open-sora-plan": {
        "overview": (
            "Method overview: The model architecture of the Open-Sora Plan consists of a VAE, a Diffusion Transformer, and conditional encoders. The conditional injection encoders enable precise manipulation of individual frames (whether it’s the.",
            "论文方法总览：Open-Sora-Plan 的机器人操作架构。",
        ),
    },
    "openvla": {
        "teaser": (
            "Official paper teaser: OpenVLA performing generalist robot manipulation from language.",
            "论文官方 Teaser：OpenVLA 根据语言做通用机器人操作。",
        ),
        "overview": (
            "Method overview: DinoV2 + SigLIP vision tokens enter Llama 2 7B, then an action detokenizer emits 7-DoF robot actions.",
            "论文方法总览：DinoV2 与 SigLIP 视觉 token 进入 Llama 2 7B，再经动作反分词输出 7 维机器人动作。",
        ),
    },
    "openvla-oft": {
        "teaser": (
            "Official paper teaser: OpenVLA-OFT fine-tunes OpenVLA with parallel decoding for faster robot control.",
            "论文官方 Teaser：OpenVLA-OFT 用并行解码微调 OpenVLA，以加快机器人控制。",
        ),
    },
    "pi0": {
        "teaser": (
            "Official paper teaser: π0 trained on cross-embodiment robot data and internet-scale pretraining, then post-trained for hard and unseen tasks.",
            "论文官方 Teaser：π0 在跨本体机器人数据与互联网预训练上训练，再针对困难与未见任务做后训练。",
        ),
        "overview": (
            "Method overview: a pretrained VLM (SigLIP + Gemma) plus a flow-matching action expert that outputs an action horizon.",
            "论文方法总览：预训练 VLM（SigLIP + Gemma）加上 flow-matching 动作专家，输出一段动作序列。",
        ),
    },
    "pi05": {
        "teaser": (
            "Official paper teaser: π0.5 generalizing from pretraining to open-world robot tasks.",
            "论文官方 Teaser：π0.5 从预训练泛化到开放世界机器人任务。",
        ),
        "overview": (
            "Method overview: π0.5’s pretraining recipe and the action expert used at inference.",
            "论文方法总览：π0.5 的预训练配方，以及推理时使用的动作专家。",
        ),
    },
    "pi3": {
        "teaser": (
            "Official paper teaser: π³ predicting cameras, depth, and point maps from images.",
            "论文官方 Teaser：π³ 从图像预测相机、深度与点图。",
        ),
        "overview": (
            "Method overview: π³’s feed-forward 3D reconstruction transformer from the paper.",
            "论文方法总览：π³ 论文中的前馈三维重建 Transformer。",
        ),
    },
    "pixelsplat": {
        "teaser": (
            "Official paper teaser: Overview. Given a pair of input images, pixelSplat reconstructs a 3D radiance field parameterized via 3D Gaussian primitives. This yields an explicit 3D representation that is renderable in real time, remains.",
            "论文官方 Teaser：pixelSplat 的方法论文配图。",
        ),
        "overview": (
            "Method overview: 3D Gaussians (top) and corresponding depth maps (bottom) predicted by our method. In contrast to light field rendering methods like GPNR [ 47 ] and that of Du et al. [ 10 ] , our method produces an explicit 3D.",
            "论文方法总览：pixelSplat 的几何与深度架构。",
        ),
    },
    "pointworld": {
        "overview": (
            "Method overview: Overview of PointWorld . Given calibrated RGB-D, robot joint-space actions, and a robot description file (URDF), we convert actions to robot flows and concatenate with scene to form a single point cloud serving as an.",
            "论文方法总览：PointWorld 的机器人操作架构。",
        ),
    },
    "pusa-vidgen": {
        "teaser": (
            "Official paper teaser: Previous conventional video diffusion models (b) directly extend image diffusion models (a) utilizing a single scalar timestep on the whole video clip. This straightforward adaption restricts the flexibilities of.",
            "论文官方 Teaser：Pusa VidGen 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: Diverse Applications of FVDM. (a) Standard Video Generation: Implements uniform timestep across frames, [ t , t , … , t ] [t,t,,t] . (b) Image-to-Video Generation: Transforms a static image into a video using a.",
            "论文方法总览：Pusa VidGen 的视频生成架构。",
        ),
    },
    "qwen2.5-omni": {
        "teaser": (
            "Official paper teaser: Qwen2.5-Omni samples from the paper.",
            "论文官方 Teaser：Qwen2.5-Omni 论文中的样例。",
        ),
        "overview": (
            "Method overview: Qwen2.5-Omni’s architecture from the paper.",
            "论文方法总览：Qwen2.5-Omni 论文中的方法架构。",
        ),
    },
    "rdt-1b": {
        "teaser": (
            "Official paper teaser: RDT-1B samples from the paper.",
            "论文官方 Teaser：RDT-1B 论文中的样例。",
        ),
        "overview": (
            "Method overview: RDT-1B’s architecture from the paper.",
            "论文方法总览：RDT-1B 论文中的方法架构。",
        ),
    },
    "real-time-chunking": {
        "teaser": (
            "Official paper teaser: Top: Real-time chunking (RTC) enables the robot to perform highly dexterous and dynamic tasks, such as lighting a match—even in the presence of inference delays in excess of 300 milliseconds, corresponding to more.",
            "论文官方 Teaser：Real-Time Chunking 的机器人操作论文配图。",
        ),
    },
    "recammaster": {
        "overview": (
            "Method overview: Overview of ReCamMaster. Left: The training pipeline of ReCamMaster. A latent diffusion model is optimized to reconstruct the target video V t V_{t} , conditioned on the source video V s V_{s} , target camera pose c.",
            "论文方法总览：ReCamMaster 的相机控制架构。",
        ),
    },
    "rise": {
        "teaser": (
            "Official paper teaser: RISE samples from the paper.",
            "论文官方 Teaser：RISE 论文中的样例。",
        ),
        "overview": (
            "Method overview: RISE’s architecture from the paper.",
            "论文方法总览：RISE 论文中的方法架构。",
        ),
    },
    "roboflamingo": {
        "overview": (
            "Method overview: RoboFlamingo’s architecture from the paper.",
            "论文方法总览：RoboFlamingo 论文中的方法架构。",
        ),
    },
    "rolling-forcing": {
        "teaser": (
            "Official paper teaser: Rolling Forcing performs real-time streaming text-to-video generation at 16 fps on a single GPU and is capable of producing multi-minute-long videos with minimal error accumulation. More results, code, and demo can.",
            "论文官方 Teaser：RollingForcing 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: Illustration of the Rolling Forcing denoising process with T = 4 T=4 . Rolling Forcing jointly denoises a short window of consecutive frames that are assigned progressively higher noise levels and connected by.",
            "论文方法总览：RollingForcing 的方法架构。",
        ),
    },
    "rt-1": {
        "teaser": (
            "Official paper teaser: RT-1 performing diverse real-robot manipulation skills from language.",
            "论文官方 Teaser：RT-1 根据语言执行多样真实机器人操作技能。",
        ),
    },
    "sama-14b": {
        "overview": (
            "Method overview: Overall pipeline. SAMA first performs factorized pre-training (stage 0) on additional perturbed videos by completing a pretext task conditioned on the given captions. It then performs normal supervised fine-tuning.",
            "论文方法总览：SAMA 的视频生成架构。",
        ),
    },
    "sana": {
        "teaser": (
            "Official paper teaser: Sana efficient high-resolution image generation samples from the paper.",
            "论文官方 Teaser：Sana 论文中的高效高分辨率图像生成样例。",
        ),
        "overview": (
            "Method overview: Sana’s linear DiT pipeline for deep-compression latent image generation.",
            "论文方法总览：Sana 用线性 DiT 在深压缩潜空间生成图像。",
        ),
    },
    "sana-wm": {
        "teaser": (
            "Official paper teaser: SANA-WM teaser. From one image and an action trajectory, SANA-WM generates minute-scale 720p worlds with precise control, 64-GPU training, and single-GPU inference.",
            "论文官方 Teaser：SANA-WM 的方法论文配图。",
        ),
        "overview": (
            "Method overview: SANA-WM Architecture. Text, video, and pose tokens pass through alternating GDN and softmax attention blocks. Geometry-aware components (UCPE attention and Plücker mixing) are integrated to enable pose-conditioned.",
            "论文方法总览：SANA-WM 的视频生成架构。",
        ),
    },
    "scope": {
        "teaser": (
            "Official paper teaser: S CO P E executes complex multi-action controls and action-environment interactions (highlighted in red boxes ) across diverse, unseen first-person scenes without retraining.",
            "论文官方 Teaser：SCOPE 的方法论文配图。",
        ),
        "overview": (
            "Method overview: S CO P E architecture. A SCOPE module is inserted into each DiT block. Discrete inputs use cross-attention with visual queries to confine effects to in-scope regions. Continuous inputs use MLP fusion and temporal.",
            "论文方法总览：SCOPE 的视频生成架构。",
        ),
    },
    "self-forcing": {
        "overview": (
            "Method overview: Training paradigms for AR video diffusion models. (a) In Teacher Forcing, the model is trained to denoise each frame conditioned on the preceding clean, ground-truth context frames. (b) In Diffusion Forcing, the.",
            "论文方法总览：Self-Forcing 的视频生成架构。",
        ),
    },
    "shotstream": {
        "teaser": (
            "Official paper teaser: Overview of the ShotStream workflow, which enables real-time, long, multi-shot video generation from streaming prompts.",
            "论文官方 Teaser：ShotStream 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: Architecture of the Bidirectional Next-Shot Teacher Model. To realize ShotStream, we first fine-tune a text-to-video model into a bidirectional next-shot model, which generates subsequent shots conditioned on sparse.",
            "论文方法总览：ShotStream 的视频生成架构。",
        ),
    },
    "show-o": {
        "overview": (
            "Method overview: Show-o unifies autoregressive language with diffusion vision in one LLM block.",
            "论文方法总览：Show-o 在同一个 LLM 块里统一自回归语言与扩散视觉。",
        ),
    },
    "simworld": {
        "teaser": (
            "Official paper teaser: An Overview of the SimWorld Simulator , featuring three key designs: (1) realistic, open-ended world simulation, (2) rich interface for LLM/VLM agents, and (3) diverse physical and social reasoning scenarios.",
            "论文官方 Teaser：SimWorld 的方法论文配图。",
        ),
        "overview": (
            "Method overview: Architecture of SimWorld . SimWorld adopts a hierarchical, closed-loop architecture that decouples agent reasoning from high-performance rendering while maintaining coherent information flow across modules. At its.",
            "论文方法总览：SimWorld 的方法架构。",
        ),
    },
    "skyreels-v2": {
        "teaser": (
            "Official paper teaser: SkyReels-V2 produces stunningly realistic and cinematic high-resolution videos of virtually unlimited length. The model excels at maintaining visual consistency of the main subject across all frames, ensuring no.",
            "论文官方 Teaser：SkyReels-V2 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: Overview of the proposed method.",
            "论文方法总览：SkyReels-V2 的方法架构。",
        ),
    },
    "skyreels-v3": {
        "teaser": (
            "Official paper teaser: Reference Images to Video Results. SkyReels-V3 can facilitate dynamic interplay between different subjects within specified contexts.",
            "论文官方 Teaser：SkyReels V3 的视频生成论文配图。",
        ),
    },
    "solaris": {
        "teaser": (
            "Official paper teaser: Solaris multiplayer Minecraft world-model samples from the paper.",
            "论文官方 Teaser：Solaris 论文中的多人 Minecraft 世界模型样例。",
        ),
        "overview": (
            "Method overview: shared self-attention over multi-player video tokens, then per-player action and first-frame conditioning.",
            "论文方法总览：多玩家视频 token 先做共享自注意力，再按玩家注入动作与首帧条件。",
        ),
    },
    "solarwm": {
        "overview": (
            "Method overview: Table 1 : Release matrix for representative interactive video world models. Release status was verified from official artifacts as of August 18, 2026; SolarWM entries indicate commitments for this release. ✓: public;.",
            "论文方法总览：SolarWM 的交互世界模型架构。",
        ),
    },
    "spatia": {
        "overview": (
            "Method overview: Overview of the training stage of Spatia. Each training video is divided into a target clip, a preceding clip, and a candidate-frame set. Text tokens are omitted for simplicity. (a) A frame is randomly selected from.",
            "论文方法总览：Spatia 的视频生成架构。",
        ),
    },
    "splatt3r": {
        "overview": (
            "Method overview: Method overview. We encode the two uncalibrated images using MASt3R’s pretrained ViT encoder and cross-attention transformers, which we freeze during training. In addition to MASt3R’s prediction head for point.",
            "论文方法总览：Splatt3R 的视频生成架构。",
        ),
    },
    "stable-video-infinity": {
        "teaser": (
            "Official paper teaser: Comparison among (a) video generative DiT, (b) restoration DiT, and (c) our Stable Video Infinity regarding the scheme ( row 1 ), training-test hypothesis gap ( row 2 ), and outcome ( row 3 ).",
            "论文官方 Teaser：Stable Video Infinity 2.0 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: Stable Video Infinity . We (a) inject errors into clean latent to break the error-free hypothesis, (b) approximate predictions via one-step integration to calculate bidirectional errors, and (c) dynamically bank and.",
            "论文方法总览：Stable Video Infinity 2.0 的视频生成架构。",
        ),
    },
    "stable-virtual-camera": {
        "overview": (
            "Method overview: Method. Seva is trained with fixed sequence length as a “ M M -in N N -out” multi-view diffusion model with standard architecture. It conditions on CLIP embeddings, VAE latents of the input views, and their.",
            "论文方法总览：Stable Virtual Camera 的相机控制架构。",
        ),
    },
    "starvla": {
        "teaser": (
            "Official paper teaser: Conceptual view of the unified VLA formulation adopted in StarVLA. A policy π maps visual observations and a language instruction to a future action chunk. The training objective decomposes as ℒ = ℒ action + ℒ aux.",
            "论文官方 Teaser：StarVLA 的机器人操作论文配图。",
        ),
    },
    "starwm": {
        "teaser": (
            "Official paper teaser: Case study comparing our world-model-augmented decision system (StarWM-Agent) with a policy that does not use a world model. Given the current observation, the LLM policy initially proposes Build Supply Depot . A.",
            "论文官方 Teaser：StarWM 的交互世界模型论文配图。",
        ),
        "overview": (
            "Method overview: Framework of our StarWM-Agent. The agent interacts with the SC2 engine via a textual interface and follows a Generate–Simulate–Refine decision loop: the policy first generates an initial action proposal from the.",
            "论文方法总览：StarWM 的方法架构。",
        ),
    },
    "step-video-t2v": {
        "overview": (
            "Method overview: Architecture overview of Step-Video-T2V. Videos are represented by a high-compression Video-VAE, achieving 16x16 spatial and 8x temporal compression ratios. User prompts are encoded using two bilingual pre-trained.",
            "论文方法总览：Step-Video-T2V 的视频生成架构。",
        ),
    },
    "t2v_turbo_t2v": {
        "teaser": (
            "Official paper teaser: T2V-Turbo distilled text-to-video samples from the paper.",
            "论文官方 Teaser：T2V-Turbo 论文中的蒸馏文生视频样例。",
        ),
        "overview": (
            "Method overview: T2V-Turbo’s consistency-distillation stack for few-step video generation.",
            "论文方法总览：T2V-Turbo 用一致性蒸馏做少步视频生成。",
        ),
    },
    "tesseract": {
        "overview": (
            "Method overview: Architecture and Training Overview of TesserAct.",
            "论文方法总览：TesserAct 的方法架构。",
        ),
    },
    "tinyvla": {
        "overview": (
            "Method overview: MobileVLM V2’s architecture. 𝐗 v {X}_{v} and 𝐗 q {X}_{q} indicate image and language instruction, respectively, and 𝐘 a {Y}_{a} refers to the text response from the language model MobileLLaMA. The diagram in the.",
            "论文方法总览：TinyVLA 的机器人操作架构。",
        ),
    },
    "track-anything-prior": {
        "overview": (
            "Method overview: The Pipeline of SAM-Track. The Interactive Tracking Mode is used only in the first frame of the video to obtain annotations, while the Automatic Tracking Mode is called every nth frame thereafter. The S t S^{t} , T t.",
            "论文方法总览：Segment and Track Anything 的交互世界模型架构。",
        ),
    },
    "uni3c": {
        "overview": (
            "Method overview: Figure 3. Pipeline of PCDController, which is built as a lightweight DiT trained from scratch. We first obtain point clouds via monocular depth from the first view. Then, the point clouds are warped and rendered into.",
            "论文方法总览：Uni3C 的几何与深度架构。",
        ),
    },
    "unidepth-v2-prior": {
        "teaser": (
            "Official paper teaser: Fig. 1: We introduce UniDepthV2, a novel approach that directly predicts 3D points in a scene with only one image as input. UniDepthV2 incorporates a camera self-prompting mechanism and leverages a spherical 3D.",
            "论文官方 Teaser：UniDepth V2 的几何与深度论文配图。",
        ),
        "overview": (
            "Method overview: Fig. 2: Model Architecture. UniDepthV2 utilizes solely the input image to generate the 3D output ( 𝐎 {O} ). It bootstraps a dense camera prediction ( 𝐂 {C} ) from the Camera Module, injecting prior knowledge on scene.",
            "论文方法总览：UniDepth V2 的几何与深度架构。",
        ),
    },
    "unik3d-prior": {
        "teaser": (
            "Official paper teaser: UniK3D introduces a novel and versatile approach that delivers precise metric 3D geometry estimation from a single image and for any camera type, ranging from pinhole to panoramic, without requiring any camera.",
            "论文官方 Teaser：UniK3D 的几何与深度论文配图。",
        ),
        "overview": (
            "Method overview: Model architecture. UniK3D utilizes solely the single input image to generate the 3D output point cloud ( 𝐎 {O} ) for any camera. The projective geometry of the camera is predicted by the Angular Module. The camera.",
            "论文方法总览：UniK3D 的几何与深度架构。",
        ),
    },
    "uwm": {
        "overview": (
            "Method overview: Fig. 1: Unified World Models integrates action and video diffusion in a unified transformer architecture controlled by modality-specific diffusion timesteps. The model can be trained on large robotics datasets and.",
            "论文方法总览：Unified World Models 的机器人操作架构。",
        ),
    },
    "vchitect-2-t2v": {
        "teaser": (
            "Official paper teaser: Fig. 1 : High quality videos generated by our 2B model, demonstrating its capabilities in generating videos with high-fidelity and consistent actions.",
            "论文官方 Teaser：Vchitect-2.0 T2V 的视频生成论文配图。",
        ),
        "overview": (
            "Method overview: Fig. 2 : Overview of the parallel transformer architecture used in Vchitect-2.0. The model combines text and video features using a unified framework. The Text Encoder processes input prompts, generating textual.",
            "论文方法总览：Vchitect-2.0 T2V 的视频生成架构。",
        ),
    },
    "versecrafter": {
        "teaser": (
            "Official paper teaser: VerseCrafter enables precise control of camera motion and multi-object motion via a 4D Geometric Control representation built from a static background point cloud and per-object 3D Gaussian trajectories, producing.",
            "论文官方 Teaser：VerseCrafter 的相机控制论文配图。",
        ),
        "overview": (
            "Method overview: Framework of VerseCrafter. Given an input image and a text prompt, we estimate depth and obtain user-specified object masks to construct 4D Geometric Control consisting of a static background point cloud and.",
            "论文方法总览：VerseCrafter 的几何与深度架构。",
        ),
    },
    "vggt-omega": {
        "overview": (
            "Method overview: Architecture Overview. VGGT- Ω appends camera and scene tokens (registers) to image tokens, and then alternates between global attention (or register attention) and frame attention layers. We replace the redundant.",
            "论文方法总览：VGGT-Omega 的几何与深度架构。",
        ),
    },
    "vggt-world": {
        "teaser": (
            "Official paper teaser: From video world models to geometry world models. Video world models predict future RGB in VAE latent space, coupling scene dynamics with appearance reconstruction. As a result, decoded predictions can remain.",
            "论文官方 Teaser：VGGT-World 的几何与深度论文配图。",
        ),
        "overview": (
            "Method overview: Point cloud forecasting on TartanAir . Gen3R produces structurally disorganized geometry on walls and rooftops, whereas our method preserves coherent structure and yields more accurate predictions.",
            "论文方法总览：VGGT-World 的几何与深度架构。",
        ),
    },
    "vid2world": {
        "teaser": (
            "Official paper teaser: Vid2World repurposes video diffusion models for interactive world modeling . From the perspective of the data pyramid for world models , it leverages vast pre-trained knowledge from internet-scale, action-free video.",
            "论文官方 Teaser：Vid2World 的交互世界模型论文配图。",
        ),
        "overview": (
            "Method overview: Transforming video diffusion models into interactive world models involves two key challenges : (1) Causal generation: converting full-sequence diffusion models into causal diffusion models; (2) Action conditioning:.",
            "论文方法总览：Vid2World 的交互世界模型架构。",
        ),
    },
    "video-depth-anything-prior": {
        "overview": (
            "Method overview: Overall pipeline and the spatio-temporal head . Left: Our model is composed of a backbone encoder from Depth Anything V2 and a newly proposed spatio-temporal head. We jointly train our model on video data using.",
            "论文方法总览：Video Depth Anything 的几何与深度架构。",
        ),
    },
    "vlanext": {
        "teaser": (
            "Official paper teaser: Performance comparison on the LIBERO and LIBERO-plus benchmarks . We compare VLANeXt with representative VLA baselines across model scales. Despite its smaller model size, VLANeXt achieves higher success rates than.",
            "论文官方 Teaser：VLANeXt 的机器人操作论文配图。",
        ),
        "overview": (
            "Method overview: VLANeXt architecture . Multi-view visual inputs, language instructions, and proprioception are tokenized and processed by a multimodal LLM, with meta queries enabling soft interaction with the policy module. Action.",
            "论文方法总览：VLANeXt 的机器人操作架构。",
        ),
    },
    "vmem": {
        "teaser": (
            "Official paper teaser: Method. Given target camera viewpoints { 𝐜 T + m } m = 1 M \\{{c}_{T+m}\\}_{m=1}^{M} , we query our Surfel-Indexed View Memory to retrieve the most relevant K K past views 𝒱 ∗ ⊂ 𝒱 ( s ) {V}^{*}{V}^{(s)} where 𝒱 ∗ = { v.",
            "论文官方 Teaser：VMem 的相机控制论文配图。",
        ),
        "overview": (
            "Method overview: Long sequences with revisitations. We compare our VMem against a baseline without memory that relies solely on the last K K frames for context. Each sequence: input images (left), then generated images at selected.",
            "论文方法总览：VMem 的方法架构。",
        ),
    },
    "wan2.1": {
        "teaser": (
            "Official paper teaser: Wan2.1 text-to-video samples from the paper.",
            "论文官方 Teaser：Wan2.1 论文中的文生视频样例。",
        ),
        "overview": (
            "Method overview: Wan-Encoder and Wan-Decoder around stacked DiT blocks, with umT5 text entering by cross-attention.",
            "论文方法总览：Wan 编码器与解码器夹着一组 DiT 模块，文本经 umT5 交叉注意力进入。",
        ),
    },
    "wan2.1-vace": {
        "teaser": (
            "Official paper teaser: VACE as an all-in-one video creation and editing model on the Wan stack.",
            "论文官方 Teaser：VACE 作为建立在 Wan 栈上的一体化视频创作与编辑模型。",
        ),
    },
    "wan2.2": {
        "overview": (
            "Method overview: Wan2.2 switches a high-noise expert early in denoising and a low-noise expert later.",
            "论文方法总览：Wan2.2 在去噪前期用高噪声专家，后期切换到低噪声专家。",
        ),
    },
    "warp-as-history": {
        "teaser": (
            "Official paper teaser: Warp-as-History samples from the paper.",
            "论文官方 Teaser：Warp-as-History 论文中的样例。",
        ),
        "overview": (
            "Method overview: Warp-as-History’s architecture from the paper.",
            "论文方法总览：Warp-as-History 论文中的方法架构。",
        ),
    },
    "wildworld": {
        "teaser": (
            "Official paper teaser: We present a large-scale dataset curated from game engines for dynamic world modeling. It contains RGB frames with aligned depth maps, camera poses, skeleton, and action / state ground truth. We provide both.",
            "论文官方 Teaser：WildWorld 的几何与深度论文配图。",
        ),
        "overview": (
            "Method overview: The WildWorld dataset curation pipeline.",
            "论文方法总览：WildWorld 的方法架构。",
        ),
    },
    "wonderjourney": {
        "overview": (
            "Method overview: The proposed WonderJourney framework and workflow across modules . Our modular design does not require any training, allowing easy future improvements from the quick advances in vision and language models.",
            "论文方法总览：WonderJourney 的方法架构。",
        ),
    },
    "wonderworld": {
        "overview": (
            "Method overview: The proposed WonderWorld: Our system takes a single image as input and generates connected diverse 3D scenes. Users can specify where (by moving the real-time rendering camera) and what to generate (by typing text.",
            "论文方法总览：WonderWorld 的相机控制架构。",
        ),
    },
    "worldcam": {
        "teaser": (
            "Official paper teaser: Teaser (Best viewed in color and zoomed in): WorldCam is an interactive 3D gaming model that enables precise action control under challenging keyboard and mouse inputs (top), supports long-horizon interactions.",
            "论文官方 Teaser：WorldCam 的交互世界模型论文配图。",
        ),
    },
    "worldfm": {
        "teaser": (
            "Official paper teaser: Examples of generated worlds across diverse styles, including photorealistic, science-fiction, game-like, and artistic environments. The joystick interface enables real-time interactive exploration with negligible.",
            "论文官方 Teaser：InSpatio-WorldFM 的交互世界模型论文配图。",
        ),
        "overview": (
            "Method overview: Overview. In the offline stage, a multi-view-consistent model generates plausible observations that provide 3D anchors and reference appearances. In the online stage, frame model performs fast real-time inference.",
            "论文方法总览：InSpatio-WorldFM 的方法架构。",
        ),
    },
    "worldgen": {
        "overview": (
            "Method overview: WorldGen overview . Our pipeline begins by planning the scene layout, producing a blockout ( B B ), reference image ( 𝐑 {R} ), and navigation mesh (S) (Stage 1). Next, we generate a single 3D mesh that aligns with.",
            "论文方法总览：WorldGen 的方法架构。",
        ),
    },
    "worldgrow": {
        "overview": (
            "Method overview: Overview of WorldGrow. Our goal is to generate infinite 3D scenes through modular, block-by-block synthesis. We begin by curating high-quality scene blocks and adapting SLAT to better model structured 3D context. A.",
            "论文方法总览：WorldGrow 的方法架构。",
        ),
    },
    "worldmem": {
        "teaser": (
            "Official paper teaser: WorldMem enables long-term consistent world generation with an integrated memory mechanism. (a) Previous world generation methods typically face the problem of inconsistent world due to limited temporal context.",
            "论文官方 Teaser：WorldMem 的方法论文配图。",
        ),
    },
    "wow": {
        "teaser": (
            "Official paper teaser: WoW is a world model that integrates perception, prediction, Judgement, reflection, and action. It learns from real-world interaction data and generates high-quality, physically consistent robot videos in seen and.",
            "论文官方 Teaser：WoW 的机器人操作论文配图。",
        ),
        "overview": (
            "Method overview: The architecture of an embodied agent with a world model. An intelligent agent perceives the environment through various sensory inputs (e.g., visual, sound, heat, force). These perceptions are processed by a World.",
            "论文方法总览：WoW 的交互世界模型架构。",
        ),
    },
    "x-wam": {
        "teaser": (
            "Official paper teaser: Overview of X-WAM. Top: X-WAM is a unified 4D World Action Model that jointly predicts future multi-view RGB-D videos and robot actions from video priors, featuring a lightweight depth adaptation module for spatial.",
            "论文官方 Teaser：X-WAM 的几何与深度论文配图。",
        ),
        "overview": (
            "Method overview: Overview of X-WAM. (a) Model architecture: multi-view RGB observations, proprioceptive states, and noisy actions are encoded and jointly denoised by a Diffusion Transformer initialized from Wan2.2-5B, with a.",
            "论文方法总览：X-WAM 的视频生成架构。",
        ),
    },
    "xiaomi-robotics-0": {
        "teaser": (
            "Official paper teaser: Overview. Xiaomi-Robotics-0 achieves state-of-the-art performance in three widely-used simulation benchmarks. It also attains high throughput on two challenging real-robot bimanual manipulation tasks. Furthermore, it.",
            "论文官方 Teaser：Xiaomi-Robotics-0 的机器人操作论文配图。",
        ),
    },
    "xiaomi-robotics-1": {
        "teaser": (
            "Official paper teaser: Overview. Xiaomi-Robotics-1 is pre-trained on over 100k hours of real-world UMI trajectories with auto-labeled state-transition language prompts. It is then aligned to robot embodiments and imperative instruction.",
            "论文官方 Teaser：Xiaomi-Robotics-1 的机器人操作论文配图。",
        ),
        "overview": (
            "Method overview: Model Architecture. Xiaomi-Robotics-1 adopts a Mixture-of-Transformers [ 44 ] architecture that couples a pre-trained VLM with a DiT. The VLM encodes the observation and language instruction, and additionally.",
            "论文方法总览：Xiaomi-Robotics-1 的机器人操作架构。",
        ),
    },
    "xvla": {
        "teaser": (
            "Official paper teaser: X-VLA as a soft-prompted cross-embodiment generalist policy.",
            "论文官方 Teaser：X-VLA 作为带软提示的跨本体通用策略。",
        ),
        "overview": (
            "Method overview: X-VLA’s vision-language-action architecture from the paper.",
            "论文方法总览：X-VLA 论文中的视觉-语言-动作架构。",
        ),
    },
    "yume": {
        "teaser": (
            "Official paper teaser: An example of re-annotating the dataset. The original and new captions are used for T2V and I2V training, respectively. The Original caption describes detail scene context, while the New caption, generated by VLM,.",
            "论文官方 Teaser：YUME 的视频生成论文配图。",
        ),
    },
    "ati-wan21-14b": {
        "teaser": (
            "Official paper teaser: ATI animates an image along user-drawn point trajectories across objects and cameras.",
            "论文官方 Teaser：ATI 按用户点轨迹驱动图像中的物体与相机运动。",
        ),
        "overview": (
            "Method overview: a Trajectory Instruction module injects point paths into Wan DiT latents for controllable image-to-video.",
            "论文方法总览：轨迹指令模块把点路径注入 Wan DiT 潜变量，做可控图生视频。",
        ),
    },
    "egowm": {
        "teaser": (
            "Official paper teaser: EgoWM predicts future frames that follow robot actions for quadruped navigation and humanoid interaction.",
            "论文官方 Teaser：EgoWM 按机器人动作预测未来帧，覆盖四足导航与人形交互。",
        ),
        "overview": (
            "Method overview: an action-projection module conditions a trainable U-Net or DiT on a frozen VAE latent of the first frame.",
            "论文方法总览：动作投影模块在冻结 VAE 的首帧潜变量上条件化可训练的 U-Net 或 DiT。",
        ),
    },
    "emu3.5": {
        "overview": (
            "Method overview: Emu3.5 trains with next-token prediction and infers visual tokens in parallel via discrete diffusion.",
            "论文方法总览：Emu3.5 用下一 token 预测做大规模训练，推理时用离散扩散并行生成视觉 token。",
        ),
    },
    "h-rdt": {
        "overview": (
            "Method overview: H-RDT pretrains on human hand poses, then finetunes modular state and action adapters for multiple robot embodiments.",
            "论文方法总览：H-RDT 先在人手姿态上预训练，再用模块化状态/动作适配器微调到多种机器人本体。",
        ),
    },
    "joyai-echo-wm": {
        "teaser": (
            "Official project teaser: JoyAI-Echo-1.5 samples spanning interactive worlds and real-world human scenes.",
            "官方项目 Teaser：JoyAI-Echo-1.5 的交互世界与真人场景生成样例。",
        ),
    },
    "liveworld": {
        "teaser": (
            "Official paper teaser: LiveWorld keeps out-of-sight entities evolving on virtual monitors, then renders a new camera trajectory.",
            "论文官方 Teaser：LiveWorld 让视野外实体在虚拟监视器上继续演化，再沿新相机轨迹渲染。",
        ),
        "overview": (
            "Method overview: dynamic entity evolution and static SLAM accumulation feed a video diffusion transformer with LoRA.",
            "论文方法总览：动态实体演化与静态 SLAM 累积，再送入带 LoRA 的视频扩散 Transformer。",
        ),
    },
    "ltx-2.x": {
        "overview": (
            "Method overview: LTX-2 encodes audio and video with causal VAEs and couples dual streams through audio-visual cross-attention.",
            "论文方法总览：LTX-2 用因果 VAE 编码音视频，再以音视交叉注意力耦合双流。",
        ),
    },
    "lyra": {
        "teaser": (
            "Official paper teaser: Lyra 2.0 turns one image into a long-horizon 3D-consistent world via video generation and 3D Gaussian Splatting.",
            "论文官方 Teaser：Lyra 2.0 从单图经视频生成与 3D 高斯泼溅得到长程三维一致世界。",
        ),
        "overview": (
            "Method overview: spatial memory and dense 3D correspondence condition a DiT that generates camera-guided video, then reconstructs 3DGS.",
            "论文方法总览：空间记忆与稠密三维对应条件化 DiT 生成相机引导视频，再重建 3DGS。",
        ),
    },
    "magicworld": {
        "overview": (
            "Method overview: an action-guided geometry prior and cached Video DiT, with flow-based motion preservation and teacher-student training.",
            "论文方法总览：动作引导的几何先验与带缓存的 Video DiT，并用光流保运动与师生训练。",
        ),
    },
    "matrix-game-3.5-first-person": {
        "teaser": (
            "Official project figure: a motion-aware object filter lifts previous frames and references into 3D, then emits mosaic and masked context tokens.",
            "官方项目配图：运动感知物体过滤器把历史帧与参考图抬到三维，再输出马赛克与掩码上下文 token。",
        ),
        "overview": (
            "Method overview: object, context, and anchor tokens along a camera trajectory, with noisy targets and patch memory sharing RoPE time.",
            "论文方法总览：沿相机轨迹排列物体、上下文与锚点 token，噪声目标与 patch 记忆共享 RoPE 时间。",
        ),
    },
    "metric3d-prior": {
        "teaser": (
            "Official paper teaser: Metric3D v2 predicts metric depth and surface normals on diverse web images versus Marigold.",
            "论文官方 Teaser：Metric3D v2 在多样网络图像上预测度量深度与表面法向，并与 Marigold 对照。",
        ),
        "overview": (
            "Method overview: one network predicts a unified metric depth distribution and surface normals, trained once for many applications.",
            "论文方法总览：单个网络预测统一度量深度分布与表面法向，一次训练覆盖多种应用。",
        ),
    },
    "nwm": {
        "teaser": (
            "Official paper teaser: Navigation World Model follows a commanded trajectory through unknown environments from a single image.",
            "论文官方 Teaser：Navigation World Model 从单图沿给定轨迹在未知环境中导航。",
        ),
        "overview": (
            "Method overview: a Conditional Diffusion Transformer block with AdaLN action conditioning and cross-attention over context states.",
            "论文方法总览：条件扩散 Transformer 块用 AdaLN 注入动作，并对上下文状态做交叉注意力。",
        ),
    },
    "pandora": {
        "overview": (
            "Method overview: a pretrained LLM backbone consumes vision-encoded states and language actions, then a video generator rolls out the next world state.",
            "论文方法总览：预训练 LLM 骨干接收视觉编码状态与语言动作，再由视频生成器滚动出下一世界状态。",
        ),
    },
    "prior-depth-anything": {
        "teaser": (
            "Official paper teaser: Prior Depth Anything fuses dense geometric estimates with sparse metric measurements.",
            "论文官方 Teaser：Prior Depth Anything 把稠密几何估计与稀疏度量测量融合。",
        ),
        "overview": (
            "Method overview: coarse metric alignment of any depth prior, then implicit RGB-conditioned refinement through a frozen monocular depth estimator.",
            "论文方法总览：先对任意深度先验做粗度量对齐，再用冻结单目深度估计器做 RGB 条件精细化。",
        ),
    },
    "spatial-forcing": {
        "teaser": (
            "Official paper teaser: Spatial Forcing aligns VLA visual embeddings with a frozen 3D foundation model.",
            "论文官方 Teaser：Spatial Forcing 把 VLA 视觉嵌入与冻结的三维基础模型对齐。",
        ),
        "overview": (
            "Method overview: Spatial Forcing implicitly forces spatial features, instead of feeding explicit depth or an external 3D expert.",
            "论文方法总览：Spatial Forcing 隐式强迫空间特征，而不是输入显式深度或外挂三维专家。",
        ),
    },
    "unianimate-dit": {
        "teaser": (
            "Official paper teaser: UniAnimate-DiT animates stylized and photorealistic characters from driving poses.",
            "论文官方 Teaser：UniAnimate-DiT 根据驱动姿态动画化风格化与写实角色。",
        ),
        "overview": (
            "Method overview: Wan-DiT with tunable pose encoders and QKV-LoRA generates pose-driven video from a reference image.",
            "论文方法总览：Wan-DiT 加上可训练姿态编码器与 QKV-LoRA，从参考图生成姿态驱动视频。",
        ),
    },
    "vggt": {
        "teaser": (
            "Official paper teaser: VGGT reconstructs 3D from one to many views faster and cleaner than DUSt3R.",
            "论文官方 Teaser：VGGT 从一到多视图重建三维，比 DUSt3R 更快更干净。",
        ),
        "overview": (
            "Method overview: DINO tokens plus camera tokens go through alternating global and frame attention, then camera, depth, point-map, and track heads.",
            "论文方法总览：DINO token 与相机 token 交替做全局与帧内注意力，再接到相机、深度、点图与轨迹头。",
        ),
    },
    "wilddet3d": {
        "teaser": (
            "Official paper teaser: WildDet3D does open-vocabulary 3D detection from any image, device, and prompt.",
            "论文官方 Teaser：WildDet3D 从任意图像、设备与提示做开放词汇三维检测。",
        ),
        "overview": (
            "Method overview: an image encoder and RGBD encoder fuse depth latents into a promptable detector with 2D and 3D heads.",
            "论文方法总览：图像编码器与 RGBD 编码器融合深度潜变量，再进入带 2D/3D 头的可提示检测器。",
        ),
    },
}


def first_existing(model_id: str, names: tuple[str, ...]) -> Path | None:
    folder = PUBLIC_MODELS / model_id
    for name in names:
        path = folder / name
        if path.is_file() and path.stat().st_size > 800:
            return path
    return None


def public_src(path: Path) -> str:
    relative = path.relative_to(DOCS_ROOT / "public").as_posix()
    return f"/{relative}"


def resolve_asset_model(model_id: str, names: tuple[str, ...]) -> str:
    if first_existing(model_id, names):
        return model_id
    shared = SHARE_FROM.get(model_id)
    if shared and first_existing(shared, names):
        return shared
    return model_id


def teaser_path(model_id: str) -> Path | None:
    return first_existing(resolve_asset_model(model_id, TEASER_NAMES), TEASER_NAMES)


def overview_path(model_id: str) -> Path | None:
    return first_existing(resolve_asset_model(model_id, OVERVIEW_NAMES), OVERVIEW_NAMES)


def teaser_src(model_id: str) -> str | None:
    path = teaser_path(model_id)
    return public_src(path) if path else None


def overview_src(model_id: str) -> str | None:
    path = overview_path(model_id)
    return public_src(path) if path else None


def caption_pair(model_id: str, role: str) -> tuple[str, str]:
    owner = model_id
    if model_id not in CAPTIONS and SHARE_FROM.get(model_id) in CAPTIONS:
        owner = SHARE_FROM[model_id]
    localized = CAPTIONS.get(owner, {}).get(role)
    if localized:
        return localized
    return DEFAULT_TEASER if role == "teaser" else DEFAULT_OVERVIEW


def has_paper_figures(model_id: str) -> bool:
    return bool(teaser_src(model_id) or overview_src(model_id))


def figure_records(model_id: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    teaser = teaser_src(model_id)
    if teaser:
        en, zh = caption_pair(model_id, "teaser")
        records.append(
            {
                "kind": "image",
                "role": "teaser",
                "src": teaser,
                "caption": en,
                "captionZh": zh,
            }
        )
    overview = overview_src(model_id)
    if overview:
        en, zh = caption_pair(model_id, "overview")
        records.append(
            {
                "kind": "image",
                "role": "overview",
                "src": overview,
                "caption": en,
                "captionZh": zh,
            }
        )
    return records
