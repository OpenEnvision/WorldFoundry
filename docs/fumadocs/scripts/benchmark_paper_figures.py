"""Shared paths and captions for official benchmark paper figures.

Body figures are never a PDF page-1 cover. ``CAPTIONS`` are one-sentence
explanations shown under each image, not short labels.
"""

from __future__ import annotations

from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BENCHMARKS = DOCS_ROOT / "public" / "benchmarks"

TEASER_NAMES = ("teaser.png", "teaser.jpg", "teaser.webp")
OVERVIEW_NAMES = ("overview.png", "overview.jpg", "overview.webp")
RESULTS_NAMES = ("results.png", "results.jpg", "results.webp")

DEFAULT_TEASER = ("Official paper teaser.", "论文官方 Teaser。")
DEFAULT_OVERVIEW = ("Official paper figure.", "论文主图。")
DEFAULT_RESULTS = ("Official paper results figure.", "论文官方结果图。")

# Benches that reuse a sibling's already-curated assets (same official paper).
SHARE_FROM: dict[str, str] = {}

CAPTIONS: dict[str, dict[str, tuple[str, str]]] = {
    "ai2thor": {
        "teaser": (
            "Official paper teaser: interactive indoor scenes in the AI2-THOR Unity simulator.",
            "论文官方 Teaser：AI2-THOR Unity 模拟器中的可交互室内场景。",
        ),
    },
    "behavior1k": {
        "teaser": (
            "Official paper teaser: 1,000 everyday household activities in OmniGibson.",
            "论文官方 Teaser：OmniGibson 中的 1,000 项日常家务活动。",
        ),
    },
    "calvin": {
        "teaser": (
            "Official paper teaser: four tabletop environments for long-horizon language-conditioned control.",
            "论文官方 Teaser：四个桌面环境，用于长时程语言条件控制。",
        ),
    },
    "larybench": {
        "teaser": (
            "Official paper teaser: vision-to-action evaluation on action generalization and robot control.",
            "论文官方 Teaser：从视觉到动作，同时评测动作泛化与机器人控制。",
        ),
        "overview": (
            "Official paper figure: atomic and composite actions across domains, then classification and control.",
            "论文主图：跨域的原子 / 组合动作，再做分类与控制评测。",
        ),
    },
    "libero": {
        "teaser": (
            "Official paper teaser: four procedurally generated suites — Spatial, Object, Goal, and Long — that isolate what knowledge transfers.",
            "论文官方 Teaser：四个程序生成套件（Spatial、Object、Goal、Long），分别隔离「迁了什么」。",
        ),
        "overview": (
            "Official paper figure: LIBERO’s pipeline from Ego4D behavioral templates to task instructions and scenes.",
            "论文主图：LIBERO 从 Ego4D 行为模板到任务指令与场景的生成流程。",
        ),
    },
    "libero-mem": {
        "teaser": (
            "Official paper teaser: object-level POMDP tasks that require remembering prior actions, not just the current frame.",
            "论文官方 Teaser：物体级 POMDP 任务——必须记住先前动作，而不能只看当前帧。",
        ),
    },
    "libero-plus": {
        "overview": (
            "Official paper figure: 10,030 tasks across seven perturbation factors and twenty-one components.",
            "论文主图：10,030 个任务，覆盖七个扰动因子与二十一个子维度。",
        ),
    },
    "libero-pro": {
        "overview": (
            "Official paper figure: five perturbation types — object, configuration, instruction, task, and environment.",
            "论文主图：五类扰动——物体、构型、指令、任务与环境。",
        ),
    },
    "likephys": {
        "teaser": (
            "Official paper teaser: a model with learned physics should assign higher likelihood to valid videos than to violations.",
            "论文官方 Teaser：学到物理的模型应对合法视频给出更高似然，而不是违规视频。",
        ),
        "overview": (
            "Official paper figure: valid vs. invalid simulated videos scored by the video diffusion model’s likelihood.",
            "论文主图：用视频扩散模型的似然，对比合法与故意违规的仿真视频。",
        ),
    },
    "maniskill": {
        "teaser": (
            "Official paper teaser: OpenCabinet, PushChair, and MoveBucket across many articulated objects.",
            "论文官方 Teaser：OpenCabinet、PushChair、MoveBucket 等任务，覆盖大量铰接物体。",
        ),
    },
    "maniskill2": {
        "teaser": (
            "Official paper teaser: a unified suite of stationary/mobile, single/dual-arm, rigid/soft-body tasks.",
            "论文官方 Teaser：统一套件，覆盖固定/移动基座、单/双臂、刚/软体任务。",
        ),
    },
    "metaworld": {
        "teaser": (
            "Official paper teaser: 50 tabletop manipulation tasks for multi-task and meta-RL.",
            "论文官方 Teaser：50 个桌面操作任务，用于多任务与元强化学习。",
        ),
        "results": (
            "Official paper Figure 3: multi-task and meta-RL success curves on MT10 / ML10.",
            "论文 Figure 3：MT10 / ML10 上的多任务与元强化学习成功率曲线。",
        ),
    },
    "phygenbench": {
        "teaser": (
            "Official paper teaser: T2V models failing 44 physical-commonsense aspects in PhyGenBench.",
            "论文官方 Teaser：T2V 模型在 PhyGenBench 的 44 个物理常识侧面中的失败样例。",
        ),
        "overview": (
            "Official paper figure: four physics categories and the PhyGenBench data pipeline.",
            "论文主图：四类物理范畴，以及 PhyGenBench 的数据管线。",
        ),
    },
    "physics-iq": {
        "teaser": (
            "Official paper teaser: filmed scenarios that test whether a video model continues real physics.",
            "论文官方 Teaser：实拍场景，检验视频模型能否续写真实物理。",
        ),
        "overview": (
            "Official paper figure: generate a 5-second continuation and score it against the real future.",
            "论文主图：生成 5 秒续写，再与真实未来对比打分。",
        ),
    },
    "physics-iq-verified": {
        "overview": (
            "Official paper figure: verified prompts, verified masks, and per-view aggregation versus the original pipeline.",
            "论文主图：相对原协议的三处修订——verified prompt、verified mask，以及按视角聚合。",
        ),
    },
    "rlbench": {
        "teaser": (
            "Official paper teaser: 100 hand-designed manipulation tasks in a single environment.",
            "论文官方 Teaser：同一环境中的 100 个手写操作任务。",
        ),
    },
    "robocasa": {
        "teaser": (
            "Official paper teaser: robots in diverse simulated kitchens.",
            "论文官方 Teaser：多样仿真厨房中的机器人。",
        ),
        "overview": (
            "Official paper figure: LLM-generated kitchen tasks from high-level activities down to executable scenes.",
            "论文主图：用大语言模型从高层厨房活动生成可执行任务。",
        ),
    },
    "robotwin": {
        "teaser": (
            "Official paper teaser: bimanual manipulation tasks with domain randomization.",
            "论文官方 Teaser：带域随机化的双臂操作任务。",
        ),
        "overview": (
            "Official paper figure: RoboTwin’s dual-arm benchmark and data-generation stack.",
            "论文主图：RoboTwin 的双臂评测与数据生成栈。",
        ),
    },
    "simpler-env": {
        "teaser": (
            "Official paper teaser: SIMPLER’s Google Robot and WidowX / Bridge real-to-sim setups.",
            "论文官方 Teaser：SIMPLER 对应 Google Robot 与 WidowX / Bridge 的 real-to-sim 设置。",
        ),
        "overview": (
            "Official paper figure: open-source simulated eval environments matched to common real robot setups.",
            "论文主图：与常见真机设置对齐的开源仿真评测环境。",
        ),
    },
    "t2v-compbench": {
        "teaser": (
            "Official paper teaser: seven compositional categories for text-to-video generation.",
            "论文官方 Teaser：文生视频的七个组合性类别。",
        ),
        "overview": (
            "Official paper figure: how prompts are generated for the seven compositional categories.",
            "论文主图：七个组合性类别的 prompt 如何生成。",
        ),
    },
    "vbench": {
        "teaser": (
            "Official paper teaser: a hierarchical 16-dimension suite for video generative models.",
            "论文官方 Teaser：面向视频生成模型的 16 维分层评测套件。",
        ),
        "results": (
            "Official paper radar: four models across all 16 VBench dimensions.",
            "论文雷达图：四个模型在 VBench 全部 16 个维度上的结果。",
        ),
    },
    "4dworldbench": {
        "teaser": (
            "Official paper figure: 4DWorldBench’s unified evaluation framework for 3D/4D world generation.",
            "论文主图：4DWorldBench 面向 3D/4D 世界生成的统一评测框架。",
        ),
    },
    "aigcbench": {
        "teaser": (
            "Official paper teaser: AIGCBench’s three modules — dataset, metrics, and image-to-video models.",
            "论文官方 Teaser：AIGCBench 的三个模块——评测数据、指标与待测图生视频模型。",
        ),
        "overview": (
            "Official paper figure: the image–text dataset generation pipeline used to build AIGCBench.",
            "论文主图：构建 AIGCBench 所用的图文数据生成管线。",
        ),
    },
    "apple-pi": {
        "teaser": (
            "Official paper teaser: thinking-with-video tasks that test law-grounded physical intelligence.",
            "论文官方 Teaser：用视频思考的任务，检验有定律依据的物理智能。",
        ),
    },
    "bridgedata-v2": {
        "teaser": (
            "Official paper teaser: BridgeData V2’s large-scale real-robot manipulation dataset.",
            "论文官方 Teaser：BridgeData V2 大规模真机操作数据集。",
        ),
        "overview": (
            "Official paper figure: skill and environment coverage across the collected robot tasks.",
            "论文主图：采集任务覆盖的技能与环境。",
        ),
    },
    "camerabench": {
        "teaser": (
            "Official paper teaser: camera-motion understanding across diverse video genres.",
            "论文官方 Teaser：在多样视频类型中理解相机运动。",
        ),
        "overview": (
            "Official paper figure: CameraBench’s annotation and evaluation pipeline.",
            "论文主图：CameraBench 的标注与评测管线。",
        ),
    },
    "chronomagic-bench": {
        "teaser": (
            "Official paper teaser: four metamorphic time-lapse categories — biological, human, urban, and natural.",
            "论文官方 Teaser：四类形变延时——生物、人文、城市与自然。",
        ),
    },
    "devil-dynamics": {
        "overview": (
            "Official paper figure: how DEVIL turns dynamics scores and prompts into evaluation metrics.",
            "论文主图：DEVIL 如何把动态分数与 prompt 变成评测指标。",
        ),
    },
    "evalcrafter": {
        "teaser": (
            "Official paper teaser: EvalCrafter’s large-scale video-generation evaluation suite.",
            "论文官方 Teaser：EvalCrafter 的大规模视频生成评测套件。",
        ),
        "overview": (
            "Official paper figure: the prompt-collection and scoring pipeline.",
            "论文主图：prompt 采集与打分管线。",
        ),
    },
    "ewmbench": {
        "teaser": (
            "Official paper teaser: why embodied world-model video differs from general video generation.",
            "论文官方 Teaser：具身世界模型视频与通用视频生成的差异。",
        ),
        "overview": (
            "Official paper figure: EWMBench’s scene, motion, and semantic evaluation pipeline.",
            "论文主图：EWMBench 的场景、运动与语义评测管线。",
        ),
    },
    "fetv": {
        "teaser": (
            "Official paper figure: FETV’s three-axis prompt taxonomy — content, attributes, and complexity.",
            "论文主图：FETV 的三轴 prompt 分类——内容、属性与复杂度。",
        ),
        "results": (
            "Official paper leaderboard figure: human ratings across FETV’s evaluation axes.",
            "论文官方榜图：FETV 各评测轴上的人工评分。",
        ),
    },
    "genai-bench": {
        "teaser": (
            "Official paper teaser: GenAI Arena’s community voting, automated bench, and leaderboard.",
            "论文官方 Teaser：GenAI Arena 的社区投票、自动评测与排行榜。",
        ),
    },
    "ipv-bench": {
        "teaser": (
            "Official paper teaser: impossible-video examples that break physical or commonsense rules.",
            "论文官方 Teaser：违背物理或常识的不可能视频样例。",
        ),
        "overview": (
            "Official paper figure: IPV-Bench’s taxonomy and evaluation setup.",
            "论文主图：IPV-Bench 的分类体系与评测设置。",
        ),
    },
    "iworld-bench": {
        "teaser": (
            "Official paper teaser: six interactive tasks across UGV, UAV, human, and robot viewpoints.",
            "论文官方 Teaser：覆盖地面、空中、人与机器人视角的六类交互任务。",
        ),
        "overview": (
            "Official paper figure: the data pipeline that builds iWorld-Bench’s 330k clips.",
            "论文主图：构建 iWorld-Bench 33 万片段的数据管线。",
        ),
    },
    "kinetix": {
        "teaser": (
            "Official paper teaser: a general agent trained on random physics tasks, then tested on hand-designed levels.",
            "论文官方 Teaser：在随机物理任务上训练通用智能体，再迁移到手写关卡。",
        ),
        "results": (
            "Official paper learning curves: return vs environment steps on held-out physics levels.",
            "论文学习曲线：在留出物理关卡上的回报随环境步数变化。",
        ),
    },
    "memobench": {
        "teaser": (
            "Official paper teaser: Visible–Disappear–Reappear clips that test memory of changing objects.",
            "论文官方 Teaser：可见—消失—再现片段，检验对变化物体的记忆。",
        ),
        "overview": (
            "Official paper figure: how synthetic and real MemoBench clips are curated.",
            "论文主图：MemoBench 合成与实拍片段如何筛选。",
        ),
    },
    "mikasa": {
        "teaser": (
            "Official paper teaser: a taxonomy of memory problems for robot RL.",
            "论文官方 Teaser：面向机器人强化学习的记忆问题分类。",
        ),
        "results": (
            "Official paper spider chart: PPO and related agents across MiKASA memory axes.",
            "论文蛛网图：PPO 及相关智能体在 MiKASA 记忆轴上的表现。",
        ),
        "overview": (
            "Official paper figure: demonstrative memory-intensive tasks in MiKASA-Robo.",
            "论文主图：MiKASA-Robo 中需要记忆的示范任务。",
        ),
    },
    "mind": {
        "teaser": (
            "Official paper teaser: memory consistency and action control in world models.",
            "论文官方 Teaser：世界模型中的记忆一致性与动作控制。",
        ),
        "overview": (
            "Official paper figure: MIND’s scene layout and evaluation protocol.",
            "论文主图：MIND 的场景布置与评测协议。",
        ),
    },
    "molmospaces": {
        "teaser": (
            "Official paper teaser: MolmoBot zero-shot manipulation across simulated MolmoSpaces.",
            "论文官方 Teaser：MolmoBot 在仿真 MolmoSpaces 中的零样本操作。",
        ),
    },
    "pawbench": {
        "teaser": (
            "Official paper teaser: one plausible future is not enough — world models must match a distribution.",
            "论文官方 Teaser：一条看似合理的未来不够——世界模型必须对齐分布。",
        ),
        "overview": (
            "Official paper figure: PAWEval turns repeated rollouts into a distributional test.",
            "论文主图：PAWEval 把多次 rollout 变成分布检验。",
        ),
    },
    "phyeduvideo": {
        "teaser": (
            "Official paper teaser: text-to-video failures on physics-education demonstrations.",
            "论文官方 Teaser：文生视频在物理教学演示上的失败样例。",
        ),
    },
    "phyfps-bench-gen": {
        "teaser": (
            "Official paper teaser: chronometric hallucination — generated motion that does not match Meta FPS.",
            "论文官方 Teaser：计时幻觉——生成运动与 Meta FPS 对不上。",
        ),
    },
    "phyground": {
        "teaser": (
            "Official paper teaser: physical-law violations in generative world models.",
            "论文官方 Teaser：生成式世界模型中的物理定律违规。",
        ),
    },
    "physical-ai-bench": {
        "teaser": (
            "Official paper teaser: PAI-Bench’s generation and understanding tracks for Physical AI.",
            "论文官方 Teaser：PAI-Bench 面向 Physical AI 的生成与理解赛道。",
        ),
    },
    "physvidbench": {
        "teaser": (
            "Official paper teaser: everyday physical-commonsense tasks for video generators.",
            "论文官方 Teaser：面向视频生成模型的日常物理常识任务。",
        ),
        "overview": (
            "Official paper figure: PhysVidBench’s concept coverage and evaluation protocol.",
            "论文主图：PhysVidBench 的概念覆盖与评测协议。",
        ),
    },
    "rbench": {
        "teaser": (
            "Official paper teaser: temporal grids that test video models as embodied world models.",
            "论文官方 Teaser：用时间网格检验视频模型能否当具身世界模型。",
        ),
        "overview": (
            "Official paper figure: RBench’s data-construction pipeline.",
            "论文主图：RBench 的数据构建管线。",
        ),
    },
    "robocerebra": {
        "teaser": (
            "Official paper teaser: long-horizon household tasks that need memory and plan updates.",
            "论文官方 Teaser：需要记忆与计划更新的长时程家务任务。",
        ),
        "overview": (
            "Official paper figure: RoboCerebra’s task-generation pipeline.",
            "论文主图：RoboCerebra 的任务生成管线。",
        ),
    },
    "robomme": {
        "teaser": (
            "Official paper teaser: memory-augmented manipulation suites for generalist robot policies.",
            "论文官方 Teaser：面向通用机器人策略的记忆增强操作套件。",
        ),
    },
    "sana-wm-bench": {
        "teaser": (
            "Official paper figure: the 80 initial scenes used in the one-minute SANA-WM benchmark.",
            "论文主图：SANA-WM 一分钟评测所用的 80 个初始场景。",
        ),
    },
    "stevo-bench": {
        "teaser": (
            "Official paper teaser: out-of-sight state evolution that video world models must track.",
            "论文官方 Teaser：视频世界模型必须跟踪的视野外状态演化。",
        ),
        "overview": (
            "Official paper figure: STEVO-Bench’s hidden-state evaluation protocol.",
            "论文主图：STEVO-Bench 的隐状态评测协议。",
        ),
    },
    "t2v-safety-bench": {
        "teaser": (
            "Official paper teaser: twelve safety aspects for text-to-video generation.",
            "论文官方 Teaser：文生视频的十二个安全侧面。",
        ),
    },
    "t2vworldbench": {
        "teaser": (
            "Official paper teaser: six world-knowledge domains with generated-video examples.",
            "论文官方 Teaser：六个世界知识域及其生成视频样例。",
        ),
    },
    "vbench-2.0": {
        "teaser": (
            "Official paper teaser: VBench-2.0’s intrinsic-faithfulness dimension suite.",
            "论文官方 Teaser：VBench-2.0 的内在忠实度维度套件。",
        ),
        "overview": (
            "Official paper figure: the anomaly-detection framework behind VBench-2.0 metrics.",
            "论文主图：VBench-2.0 指标背后的异常检测框架。",
        ),
    },
    "vbench-plus-plus": {
        "teaser": (
            "Official paper teaser: VBench++ extensions for image-to-video and more versatile eval.",
            "论文官方 Teaser：VBench++ 对图生视频等更广评测的扩展。",
        ),
    },
    "video-bench": {
        "overview": (
            "Official paper figure: Video-Bench’s chain-of-query MLLM scoring framework.",
            "论文主图：Video-Bench 的链式提问 MLLM 打分框架。",
        ),
    },
    "videophy": {
        "teaser": (
            "Official paper teaser: T2V models violating conservation of mass, Newton’s first law, and rigidity.",
            "论文官方 Teaser：文生视频模型违背质量守恒、牛顿第一定律与刚体性。",
        ),
    },
    "videophy2": {
        "teaser": (
            "Official paper teaser: action-centric physical-commonsense failures in VideoPhy-2.",
            "论文官方 Teaser：VideoPhy-2 中以动作为中心的物理常识失败。",
        ),
    },
    "videoscore": {
        "teaser": (
            "Official paper teaser: VideoFeedback construction and the VideoScore auto-metric.",
            "论文官方 Teaser：VideoFeedback 的构建，以及 VideoScore 自动指标。",
        ),
    },
    "videoverse": {
        "teaser": (
            "Official paper teaser: VideoVerse’s world-model evaluation dimensions for T2V.",
            "论文官方 Teaser：VideoVerse 面向文生视频的世界模型评测维度。",
        ),
    },
    "visual-chronometer": {
        "teaser": (
            "Official paper teaser: recovering PhyFPS from visual motion, not container metadata.",
            "论文官方 Teaser：从视觉运动恢复 PhyFPS，而不是信容器元数据。",
        ),
    },
    "vlabench": {
        "teaser": (
            "Official paper teaser: 100 language-conditioned manipulation categories with 2,000+ objects.",
            "论文官方 Teaser：100 类语言条件操作任务，覆盖 2,000+ 物体。",
        ),
        "overview": (
            "Official paper figure: example long-horizon VLABench tasks that need world knowledge.",
            "论文主图：需要世界知识的长时程 VLABench 任务样例。",
        ),
    },
    "vmbench": {
        "teaser": (
            "Official paper teaser: six motion-pattern categories in VMBench.",
            "论文官方 Teaser：VMBench 的六类运动模式。",
        ),
        "overview": (
            "Official paper figure: VMBench’s perception-aligned motion evaluation pipeline.",
            "论文主图：VMBench 的感知对齐运动评测管线。",
        ),
    },
    "wbench": {
        "teaser": (
            "Official paper teaser: WBench’s multi-turn world-generation evaluation axes.",
            "论文官方 Teaser：WBench 的多轮世界生成评测轴。",
        ),
    },
    "world-in-world": {
        "teaser": (
            "Official paper teaser: closed-loop world-model evaluation in World-in-World.",
            "论文官方 Teaser：World-in-World 的闭环世界模型评测。",
        ),
        "overview": (
            "Official paper figure: the evaluation framework across the four World-in-World tasks.",
            "论文主图：World-in-World 四类任务的评测框架。",
        ),
    },
    "worldarena": {
        "teaser": (
            "Official paper teaser: sixteen video-perception metrics across six sub-dimensions.",
            "论文官方 Teaser：六个子维度上的十六项视频感知指标。",
        ),
        "overview": (
            "Official paper figure: embodied-task evaluation — data engine, policy evaluator, and planner.",
            "论文主图：具身任务评测——数据引擎、策略评估器与规划器。",
        ),
    },
    "worldbench": {
        "teaser": (
            "Official paper teaser: isolate physics concepts, then generate and score world-model video.",
            "论文官方 Teaser：先隔离物理概念，再生成并给世界模型视频打分。",
        ),
        "overview": (
            "Official paper figure: physical-parameter estimation used to score WorldBench videos.",
            "论文主图：给 WorldBench 视频打分所用的物理参数估计。",
        ),
    },
    "worldmodelbench": {
        "teaser": (
            "Official paper teaser: judging video generators as world models on physics and instruction following.",
            "论文官方 Teaser：把视频生成器当世界模型，评物理与指令跟随。",
        ),
    },
    "worldolympiad": {
        "teaser": (
            "Official paper teaser: WorldOlympiad’s triathlon-style world-model evaluation.",
            "论文官方 Teaser：WorldOlympiad 的铁人三项式世界模型评测。",
        ),
        "overview": (
            "Official paper figure: how WorldOlympiad data is collected and chunked.",
            "论文主图：WorldOlympiad 数据如何采集与切分。",
        ),
    },
    "worldreasonbench": {
        "teaser": (
            "Official paper teaser: video generators scored as future world-state predictors.",
            "论文官方 Teaser：把视频生成器当未来世界状态预测器来打分。",
        ),
        "overview": (
            "Official paper figure: WorldReasonBench’s construction pipeline.",
            "论文主图：WorldReasonBench 的构建管线。",
        ),
    },
    "worldscore": {
        "teaser": (
            "Official paper teaser: WorldScore versus prior video benches on controllability and 3D consistency.",
            "论文官方 Teaser：WorldScore 相对已有视频评测在可控性与 3D 一致性上的覆盖。",
        ),
        "overview": (
            "Official paper figure: WorldScore’s world-specification and metric framework.",
            "论文主图：WorldScore 的世界规格与指标框架。",
        ),
    },
    "wrbench": {
        "teaser": (
            "Official paper teaser: a shared scene, event, and viewpoint test for persistent world state.",
            "论文官方 Teaser：用同一场景、事件与视角干预，检验持久世界状态。",
        ),
        "overview": (
            "Official paper figure: WRBench’s event–view records and evaluation pipeline.",
            "论文主图：WRBench 的事件–视角记录与评测管线。",
        ),
    },
}


def first_existing(benchmark_id: str, names: tuple[str, ...]) -> Path | None:
    folder = PUBLIC_BENCHMARKS / benchmark_id
    for name in names:
        path = folder / name
        if path.is_file() and path.stat().st_size > 800:
            return path
    return None


def public_src(path: Path) -> str:
    relative = path.relative_to(DOCS_ROOT / "public").as_posix()
    return f"/{relative}"


def resolve_asset_id(benchmark_id: str, names: tuple[str, ...]) -> str:
    if first_existing(benchmark_id, names):
        return benchmark_id
    shared = SHARE_FROM.get(benchmark_id)
    if shared and first_existing(shared, names):
        return shared
    return benchmark_id


def teaser_path(benchmark_id: str) -> Path | None:
    return first_existing(resolve_asset_id(benchmark_id, TEASER_NAMES), TEASER_NAMES)


def overview_path(benchmark_id: str) -> Path | None:
    return first_existing(resolve_asset_id(benchmark_id, OVERVIEW_NAMES), OVERVIEW_NAMES)


def teaser_src(benchmark_id: str) -> str | None:
    path = teaser_path(benchmark_id)
    return public_src(path) if path else None


def overview_src(benchmark_id: str) -> str | None:
    path = overview_path(benchmark_id)
    return public_src(path) if path else None


def results_path(benchmark_id: str) -> Path | None:
    return first_existing(resolve_asset_id(benchmark_id, RESULTS_NAMES), RESULTS_NAMES)


def results_src(benchmark_id: str) -> str | None:
    path = results_path(benchmark_id)
    return public_src(path) if path else None


def caption_pair(benchmark_id: str, role: str) -> tuple[str, str]:
    owner = benchmark_id
    if benchmark_id not in CAPTIONS and SHARE_FROM.get(benchmark_id) in CAPTIONS:
        owner = SHARE_FROM[benchmark_id]
    localized = CAPTIONS.get(owner, {}).get(role)
    if localized:
        return localized
    if role == "teaser":
        return DEFAULT_TEASER
    if role == "results":
        return DEFAULT_RESULTS
    return DEFAULT_OVERVIEW


def has_paper_figures(benchmark_id: str) -> bool:
    return bool(teaser_src(benchmark_id) or overview_src(benchmark_id) or results_src(benchmark_id))
