"""Bilingual intro blurbs for the Python API reference.

Curated entries override the generator's docstring-based intros. Keys are
``public_module.name`` as emitted in ``python-api.json``.
"""

from __future__ import annotations

# Evaluation / runtime public surface — written for readers, not copied from source.
CURATED_INTROS: dict[str, dict[str, str]] = {
    "worldfoundry.evaluation.api.ArtifactRef": {
        "en": "A portable pointer to an artifact (path or URI) with optional size, hash, and media metadata. Use it to pass outputs across process boundaries without embedding file bytes.",
        "zh": "指向产物的可移植引用（本地路径或 URI），可附带大小、哈希与媒体元数据。用于跨进程传递输出，而不嵌入文件字节。",
    },
    "worldfoundry.evaluation.api.local_path_for_uri": {
        "en": "Resolve a URI to a local filesystem path when possible. Returns None for remote or non-file schemes so callers do not treat HTTP/object-store URIs as local files.",
        "zh": "在可能时把 URI 解析为本地路径；远程或非文件协议返回 None，避免把 HTTP/对象存储地址误当成本地文件。",
    },
    "worldfoundry.evaluation.api.enrich_artifact_ref": {
        "en": "Fill size and SHA-256 on an ArtifactRef when the URI resolves to an existing local file. Missing or remote refs are left unchanged.",
        "zh": "当 URI 对应本地已存在文件时，为 ArtifactRef 补齐大小与 SHA-256；缺失或远程引用保持不变。",
    },
    "worldfoundry.evaluation.api.GenerationRequest": {
        "en": "One evaluation sample’s generation request: inputs, controls, sampling kwargs, and expected outputs. Prefer one request per sample rather than packing a whole batch.",
        "zh": "单条样本的生成请求：输入、控制量、采样参数与期望输出。应按样本构造，而不是把整批塞进一个 request。",
    },
    "worldfoundry.evaluation.api.GenerationResult": {
        "en": "Normalized generation outcome for one sample: artifacts on success, or status/error on failure. Benchmarks and metrics consume this shape, not the model runtime.",
        "zh": "单条样本的规范化生成结果：成功时放 artifacts，失败时保留 status/error。Benchmark 与 metric 消费该结构，而不是模型 runtime。",
    },
    "worldfoundry.evaluation.api.normalize_generation_status": {
        "en": "Map nearby status strings (success/completed/done, …) onto a canonical status before success checks. Does not verify that artifact files exist.",
        "zh": "把相近的状态词（success/completed/done 等）归一成规范 status，供成功判定使用；不检查产物文件是否存在。",
    },
    "worldfoundry.evaluation.api.is_generation_result_successful": {
        "en": "True when the normalized status is an accepted success value and error is empty. Artifact presence and leaderboard eligibility are separate checks.",
        "zh": "当归一化 status 属于成功集合且 error 为空时返回 True。产物是否存在、是否可上榜需另行校验。",
    },
    "worldfoundry.evaluation.api.WorldModelManifest": {
        "en": "Compact public DTO for model identity and capabilities after catalog resolution. Not a full YAML dump and not proof that a checkpoint loaded.",
        "zh": "目录解析后的精简公开 DTO，描述模型身份与能力。不是完整 YAML，也不能证明 checkpoint 已加载。",
    },
    "worldfoundry.evaluation.api.WorldModelConfig": {
        "en": "Construction payload for a runner: model id, runner target, variant, parameters, and runtime placement. Keep model-native knobs in parameters.",
        "zh": "交给 runner 的构造载荷：model id、runner 目标、variant、parameters 与 runtime 放置。模型私有旋钮放在 parameters 中。",
    },
    "worldfoundry.evaluation.api.WorldModelRunner": {
        "en": "Minimal runtime-checkable protocol: accept GenerationRequest(s) and return GenerationResult(s). Local checkpoints, APIs, and simulators can all implement it.",
        "zh": "最小可运行时检查的协议：接受 GenerationRequest，返回 GenerationResult。本地 checkpoint、远程 API、仿真器都可实现。",
    },
    "worldfoundry.pipelines.pipeline_utils.PipelineABC": {
        "en": "Model-facing pipeline base that owns load and native inference helpers. Pair with WorldModelRunner when evaluation needs a normalized boundary.",
        "zh": "面向模型的 pipeline 基类，负责加载与原生推理辅助。需要规范化评测边界时再与 WorldModelRunner 组合。",
    },
    "worldfoundry.evaluation.api.MetricSpec": {
        "en": "Declarative description of a metric’s identity, inputs, and aggregation expectations used by registries and scorecards.",
        "zh": "度量指标的声明式描述：身份、输入与聚合期望，供 registry 与 scorecard 使用。",
    },
    "worldfoundry.evaluation.api.MetricResult": {
        "en": "Per-sample metric output with score payload and optional diagnostics. AggregateResult rolls many of these up.",
        "zh": "单样本 metric 输出，含分数与可选诊断信息；AggregateResult 负责多样本汇总。",
    },
    "worldfoundry.evaluation.api.AggregateResult": {
        "en": "Dataset- or split-level aggregation over MetricResult rows (means, counts, custom summaries).",
        "zh": "对多条 MetricResult 的数据集/划分级汇总（均值、计数或自定义摘要）。",
    },
    "worldfoundry.evaluation.api.Metric": {
        "en": "Minimum metric implementation surface: score samples and optionally aggregate. Keep heavyweight backends behind this contract.",
        "zh": "Metric 的最小实现面：对样本打分，并可选择做聚合。重量级后端应藏在该契约之后。",
    },
    "worldfoundry.evaluation.api.EvaluationProtocolSpec": {
        "en": "Describes how a task/protocol evaluates generations (required artifacts, metrics, pass rules).",
        "zh": "描述某任务/协议如何评测生成结果（所需产物、指标与通过规则）。",
    },
    "worldfoundry.evaluation.api.WorldTaskConfig": {
        "en": "Task configuration bound to a protocol and dataset slice for one evaluation run.",
        "zh": "绑定协议与数据切片的任务配置，用于一次评测运行。",
    },
    "worldfoundry.evaluation.api.BenchmarkSpec": {
        "en": "Public benchmark identity and wiring metadata used by the hub and runners.",
        "zh": "公开 benchmark 身份与接线元数据，供 Hub 与 runner 使用。",
    },
    "worldfoundry.evaluation.public.WorldFoundryRunRequest": {
        "en": "Top-level request to launch an in-process WorldFoundry evaluation run (model, task, limits, output dirs).",
        "zh": "在进程内启动 WorldFoundry 评测的顶层请求（模型、任务、限制与输出目录）。",
    },
    "worldfoundry.evaluation.public.WorldFoundryRunResult": {
        "en": "Structured result of run_worldfoundry: paths to manifests, scorecards, and per-sample ledgers.",
        "zh": "run_worldfoundry 的结构化结果：manifest、scorecard 与逐样本 ledger 路径。",
    },
    "worldfoundry.evaluation.public.run_worldfoundry": {
        "en": "Canonical in-process entrypoint: execute a WorldFoundryRunRequest and return evidence artifacts.",
        "zh": "规范的进程内入口：执行 WorldFoundryRunRequest 并返回证据产物。",
    },
    "worldfoundry.evaluation.public.list_video_benchmarks": {
        "en": "List registered video benchmarks available to the public evaluation facade.",
        "zh": "列出公开评测 facade 可用的已注册视频 benchmark。",
    },
    "worldfoundry.evaluation.public.run_benchmark": {
        "en": "Run a named benchmark through the public facade with normalized inputs and outputs.",
        "zh": "通过公开 facade 运行指定 benchmark，输入输出均已规范化。",
    },
    "worldfoundry.evaluation.public.normalize_upstream_results": {
        "en": "Adapt upstream/vendor result payloads into WorldFoundry GenerationResult records.",
        "zh": "把上游/厂商结果载荷适配为 WorldFoundry 的 GenerationResult 记录。",
    },
    "worldfoundry.evaluation.public.benchmark_integration_spec": {
        "en": "Describe how an upstream benchmark integrates (entrypoints, artifacts, env needs).",
        "zh": "描述上游 benchmark 的接入方式（入口、产物与环境需求）。",
    },
    "worldfoundry.evaluation.reporting.build_env_requirements": {
        "en": "Collect environment requirement claims for a run manifest (packages, CUDA, assets).",
        "zh": "为 run manifest 收集环境需求声明（软件包、CUDA、资产等）。",
    },
    "worldfoundry.evaluation.reporting.build_environment": {
        "en": "Snapshot the executing environment for reproducibility evidence in the run manifest.",
        "zh": "快照当前执行环境，写入 run manifest 作为可复现证据。",
    },
    "worldfoundry.evaluation.reporting.build_run_manifest": {
        "en": "Build the run manifest document that records request, environment, and artifact index.",
        "zh": "构建记录请求、环境与产物索引的 run manifest 文档。",
    },
    "worldfoundry.evaluation.reporting.write_run_manifest_artifacts": {
        "en": "Persist run manifest JSON (and companions) into the run output directory.",
        "zh": "把 run manifest JSON（及附属文件）写入运行输出目录。",
    },
    "worldfoundry.evaluation.reporting.build_scorecard": {
        "en": "Assemble the scorecard summary from metric aggregates and protocol outcomes.",
        "zh": "从指标聚合与协议结果组装 scorecard 摘要。",
    },
    "worldfoundry.evaluation.reporting.write_scorecard": {
        "en": "Write scorecard artifacts to disk for leaderboards and human review.",
        "zh": "将 scorecard 产物落盘，供榜单与人工审阅使用。",
    },
    "worldfoundry.evaluation.reporting.build_run_summary": {
        "en": "Build a compact run summary structure used by reports and UIs.",
        "zh": "构建供报告与 UI 使用的精简运行摘要结构。",
    },
    "worldfoundry.evaluation.reporting.build_markdown_report": {
        "en": "Render a human-readable Markdown report from the run summary and scorecard.",
        "zh": "根据运行摘要与 scorecard 渲染人类可读的 Markdown 报告。",
    },
    "worldfoundry.evaluation.reporting.write_run_report_artifacts": {
        "en": "Write Markdown/HTML (or related) report files beside other run evidence.",
        "zh": "把 Markdown/HTML 等报告文件与其他运行证据一并写出。",
    },
    "worldfoundry.runtime.env.RequiredEnvReport": {
        "en": "Report of required vs present environment variables / tools for a workload.",
        "zh": "某工作负载所需环境变量/工具与当前是否具备的对照报告。",
    },
    "worldfoundry.runtime.env.WorldFoundryEnv": {
        "en": "Helpers for reading and validating WorldFoundry-related environment settings.",
        "zh": "读取与校验 WorldFoundry 相关环境设置的辅助接口。",
    },
    "worldfoundry.runtime.assets.LocalAsset": {
        "en": "Descriptor for a staged local asset (logical name → path) used by runners and tests.",
        "zh": "已落盘本地资产的描述（逻辑名 → 路径），供 runner 与测试使用。",
    },
    "worldfoundry.runtime.assets.expand_worldfoundry_path": {
        "en": "Expand WorldFoundry path tokens and ~ into concrete filesystem paths.",
        "zh": "展开 WorldFoundry 路径 token 与 ~，得到具体文件系统路径。",
    },
    "worldfoundry.runtime.assets.load_local_assets": {
        "en": "Load the local-assets mapping used to stage checkpoints, clips, and fixtures.",
        "zh": "加载用于暂存 checkpoint、片段与夹具的 local-assets 映射。",
    },
    "worldfoundry.runtime.jobs.run_bounded_command": {
        "en": "Run a subprocess with time/memory bounds and captured output for evaluation jobs.",
        "zh": "在时间/内存限制下运行子进程并捕获输出，供评测任务使用。",
    },
    # High-traffic Core symbols — bilingual reader-facing intros.
    "worldfoundry.core.scaled_dot_product_attention": {
        "en": "Exact scaled dot-product attention with an explicit backend context. Prefer this when Q/K/V already have split-head shapes; keep dropout_p=0.0 at inference.",
        "zh": "带显式后端上下文的精确 SDPA。当 Q/K/V 已是标准分头形状时优先使用；推理时保持 dropout_p=0.0。",
    },
    "worldfoundry.core.attention_forward": {
        "en": "Layout-aware attention entry that adapts einops-style QKV packs and optional fused providers through the dispatch policy.",
        "zh": "面向布局的注意力入口：适配 einops 风格 QKV，并经分发策略选择可选融合算子。",
    },
    "worldfoundry.core.flattened_multihead_attention": {
        "en": "Convenience wrapper for (batch, sequence, hidden) tensors that only need a head count to split/merge around SDPA.",
        "zh": "面向 (batch, sequence, hidden) 张量的便捷封装，只需头数即可在 SDPA 前后完成拆分/合并。",
    },
    "worldfoundry.core.resolve_attention_backend": {
        "en": "Resolve which attention backend to use from arguments, env, and capability probes.",
        "zh": "综合参数、环境变量与能力探测，解析应使用的注意力后端。",
    },
    "worldfoundry.core.attention.BlockKVCache": {
        "en": "Chunked rolling KV cache with a strict before_update → update → after_update lifecycle. Do not skip chunk indices.",
        "zh": "分块滚动 KV cache，必须遵循 before_update → update → after_update；不要跳过 chunk 下标。",
    },
    "worldfoundry.core.attention.NativeAttention": {
        "en": "Module form of Core SDPA that can attach a context-parallel process group.",
        "zh": "Core SDPA 的模块形态，可挂接 context-parallel 进程组。",
    },
    "worldfoundry.core.configuration.LazyCall": {
        "en": "Deferred constructor call used in LazyConfig graphs so objects are built only at instantiate time.",
        "zh": "LazyConfig 图中的延迟构造调用，对象只在 instantiate 时真正创建。",
    },
    "worldfoundry.core.configuration.instantiate": {
        "en": "Materialize a LazyConfig / LazyCall object graph into concrete Python objects.",
        "zh": "把 LazyConfig / LazyCall 对象图实例化为具体 Python 对象。",
    },
    "worldfoundry.core.load_state_dict": {
        "en": "Load weights into a module with Core’s placement and key-handling policy.",
        "zh": "按 Core 的放置与 key 处理策略，把权重载入模块。",
    },
    "worldfoundry.core.AttentionBackendInfo": {
        "en": "Resolved attention backend metadata after probing and normalization.",
        "zh": "完成探测与归一化后的注意力后端元数据。",
    },
    "worldfoundry.core.DropPath": {
        "en": "Stochastic depth module that drops residual paths per sample during training.",
        "zh": "训练时按样本丢弃残差路径的 Stochastic Depth 模块。",
    },
    "worldfoundry.core.LayerScale": {
        "en": "Learnable per-channel residual scaling used in modern Vision Transformer blocks.",
        "zh": "现代 ViT block 中常见的可学习逐通道残差缩放。",
    },
    "worldfoundry.core.SwiGLUFFN": {
        "en": "SwiGLU feed-forward layer shared across model integrations.",
        "zh": "各模型接入可复用的 SwiGLU 前馈层。",
    },
    "worldfoundry.core.destroy_model_parallel": {
        "en": "Tear down model-parallel process groups by clearing the stored group handles.",
        "zh": "清空已保存的进程组句柄，拆除 model-parallel 分组。",
    },
    "worldfoundry.core.drop_path": {
        "en": "Functional form of per-sample stochastic depth for residual branches.",
        "zh": "残差分支上按样本 Stochastic Depth 的函数式接口。",
    },
    "worldfoundry.core.exists_uri": {
        "en": "Return whether a WorldFoundry URI (local or remote scheme) currently exists.",
        "zh": "判断 WorldFoundry URI（本地或远程协议）当前是否存在。",
    },
    "worldfoundry.core.read_binary_uri": {
        "en": "Read all bytes from a URI through Core’s storage helpers.",
        "zh": "通过 Core 存储辅助，从 URI 读取全部字节。",
    },
    "worldfoundry.core.read_text_uri": {
        "en": "Read all text from a URI through Core’s storage helpers.",
        "zh": "通过 Core 存储辅助，从 URI 读取全部文本。",
    },
    "worldfoundry.core.mean_flat": {
        "en": "Average a tensor over all non-batch dimensions — common in diffusion losses.",
        "zh": "对除 batch 维以外的所有维度求均值，常见于 diffusion loss。",
    },
    "worldfoundry.core.init_pp_scheduler": {
        "en": "Initialize the pipeline-parallel scheduler singleton used by PP runs.",
        "zh": "初始化 pipeline-parallel 运行使用的 PPScheduler 单例。",
    },
    "worldfoundry.core.pp_scheduler": {
        "en": "Return the current pipeline-parallel scheduler instance.",
        "zh": "获取当前 pipeline-parallel 调度器实例。",
    },
}

# Short “where this sits” hints appended when no curated intro exists.
GROUP_HINTS: dict[str, dict[str, str]] = {
    "core-attention": {
        "en": "Belongs to Core attention (SDPA, backends, RoPE, packed sequences, KV cache).",
        "zh": "属于 Core 注意力（SDPA、后端、RoPE、packed sequence、KV cache）。",
    },
    "core-configuration": {
        "en": "Belongs to Core configuration (LazyConfig / deferred object graphs).",
        "zh": "属于 Core 配置（LazyConfig / 延迟对象图）。",
    },
    "core-io-media": {
        "en": "Belongs to Core I/O and media (paths, URIs, images, video, serialization).",
        "zh": "属于 Core I/O 与媒体（路径、URI、图像、视频、序列化）。",
    },
    "core-model-loading": {
        "en": "Belongs to Core model loading (checkpoints, state dicts, DiskMap, construction).",
        "zh": "属于 Core 模型加载（checkpoint、state dict、DiskMap、构造）。",
    },
    "core-distributed": {
        "en": "Belongs to Core distributed helpers (collectives and context-parallel split/gather).",
        "zh": "属于 Core 分布式辅助（集合通信与 context-parallel 切分/聚合）。",
    },
    "core-runtime": {
        "en": "Belongs to Core inference runtime (process setup, compile, timers, realtime).",
        "zh": "属于 Core 推理 Runtime（进程设置、编译、计时、realtime）。",
    },
    "core-nn-math": {
        "en": "Belongs to Core neural-net/math helpers (shared tensor transforms).",
        "zh": "属于 Core 神经网络/数学辅助（共享张量变换）。",
    },
    "core-acceleration-memory": {
        "en": "Belongs to Core acceleration and memory (caches, offload, VRAM, kernels).",
        "zh": "属于 Core 加速与内存（cache、offload、VRAM、kernels）。",
    },
    "core-foundations": {
        "en": "Belongs to Core foundations (registries, utilities, safety contracts).",
        "zh": "属于 Core 基础能力（registry、通用工具、安全契约）。",
    },
    "contracts": {
        "en": "Serializable evaluation contract — prefer worldfoundry.evaluation.api.",
        "zh": "可序列化评测契约，优先从 worldfoundry.evaluation.api 导入。",
    },
    "models": {
        "en": "Model/runner extension boundary for evaluation.",
        "zh": "面向评测的模型/runner 扩展边界。",
    },
    "metrics-tasks": {
        "en": "Metric and task contracts for scoring protocols.",
        "zh": "用于评分协议的 Metric 与 task 契约。",
    },
    "runs": {
        "en": "Public orchestration facade for runs and benchmarks.",
        "zh": "Run 与 benchmark 的公开编排 facade。",
    },
    "reporting": {
        "en": "Evidence builders for manifests, scorecards, and reports.",
        "zh": "manifest、scorecard 与报告的证据构建器。",
    },
    "runtime": {
        "en": "Runtime helpers for env, assets, and bounded subprocesses.",
        "zh": "环境、资产与受限子进程的 Runtime 辅助。",
    },
}
