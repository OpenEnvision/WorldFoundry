export type ArchLocale = 'en' | 'zh';

export type ArchText = string | { en: string; zh: string };

export function archText(text: ArchText, locale: ArchLocale): string {
  return typeof text === 'string' ? text : text[locale];
}

export type ArchNode = {
  title: ArchText;
  role?: ArchText;
  detail?: ArchText;
  file?: ArchText;
  optional?: boolean;
  steps?: ArchText[];
};

export type ArchCallStep =
  | { type: 'lane'; label: ArchText }
  | { type: 'node'; id: string }
  | { type: 'fork'; ids: string[]; note?: ArchText }
  | { type: 'call'; label: ArchText; file?: ArchText };

export type ArchSequenceMessage = {
  from: number;
  to: number;
  label: ArchText;
  file?: ArchText;
  kind?: 'call' | 'return' | 'self';
};

export type ArchCallSpec = {
  kind: 'call';
  nodes: Record<string, ArchNode>;
  steps: ArchCallStep[];
};

export type ArchSequenceSpec = {
  kind: 'sequence';
  actors: { label: ArchText; detail?: ArchText }[];
  messages: ArchSequenceMessage[];
};

export type ArchBoundarySpec = {
  kind: 'boundary';
  leftTitle: ArchText;
  rightTitle: ArchText;
  leftCaption: ArchText;
  rightCaption: ArchText;
  boundaryLabel: ArchText;
  crossMain: ArchText;
  crossSub: ArchText;
  left: ArchNode[];
  right: ArchNode[];
};

export type ArchSplitSpec = {
  kind: 'split';
  leftTitle: ArchText;
  rightTitle: ArchText;
  left: ArchCallSpec;
  right: ArchCallSpec;
  mergeCall?: { label: ArchText; file?: ArchText };
  merge: ArchNode;
};

export type ArchDiagramSpec = {
  aria: ArchText;
  title: ArchText;
  kindLabel: ArchText;
  caption?: ArchText;
} & (ArchCallSpec | ArchSequenceSpec | ArchBoundarySpec | ArchSplitSpec);

export type ArchDiagramId =
  | 'system'
  | 'model-runtime'
  | 'model-runtime-bridge'
  | 'surfaces'
  | 'evaluation-core'
  | 'native-diffusion'
  | 'artifact-boundary'
  | 'workflow'
  | 'runtime-assembly'
  | 'studio'
  | 'studio-realtime'
  | 'catalog-scorecard'
  | 'task-materialize'
  | 'matrix-game';

const KIND = {
  call: { en: 'Call graph', zh: '调用图' },
  sequence: { en: 'Sequence', zh: '时序图' },
  boundary: { en: 'Boundary', zh: '边界图' },
  split: { en: 'Call graph', zh: '调用图' },
} as const;

const SYSTEM: ArchDiagramSpec = {
  kind: 'call',
  kindLabel: KIND.call,
  aria: {
    en: 'WorldFoundry ownership call graph from surfaces to reporting',
    zh: '从使用界面到 reporting 的 WorldFoundry 职责调用图',
  },
  title: {
    en: 'How a run moves through the system',
    zh: '一次运行如何穿过各层',
  },
  caption: {
    en: 'A surface compiles intent; execution generates and/or scores; reporting writes the evidence. Model runtime never imports a benchmark evaluator.',
    zh: '界面编译意图，执行侧生成或评分，reporting 写下证据。模型运行时不会 import benchmark evaluator。',
  },
  nodes: {
    surfaces: {
      role: { en: 'Surfaces', zh: '使用界面' },
      title: 'CLI / TUI / MCP / Studio',
      detail: {
        en: 'Turn operator intent into one of the five orchestration intents',
        zh: '把操作意图收成五种 orchestration intent 之一',
      },
    },
    orchestration: {
      role: { en: 'Orchestration', zh: '编排' },
      title: 'prepare_evaluation / execute_prepared',
      file: 'evaluation/tasks/execution/orchestration/service.py',
      detail: {
        en: 'Compile intent into PreparedEvaluation, then into a runner request',
        zh: '把 intent 编译成 PreparedEvaluation，再变成 runner request',
      },
    },
    model: {
      role: { en: 'Model runtime', zh: '模型运行时' },
      title: 'Pipeline · Operator · Synthesis',
      detail: {
        en: 'Load, generate, emit GenerationResult',
        zh: '加载、生成、写出 GenerationResult',
      },
    },
    benchmark: {
      role: { en: 'Benchmark runtime', zh: 'Benchmark 运行时' },
      title: 'Metrics · official runners',
      detail: {
        en: 'Read artifacts, never checkpoints',
        zh: '读 artifact，不读 checkpoint',
      },
    },
    reporting: {
      role: { en: 'Reporting', zh: '报告' },
      title: 'scorecard.json + report.md',
      detail: {
        en: 'Coverage, blockers, claim level, eligibility',
        zh: '覆盖、blocker、claim、合格性',
      },
    },
  },
  steps: [
    { type: 'lane', label: { en: 'Entry', zh: '入口' } },
    { type: 'node', id: 'surfaces' },
    { type: 'call', label: 'intent dataclass' },
    { type: 'lane', label: { en: 'Control plane', zh: '控制平面' } },
    { type: 'node', id: 'orchestration' },
    {
      type: 'call',
      label: 'execute_prepared_evaluation()',
      file: 'service.py:775',
    },
    { type: 'lane', label: { en: 'Execution', zh: '执行' } },
    {
      type: 'fork',
      ids: ['model', 'benchmark'],
      note: {
        en: 'Same request can generate, score existing files, or do both',
        zh: '同一请求可以只生成、只评分已有文件，或两者都做',
      },
    },
    { type: 'call', label: 'GenerationResult · metrics/summary.json' },
    { type: 'lane', label: { en: 'Evidence', zh: '证据' } },
    { type: 'node', id: 'reporting' },
  ],
};

const MODEL_RUNTIME: ArchDiagramSpec = {
  kind: 'call',
  kindLabel: KIND.call,
  aria: {
    en: 'Model runtime assembly: catalog to GenerationResult',
    zh: '模型 runtime 组装：从 catalog 到 GenerationResult',
  },
  title: {
    en: 'Runtime assembly',
    zh: 'Runtime 组装',
  },
  caption: {
    en: 'Representation and memory join only when the task needs geometry or multi-turn state. Every path still ends as GenerationResult.',
    zh: 'Representation 与 Memory 只在任务需要几何产物或多轮状态时介入。所有路径仍收敛到 GenerationResult。',
  },
  nodes: {
    catalog: {
      role: { en: 'Control plane', zh: '控制平面' },
      title: { en: 'Catalog binding', zh: 'Catalog 绑定' },
      detail: 'pipeline_target / runtime_profile',
      file: 'data/models/catalog/<category>/<model-id>.yaml',
    },
    pipeline: {
      role: { en: 'Runtime', zh: '运行时' },
      title: 'PipelineABC',
      detail: 'from_pretrained / process / stream',
      file: 'pipelines/pipeline_utils.py',
    },
    operator: {
      role: { en: 'Input', zh: '输入' },
      title: 'BaseOperator',
      detail: {
        en: 'validate, media, camera/action, interaction',
        zh: '校验、媒体、camera/action、interaction',
      },
      file: 'operators/base_operator.py',
    },
    synthesis: {
      role: { en: 'Inference', zh: '推理' },
      title: 'BaseSynthesis',
      detail: {
        en: 'Native runtime or hosted API. The loop below is local diffusion only.',
        zh: 'Native runtime 或 hosted API。下方循环仅本地 diffusion。',
      },
      file: 'synthesis/base_synthesis.py',
      steps: [
        { en: 'condition', zh: 'condition' },
        { en: 'init', zh: 'init' },
        { en: 'denoise', zh: 'denoise' },
        { en: 'decode', zh: 'decode' },
      ],
    },
    representation: {
      role: { en: 'Geometry', zh: '几何' },
      title: '3D model runtime',
      detail: {
        en: 'depth / point cloud / 3DGS / scene',
        zh: '深度 / 点云 / 3DGS / scene',
      },
      optional: true,
    },
    memory: {
      role: { en: 'State', zh: '状态' },
      title: 'BaseMemory',
      detail: {
        en: 'streaming / multi-turn prior state',
        zh: '流式 / 多轮前序状态',
      },
      optional: true,
    },
    result: {
      role: { en: 'Artifact', zh: '产物' },
      title: 'GenerationResult',
      detail: {
        en: 'Normalized outputs, status, timing, metadata',
        zh: '归一化输出、状态、耗时、metadata',
      },
    },
  },
  steps: [
    { type: 'lane', label: { en: 'Control plane', zh: '控制平面' } },
    { type: 'node', id: 'catalog' },
    {
      type: 'call',
      label: 'resolve_pipeline_route()',
      file: 'evaluation/models/pipelines/bindings.py:359',
    },
    { type: 'lane', label: { en: 'Runtime', zh: '运行时' } },
    { type: 'node', id: 'pipeline' },
    { type: 'call', label: 'operator.process_*()' },
    { type: 'node', id: 'operator' },
    { type: 'call', label: 'synthesis.predict()' },
    { type: 'node', id: 'synthesis' },
    {
      type: 'fork',
      ids: ['representation', 'memory'],
      note: {
        en: 'Optional — only when the task needs geometric outputs or multi-turn state',
        zh: '可选 — 仅当任务需要几何产物或多轮状态',
      },
    },
    { type: 'call', label: { en: 'converge on one artifact contract', zh: '收敛到同一套 artifact 契约' } },
    { type: 'node', id: 'result' },
  ],
};

const MODEL_RUNTIME_BRIDGE: ArchDiagramSpec = {
  kind: 'call',
  kindLabel: KIND.call,
  aria: {
    en: 'How evaluation resolves a catalog entry into a runnable model',
    zh: 'evaluation 如何把 catalog 条目解析成可运行模型',
  },
  title: {
    en: 'Bridge to evaluation',
    zh: '与 Evaluation 的衔接',
  },
  caption: {
    en: 'The evaluation runner never imports a pipeline file unless this chain resolves to that route.',
    zh: '除非这条链解析到对应 route，evaluation runner 不会直接 import pipeline 文件。',
  },
  nodes: {
    yaml: {
      role: { en: 'Catalog', zh: '目录' },
      title: '<model-id>.yaml',
      file: 'worldfoundry/data/models/catalog/<category>/<model-id>.yaml',
      detail: {
        en: 'Public identity, readiness, binding fields',
        zh: '公开身份、就绪状态、绑定字段',
      },
    },
    binding: {
      role: { en: 'Binding', zh: '绑定' },
      title: 'runner_target / pipeline_target / runtime_profile',
      detail: {
        en: 'Route strings — not a substitute for a pipeline/operator pair',
        zh: '路由字符串，不能替代 pipeline/operator pair',
      },
    },
    route: {
      role: { en: 'Resolver', zh: '解析' },
      title: 'resolve_pipeline_route()',
      file: 'evaluation/models/pipelines/bindings.py:359',
      detail: {
        en: 'Canonical binding helpers; do not re-parse route strings',
        zh: '唯一绑定入口；不要各自重解析 route string',
      },
    },
    runner: {
      role: { en: 'Resolver', zh: '解析' },
      title: 'resolve_model_zoo_runner()',
      file: 'evaluation/models/runners/resolver.py:386',
      detail: {
        en: 'Returns a WorldModelRunner instance',
        zh: '返回 WorldModelRunner 实例',
      },
    },
    exec: {
      role: { en: 'Execution', zh: '执行' },
      title: { en: 'Pipeline class or subprocess', zh: 'Pipeline 类或 subprocess' },
      detail: {
        en: 'from_pretrained → process / generate',
        zh: 'from_pretrained → process / generate',
      },
    },
    result: {
      role: { en: 'Artifact', zh: '产物' },
      title: 'GenerationResult',
      detail: {
        en: 'Shared contract for Studio, CLI, and benchmarks',
        zh: 'Studio、CLI 与 benchmark 共用的契约',
      },
    },
  },
  steps: [
    { type: 'node', id: 'yaml' },
    { type: 'call', label: { en: 'read binding fields', zh: '读取绑定字段' } },
    { type: 'node', id: 'binding' },
    { type: 'call', label: 'resolve_pipeline_route()', file: 'bindings.py:359' },
    { type: 'node', id: 'route' },
    { type: 'call', label: 'resolve_model_zoo_runner()', file: 'resolver.py:386' },
    { type: 'node', id: 'runner' },
    { type: 'call', label: 'generate()' },
    { type: 'node', id: 'exec' },
    { type: 'call', label: { en: 'normalize artifacts', zh: '归一化 artifact' } },
    { type: 'node', id: 'result' },
  ],
};

const SURFACES: ArchDiagramSpec = {
  kind: 'sequence',
  kindLabel: KIND.sequence,
  aria: {
    en: 'Sequence from a surface through orchestration into a delegate runner',
    zh: '从使用界面经编排进入 delegate runner 的时序',
  },
  title: {
    en: 'Surface → orchestration → execution',
    zh: '界面 → 编排 → 执行',
  },
  caption: {
    en: 'A new surface adds an intent-to-request mapping. It must not reimplement generation, artifact materialization, or scoring.',
    zh: '新增界面只加 intent→request 映射，不能重做 generation、artifact materialization 或打分。',
  },
  actors: [
    {
      label: { en: 'Surface', zh: '界面' },
      detail: 'CLI / TUI / MCP / Studio',
    },
    {
      label: { en: 'Orchestration', zh: '编排' },
      detail: 'service.py',
    },
    {
      label: { en: 'Execution', zh: '执行' },
      detail: { en: 'run_evaluate / run_model_benchmark', zh: 'run_evaluate / run_model_benchmark' },
    },
  ],
  messages: [
    {
      from: 0,
      to: 1,
      label: 'Intent',
      file: {
        en: 'ModelBenchmark · ScoreArtifacts · GenerateAndScore · ScoreResults · Reproduce',
        zh: 'ModelBenchmark · ScoreArtifacts · GenerateAndScore · ScoreResults · Reproduce',
      },
    },
    {
      from: 1,
      to: 1,
      kind: 'self',
      label: 'prepare_evaluation()',
      file: 'service.py:759',
    },
    {
      from: 1,
      to: 0,
      kind: 'return',
      label: 'PreparedEvaluation',
      file: { en: 'claim policy + readiness', zh: 'claim 策略 + 就绪状态' },
    },
    {
      from: 0,
      to: 1,
      label: 'execute_prepared_evaluation()',
      file: 'service.py:775',
    },
    {
      from: 1,
      to: 2,
      label: 'EvaluateRunRequest / ModelBenchmarkRunRequest',
    },
    {
      from: 2,
      to: 2,
      kind: 'self',
      label: { en: 'run_evaluate() / run_model_benchmark()', zh: 'run_evaluate() / run_model_benchmark()' },
      file: { en: 'see Workflow', zh: '见工作流' },
    },
  ],
};

const EVALUATION_CORE: ArchDiagramSpec = {
  kind: 'call',
  kindLabel: KIND.call,
  aria: {
    en: 'Prepared evaluation dispatches to a run path, then writes metrics and a scorecard',
    zh: 'Prepared evaluation 分发到一条运行路径，再写出 metrics 与 scorecard',
  },
  title: {
    en: 'Request to scorecard',
    zh: '从 request 到 scorecard',
  },
  caption: {
    en: 'execute_prepared_evaluation() dispatches. Evaluate runs go through run_evaluate(); model-plus-benchmark calls run_model_benchmark() directly.',
    zh: 'execute_prepared_evaluation() 负责分发。Evaluate 走 run_evaluate()；model-plus-benchmark 直接调用 run_model_benchmark()。',
  },
  nodes: {
    request: {
      role: { en: 'Entry', zh: '入口' },
      title: { en: 'Run request', zh: '运行请求' },
      detail: {
        en: 'EvaluateRunRequest or ModelBenchmarkRunRequest — compiled by orchestration',
        zh: 'EvaluateRunRequest 或 ModelBenchmarkRunRequest — 由编排编译',
      },
    },
    facade: {
      role: { en: 'Facade', zh: '门面' },
      title: 'execute_prepared_evaluation()',
      file: 'evaluation/tasks/execution/orchestration/service.py:775',
      detail: {
        en: 'Dispatch: EvaluateRunRequest → run_evaluate; ModelBenchmarkRunRequest → run_model_benchmark',
        zh: '分发：EvaluateRunRequest → run_evaluate；ModelBenchmarkRunRequest → run_model_benchmark',
      },
    },
    contract: {
      role: { en: 'Delegate', zh: '委派' },
      title: 'ContractRunner',
      file: 'orchestration/contract.py:96',
      detail: {
        en: 'In-process model + Metric objects',
        zh: '进程内模型 + Metric 对象',
      },
    },
    existing: {
      role: { en: 'Delegate', zh: '委派' },
      title: 'ExistingResultsRunner',
      file: 'orchestration/existing_results.py:93',
      detail: {
        en: 'Score artifacts already on disk',
        zh: '评分磁盘上已有 artifact',
      },
    },
    benchmark: {
      role: { en: 'Delegate', zh: '委派' },
      title: 'run_model_benchmark()',
      file: 'orchestration/model_benchmark.py:825',
      detail: {
        en: 'Generate, then official suite via ManifestBenchmarkRunner',
        zh: '先生成，再经 ManifestBenchmarkRunner 跑官方 suite',
      },
    },
    metrics: {
      role: { en: 'Metrics', zh: '指标' },
      title: 'compute_sample / aggregate',
      detail: {
        en: 'per_sample.jsonl → metrics/summary.json',
        zh: 'per_sample.jsonl → metrics/summary.json',
      },
    },
    scorecard: {
      role: { en: 'Report', zh: '报告' },
      title: 'scorecard.json + report.md',
      detail: {
        en: 'Eligibility record; Markdown is derived',
        zh: '合格性记录；Markdown 为派生',
      },
    },
  },
  steps: [
    { type: 'lane', label: { en: 'Entry', zh: '入口' } },
    { type: 'node', id: 'request' },
    { type: 'call', label: 'execute_prepared_evaluation()', file: 'service.py:775' },
    { type: 'node', id: 'facade' },
    { type: 'lane', label: { en: 'Delegate', zh: '委派' } },
    {
      type: 'fork',
      ids: ['contract', 'existing', 'benchmark'],
      note: {
        en: 'One delegate owns side effects',
        zh: '副作用只由一个 delegate 负责',
      },
    },
    { type: 'call', label: 'compute_sample() / aggregate()' },
    { type: 'lane', label: { en: 'Report', zh: '报告' } },
    { type: 'node', id: 'metrics' },
    { type: 'call', label: 'build_scorecard()', file: 'reporting/scorecard.py:136' },
    { type: 'node', id: 'scorecard' },
  ],
};

const NATIVE_DIFFUSION: ArchDiagramSpec = {
  kind: 'call',
  kindLabel: KIND.call,
  aria: {
    en: 'Native diffusion assembly from catalog identity to a normalized artifact',
    zh: '原生扩散从 catalog 身份装配到归一化 artifact',
  },
  title: {
    en: 'Native diffusion assembly',
    zh: '原生扩散装配',
  },
  caption: {
    en: 'One public model ID maps to one immutable recipe. Families reuse components; they do not own a second pipeline.',
    zh: '一个公开 model ID 对应一份不可变 recipe。Family 复用 component，不再自持第二套 pipeline。',
  },
  nodes: {
    catalog: {
      role: { en: 'Control plane', zh: '控制平面' },
      title: { en: 'Catalog / runtime profile', zh: 'Catalog / runtime profile' },
      detail: {
        en: 'Model ID + runtime policy',
        zh: 'Model ID + runtime 策略',
      },
    },
    pipeline: {
      role: { en: 'Boundary', zh: '边界' },
      title: 'NativeDiffusionPipeline',
      file: 'base_models/diffusion_model/pipeline.py:16',
      detail: {
        en: 'Public native execution entry',
        zh: '公开的原生执行入口',
      },
    },
    recipe: {
      role: { en: 'Recipe', zh: '配方' },
      title: { en: 'Immutable recipe', zh: '不可变 recipe' },
      file: 'base_models/diffusion_model/recipes/',
      detail: {
        en: 'Component + checkpoint specs for one model ID',
        zh: '针对一个 model ID 的 component 与 checkpoint spec',
      },
    },
    assembler: {
      role: { en: 'Assembly', zh: '装配' },
      title: 'NativeDiffusionAssembler',
      file: 'base_models/diffusion_model/assembly.py:31',
      detail: {
        en: 'Resolve, validate, construct the runner',
        zh: '解析、校验、构造 runner',
      },
    },
    runner: {
      role: { en: 'Execution', zh: '执行' },
      title: 'NativeDiffusionRunner',
      detail: {
        en: 'Framework-owned sampling loop',
        zh: '框架自有的采样循环',
      },
      steps: [
        'condition',
        'init',
        'schedule',
        'denoise',
        'decode',
      ],
    },
    artifact: {
      role: { en: 'Output', zh: '产物' },
      title: { en: 'Normalized artifact', zh: '归一化 artifact' },
      detail: {
        en: 'WorldFoundry GenerationResult semantics',
        zh: 'WorldFoundry GenerationResult 语义',
      },
    },
  },
  steps: [
    { type: 'node', id: 'catalog' },
    { type: 'call', label: { en: 'public pipeline / operator / synthesis boundary', zh: '公开 pipeline / operator / synthesis 边界' } },
    { type: 'node', id: 'pipeline' },
    { type: 'call', label: { en: 'model ID → recipe', zh: 'model ID → recipe' } },
    { type: 'node', id: 'recipe' },
    { type: 'call', label: 'assemble()', file: 'assembly.py:31' },
    { type: 'node', id: 'assembler' },
    { type: 'call', label: { en: 'core loaders + runtime policy', zh: 'core loader + runtime 策略' } },
    { type: 'node', id: 'runner' },
    { type: 'call', label: { en: 'emit WorldFoundry artifact', zh: '写出 WorldFoundry artifact' } },
    { type: 'node', id: 'artifact' },
  ],
};

const ARTIFACT_BOUNDARY: ArchDiagramSpec = {
  kind: 'boundary',
  kindLabel: KIND.boundary,
  aria: {
    en: 'Model execution and benchmark evaluation meet only at GenerationResult',
    zh: '模型执行与 benchmark 评测只在 GenerationResult 相遇',
  },
  title: {
    en: 'The artifact boundary',
    zh: 'Artifact 边界',
  },
  caption: {
    en: 'Nothing else crosses: no checkpoint paths into metrics, no evaluator imports into pipelines.',
    zh: '只有这一层能穿过：metric 看不到 checkpoint 路径，pipeline 也不 import evaluator。',
  },
  leftTitle: { en: 'Model execution', zh: '模型执行' },
  rightTitle: { en: 'Benchmark / reporting', zh: '评测 / 报告' },
  leftCaption: {
    en: 'Model side never imports a benchmark evaluator',
    zh: '模型侧不 import 任何 benchmark evaluator',
  },
  rightCaption: {
    en: 'Benchmark side never loads a model checkpoint',
    zh: 'Benchmark 侧不加载任何模型 checkpoint',
  },
  boundaryLabel: { en: 'Artifact boundary', zh: 'Artifact 边界' },
  crossMain: 'GenerationResult',
  crossSub: 'artifacts.jsonl',
  left: [
    {
      role: { en: 'Input', zh: '输入' },
      title: 'GenerationRequest',
      detail: {
        en: 'sample id, media, text/action, params',
        zh: '样本 id、媒体、text/action、参数',
      },
    },
    {
      role: { en: 'Runtime', zh: '运行时' },
      title: 'Pipeline + Operator',
      detail: {
        en: 'Checkpoint load + native call',
        zh: 'checkpoint 加载 + 原生调用',
      },
    },
    {
      role: { en: 'Inference', zh: '推理' },
      title: { en: 'Native inference', zh: '原生推理' },
      detail: {
        en: 'Local runtime or hosted API',
        zh: '本地 runtime 或托管 API',
      },
    },
  ],
  right: [
    {
      role: { en: 'Evaluation', zh: '评测' },
      title: { en: 'Benchmark runner', zh: 'Benchmark runner' },
      detail: {
        en: 'Reads artifacts, not checkpoints',
        zh: '读 artifact，不读 checkpoint',
      },
    },
    {
      role: { en: 'Metrics', zh: '指标' },
      title: 'per_sample + summary',
      detail: {
        en: 'Per-sample rows + aggregates',
        zh: '逐样本行 + 聚合',
      },
    },
    {
      role: { en: 'Report', zh: '报告' },
      title: 'scorecard.json',
      detail: {
        en: 'Eligibility record; report.md is derived',
        zh: '合格性记录；report.md 为派生',
      },
    },
  ],
};

const WORKFLOW: ArchDiagramSpec = {
  kind: 'call',
  kindLabel: KIND.call,
  aria: {
    en: 'Evaluate workflow from CLI through a delegate runner to the artifact bundle',
    zh: '从 CLI 经 delegate runner 到 artifact bundle 的 evaluate 工作流',
  },
  title: {
    en: 'Evaluate workflow',
    zh: 'Evaluate 主流程',
  },
  caption: {
    en: 'If an output file is missing, the stage that should have written it is on this graph.',
    zh: '输出目录缺文件时，沿此图找本应写出它的阶段。',
  },
  nodes: {
    cli: {
      role: { en: 'Entry', zh: '入口' },
      title: 'cli.main()',
      file: 'worldfoundry/cli/main.py:1888',
      detail: {
        en: 'evaluate / run / existing outputs / model+benchmark',
        zh: 'evaluate / run / 已有输出 / model+benchmark',
      },
    },
    facade: {
      role: { en: 'Facade', zh: '门面' },
      title: 'run_evaluate()',
      file: 'evaluation/tasks/execution/orchestration/evaluate.py:1071',
      detail: {
        en: 'Validate, choose mode, hand off',
        zh: '校验、选 mode、移交',
      },
    },
    materialize: {
      role: { en: 'Inputs', zh: '输入' },
      title: 'load_or_materialize()',
      detail: {
        en: 'Task registry or disk paths → request/result rows',
        zh: 'Task registry 或磁盘路径 → request/result 行',
      },
    },
    contract: {
      role: { en: 'Delegate', zh: '委派' },
      title: 'ContractRunner',
      detail: {
        en: 'Model + in-process Metric objects',
        zh: '模型 + 进程内 Metric 对象',
      },
    },
    existing: {
      role: { en: 'Delegate', zh: '委派' },
      title: 'ExistingResultsRunner',
      detail: {
        en: 'Score files that already exist',
        zh: '评分已存在的文件',
      },
    },
    bundle: {
      role: { en: 'Disk', zh: '磁盘' },
      title: { en: 'Artifact bundle', zh: 'Artifact bundle' },
      detail: 'run_manifest.json · requests.jsonl · results.jsonl · scorecard.json · report.md',
    },
  },
  steps: [
    { type: 'node', id: 'cli' },
    { type: 'call', label: 'run_evaluate(EvaluateRunRequest)' },
    { type: 'node', id: 'facade' },
    { type: 'call', label: '_mode(request) · load_or_materialize()' },
    { type: 'node', id: 'materialize' },
    {
      type: 'fork',
      ids: ['contract', 'existing'],
      note: {
        en: 'Model mode with string metric IDs also lands on ExistingResultsRunner after generate()',
        zh: 'model mode 若传入 metric ID 字符串，generate() 之后也会落到 ExistingResultsRunner',
      },
    },
    { type: 'call', label: 'runner.run(requests, results)' },
    { type: 'node', id: 'bundle' },
  ],
};

const RUNTIME_ASSEMBLY: ArchDiagramSpec = {
  kind: 'call',
  kindLabel: KIND.call,
  aria: {
    en: 'Base-class call chain that assembles a model invocation',
    zh: '组装一次模型调用的基类调用链',
  },
  title: {
    en: 'Base-class assembly',
    zh: '基类组装链',
  },
  caption: {
    en: 'Every runnable method needs both a pipeline and an operator. Operators may be thin; they must exist.',
    zh: '每个可运行方法都需要 pipeline 与 operator。operator 可以很薄，但不能缺。',
  },
  nodes: {
    pipeline: {
      role: { en: 'Entrypoint', zh: '入口' },
      title: 'PipelineABC',
      file: 'pipelines/pipeline_utils.py:11',
      detail: 'from_pretrained() · process() · stream()',
    },
    operator: {
      role: { en: 'Input', zh: '输入' },
      title: 'BaseOperator',
      file: 'operators/base_operator.py:4',
      detail: {
        en: 'Validate / load / shape interactions and media',
        zh: '校验 / 加载 / 整形 interaction 与媒体',
      },
    },
    synthesis: {
      role: { en: 'Inference', zh: '推理' },
      title: 'BaseSynthesis',
      file: 'synthesis/base_synthesis.py:38',
      detail: {
        en: 'Native runtime call or hosted API',
        zh: 'Native runtime 调用或 hosted API',
      },
    },
    diffusion: {
      role: { en: 'Local diffusion', zh: '本地扩散' },
      title: 'NativeDiffusionPipeline',
      optional: true,
      detail: {
        en: 'Only for local diffusion families',
        zh: '仅本地 diffusion family',
      },
    },
    representation: {
      role: { en: 'Geometry', zh: '几何' },
      title: '3D model runtime',
      file: 'base_models/three_dimensions/',
      optional: true,
      detail: {
        en: 'depth, point cloud, 3DGS, panorama, geometry',
        zh: '深度、点云、3DGS、panorama、geometry',
      },
    },
    memory: {
      role: { en: 'State', zh: '状态' },
      title: 'BaseMemory',
      file: 'core/memory/base.py:9',
      optional: true,
      detail: {
        en: 'Prior state for streaming or multi-turn control',
        zh: '为流式或多轮控制保留前序状态',
      },
    },
    result: {
      role: { en: 'Artifact', zh: '产物' },
      title: 'GenerationResult',
    },
  },
  steps: [
    { type: 'node', id: 'pipeline' },
    { type: 'call', label: { en: 'shape inputs', zh: '整形输入' } },
    { type: 'node', id: 'operator' },
    { type: 'call', label: 'predict() / from_pretrained()' },
    { type: 'fork', ids: ['synthesis', 'diffusion'] },
    { type: 'call', label: { en: 'optional post-inference roles', zh: '可选的推理后职责' } },
    {
      type: 'fork',
      ids: ['representation', 'memory'],
      note: {
        en: 'Optional geometry and multi-turn state',
        zh: '可选的几何产物与多轮状态',
      },
    },
    { type: 'node', id: 'result' },
  ],
};

const STUDIO: ArchDiagramSpec = {
  kind: 'call',
  kindLabel: KIND.call,
  aria: {
    en: 'Studio launch call chain from args to UI update',
    zh: 'Studio 从启动参数到 UI 更新的调用链',
  },
  title: {
    en: 'Studio execution path',
    zh: 'Studio 执行路径',
  },
  caption: {
    en: 'Inference stays in studio/execution.py. app.py is layout and callbacks only.',
    zh: '推理语义留在 studio/execution.py。app.py 只负责布局与回调。',
  },
  nodes: {
    args: {
      role: { en: 'Launch', zh: '启动' },
      title: { en: 'CLI flags', zh: '启动参数' },
      detail: {
        en: 'Parse worldfoundry-studio flags into a launch config',
        zh: '把 worldfoundry-studio 的 flags 收成启动配置',
      },
      file: 'studio/app.py',
    },
    config: {
      role: { en: 'Shell', zh: '界面壳' },
      title: 'StudioLaunchConfig',
      detail: {
        en: 'Gradio layout, callbacks, and launch defaults — no inference',
        zh: 'Gradio 布局、回调与启动默认值，不含推理',
      },
      file: 'studio/app.py',
    },
    catalog: {
      role: { en: 'Discovery', zh: '发现' },
      title: 'Catalog',
      detail: {
        en: 'AST-scan pipeline modules without importing CUDA stacks',
        zh: '用 AST 扫描 pipeline 模块，不 import CUDA 栈',
      },
      file: 'studio/catalog.py',
    },
    manager: {
      role: { en: 'Orchestration', zh: '编排' },
      title: 'StudioManager',
      detail: {
        en: 'Import the pipeline, pick a driver, stage inputs, write RunRecord',
        zh: '导入 pipeline、选择 driver、准备输入、写出 RunRecord',
      },
      file: 'studio/execution.py',
    },
    inputs: {
      role: { en: 'Prepare', zh: '准备' },
      title: { en: 'Staged inputs', zh: '已落盘输入' },
      detail: {
        en: 'Stage prompts, media, interactions, and runtime kwargs',
        zh: '落盘 prompt、媒体、interaction 与 runtime kwargs',
      },
      file: 'studio/execution.py',
    },
    driver: {
      role: { en: 'Dispatch', zh: '调度' },
      title: 'Driver',
      detail: {
        en: 'Base driver, or a 3DGS / point-cloud / WorldFM specialist',
        zh: '默认 driver，或 3DGS / 点云 / WorldFM 专用实现',
      },
      file: 'studio/execution.py',
    },
    run: {
      role: { en: 'Execute', zh: '执行' },
      title: 'RunRecord',
      detail: {
        en: 'Invoke the driver, then write files and a RunRecord',
        zh: '调用 driver，再写出文件与 RunRecord',
      },
      file: 'studio/execution.py',
    },
    preview: {
      role: { en: 'Preview', zh: '预览' },
      title: { en: 'Preview assets', zh: '预览资源' },
      detail: {
        en: 'Rank video, image, and splat for Gradio or Workspace',
        zh: '为 Gradio 或 Workspace 挑选视频、图像与 splat',
      },
      file: 'studio/execution.py',
    },
  },
  steps: [
    { type: 'lane', label: { en: 'Launch', zh: '启动' } },
    { type: 'node', id: 'args' },
    { type: 'call', label: 'parse_launch_config()' },
    { type: 'node', id: 'config' },
    { type: 'call', label: 'discover_catalog()' },
    { type: 'lane', label: { en: 'Runtime', zh: '运行时' } },
    { type: 'node', id: 'catalog' },
    { type: 'call', label: 'StudioManager()' },
    { type: 'node', id: 'manager' },
    { type: 'call', label: 'prepare_inputs()' },
    { type: 'node', id: 'inputs' },
    { type: 'call', label: 'runtime_driver_for()' },
    { type: 'node', id: 'driver' },
    { type: 'call', label: 'run()' },
    { type: 'lane', label: { en: 'Output', zh: '产出' } },
    { type: 'node', id: 'run' },
    { type: 'call', label: 'materialize_run()' },
    { type: 'node', id: 'preview' },
  ],
};

const STUDIO_REALTIME: ArchDiagramSpec = {
  kind: 'call',
  kindLabel: KIND.call,
  aria: {
    en: 'Realtime serving call chain from key edges to transport',
    zh: '从按键边沿到传输的实时服务调用链',
  },
  title: {
    en: 'Realtime serving',
    zh: '实时服务',
  },
  caption: {
    en: 'Models own causal cadence through RealtimeSpec. Transports must not infer latent strides.',
    zh: '模型通过 RealtimeSpec 拥有因果时序。Transport 不得猜测 latent stride。',
  },
  nodes: {
    keys: {
      role: { en: 'Input', zh: '输入' },
      title: { en: 'Browser key edges', zh: '浏览器按键边沿' },
    },
    resampler: {
      role: { en: 'Control', zh: '控制' },
      title: 'RealtimeControlResampler',
    },
    runtime: {
      role: { en: 'Model', zh: '模型' },
      title: 'ResidentWorldRuntime',
      detail: {
        en: 'One worker thread',
        zh: '单一 worker thread',
      },
    },
    frames: {
      role: { en: 'Extract', zh: '提取' },
      title: 'realtime_frames_from_result()',
    },
    post: {
      role: { en: 'Post', zh: '后处理' },
      title: 'VideoPostprocessStream',
      detail: {
        en: 'One session-scoped chain',
        zh: '会话级状态链',
      },
    },
    transport: {
      role: { en: 'Transport', zh: '传输' },
      title: { en: 'Resolution / JPEG work', zh: '分辨率 / JPEG 处理' },
    },
    queue: {
      role: { en: 'Queue', zh: '队列' },
      title: 'latest-interactive / ordered-quality',
    },
    wire: {
      role: { en: 'Wire', zh: '线路' },
      title: 'WebRTC / WebSocket',
    },
  },
  steps: [
    { type: 'node', id: 'keys' },
    { type: 'call', label: { en: 'resample control edges', zh: '重采样控制边沿' } },
    { type: 'node', id: 'resampler' },
    { type: 'call', label: { en: 'step resident runtime', zh: '步进驻留 runtime' } },
    { type: 'node', id: 'runtime' },
    { type: 'call', label: 'realtime_frames_from_result()' },
    { type: 'node', id: 'frames' },
    { type: 'call', label: { en: 'session postprocess chain', zh: '会话后处理链' } },
    { type: 'node', id: 'post' },
    { type: 'call', label: { en: 'encode for transport', zh: '编码以便传输' } },
    { type: 'node', id: 'transport' },
    { type: 'call', label: { en: 'enqueue by policy', zh: '按策略入队' } },
    { type: 'node', id: 'queue' },
    { type: 'node', id: 'wire' },
  ],
};

const CATALOG_SCORECARD: ArchDiagramSpec = {
  kind: 'split',
  kindLabel: KIND.split,
  aria: {
    en: 'Model and benchmark catalogs meet at run_evaluate',
    zh: '模型与 benchmark catalog 在 run_evaluate 汇合',
  },
  title: {
    en: 'Catalog to scorecard',
    zh: '从 Catalog 到 Scorecard',
  },
  caption: {
    en: 'The two catalogs stay parallel until a request carries both a model_id and a benchmark_id, or materialization feeds existing-results.',
    zh: '两条 catalog 保持并行，直到 request 同时带 model_id 与 benchmark_id，或 materialization 为 existing-results 供数。',
  },
  leftTitle: { en: 'Model catalog', zh: '模型目录' },
  rightTitle: { en: 'Benchmark catalog', zh: 'Benchmark 目录' },
  left: {
    kind: 'call',
    nodes: {
      yaml: {
        role: { en: 'YAML', zh: 'YAML' },
        title: 'models/catalog/*.yaml',
      },
      schema: {
        role: { en: 'Parse', zh: '解析' },
        title: 'Schema + ModelZooRegistry',
      },
      resolve: {
        role: { en: 'Resolve', zh: '解析' },
        title: 'resolve_model_zoo_runner()',
        file: 'evaluation/models/runners/resolver.py:386',
      },
    },
    steps: [
      { type: 'node', id: 'yaml' },
      { type: 'call', label: 'parse_model_zoo_yaml()' },
      { type: 'node', id: 'schema' },
      { type: 'call', label: 'resolve_model_zoo_runner()' },
      { type: 'node', id: 'resolve' },
    ],
  },
  right: {
    kind: 'call',
    nodes: {
      yaml: {
        role: { en: 'YAML', zh: 'YAML' },
        title: 'benchmarks/catalog/*.yaml',
      },
      tasks: {
        role: { en: 'Tasks', zh: '任务' },
        title: 'benchmarks/tasks/external/*.yaml',
      },
      spec: {
        role: { en: 'Spec', zh: '规格' },
        title: 'BenchmarkSpec + task registry',
      },
    },
    steps: [
      { type: 'node', id: 'yaml' },
      { type: 'call', label: 'parse_benchmark_zoo_yaml()' },
      { type: 'node', id: 'tasks' },
      { type: 'call', label: { en: 'materialize GenerationRequests', zh: '物化 GenerationRequests' } },
      { type: 'node', id: 'spec' },
    ],
  },
  mergeCall: {
    label: 'execute_prepared_evaluation()',
    file: 'service.py:775',
  },
  merge: {
    role: { en: 'Evidence', zh: '证据' },
    title: 'scorecard.json + report.md',
  },
};

const TASK_MATERIALIZE: ArchDiagramSpec = {
  kind: 'call',
  kindLabel: KIND.call,
  aria: {
    en: 'Task YAML and optional dataset rows become an EvaluateRunRequest',
    zh: 'Task YAML 与可选 dataset 行变成 EvaluateRunRequest',
  },
  title: {
    en: 'Task materialization',
    zh: 'Task materialization',
  },
  caption: {
    en: 'Prove this step with worldfoundry-eval task materialize before spending GPU time.',
    zh: '耗 GPU 之前用 worldfoundry-eval task materialize 证明此步。',
  },
  nodes: {
    task: {
      role: { en: 'Protocol', zh: '协议' },
      title: { en: 'Task YAML (extends chains)', zh: 'Task YAML（含 extends 链）' },
    },
    dataset: {
      role: { en: 'Data', zh: '数据' },
      title: { en: 'Dataset manifest (optional)', zh: 'Dataset manifest（可选）' },
      optional: true,
    },
    registry: {
      role: { en: 'Lookup', zh: '查找' },
      title: { en: 'Task registry lookup', zh: 'Task registry 查找' },
    },
    plan: {
      role: { en: 'Plan', zh: '计划' },
      title: 'RunPlan fingerprint',
      file: 'orchestration/plan.py:113',
    },
    rows: {
      role: { en: 'Materialize', zh: '物化' },
      title: 'materialize_generation_requests()',
    },
    request: {
      role: { en: 'Request', zh: '请求' },
      title: 'EvaluateRunRequest',
    },
  },
  steps: [
    { type: 'fork', ids: ['task', 'dataset'] },
    { type: 'call', label: { en: 'registry lookup', zh: 'registry 查找' } },
    { type: 'node', id: 'registry' },
    { type: 'call', label: { en: 'fingerprint plan', zh: '计算 plan 指纹' } },
    { type: 'node', id: 'plan' },
    { type: 'call', label: 'materialize_generation_requests()' },
    { type: 'node', id: 'rows' },
    { type: 'node', id: 'request' },
  ],
};

const MATRIX_GAME: ArchDiagramSpec = {
  kind: 'call',
  kindLabel: KIND.call,
  aria: {
    en: 'Matrix-Game 2 example crossing the artifact boundary',
    zh: 'Matrix-Game 2 示例跨过 artifact 边界',
  },
  title: {
    en: 'Matrix-Game 2 across the boundary',
    zh: 'Matrix-Game 2 跨过边界',
  },
  caption: {
    en: 'Evaluation only needs artifact kind, URI, and sample identity — not how weights were loaded.',
    zh: '评测端只需 artifact 的 kind、URI 与 sample identity，不需要知道权重如何加载。',
  },
  nodes: {
    input: {
      role: { en: 'Input', zh: '输入' },
      title: { en: 'Image + action sequence + seed/fps', zh: '初始图像 + action 序列 + seed/fps' },
      file: 'data/test_cases/matrix-game-2/universal/0000.png',
    },
    operator: {
      role: { en: 'Operator', zh: '算子' },
      title: 'Matrix-Game 2 operator',
    },
    pipeline: {
      role: { en: 'Pipeline', zh: '流水线' },
      title: { en: 'Pipeline / checkpoint', zh: 'Pipeline / checkpoint' },
    },
    video: {
      role: { en: 'Output', zh: '输出' },
      title: 'generated_video + metadata',
    },
    result: {
      role: { en: 'Contract', zh: '契约' },
      title: 'ArtifactRef in GenerationResult',
    },
    review: {
      role: { en: 'Consume', zh: '消费' },
      title: {
        en: 'Visual review or a compatible metric',
        zh: '视觉 review，或读取 generated_video 的兼容 metric',
      },
    },
  },
  steps: [
    { type: 'node', id: 'input' },
    { type: 'call', label: { en: 'shape native call', zh: '整形为原生调用' } },
    { type: 'node', id: 'operator' },
    { type: 'call', label: { en: 'run checkpoint', zh: '执行 checkpoint' } },
    { type: 'node', id: 'pipeline' },
    { type: 'call', label: { en: 'write video + metadata', zh: '写出视频与 metadata' } },
    { type: 'node', id: 'video' },
    { type: 'call', label: { en: 'wrap as artifact ref', zh: '包装为 artifact ref' } },
    { type: 'node', id: 'result' },
    { type: 'call', label: { en: 'cross the boundary', zh: '穿过边界' } },
    { type: 'node', id: 'review' },
  ],
};

export const ARCH_DIAGRAMS: Record<ArchDiagramId, ArchDiagramSpec> = {
  system: SYSTEM,
  'model-runtime': MODEL_RUNTIME,
  'model-runtime-bridge': MODEL_RUNTIME_BRIDGE,
  surfaces: SURFACES,
  'evaluation-core': EVALUATION_CORE,
  'native-diffusion': NATIVE_DIFFUSION,
  'artifact-boundary': ARTIFACT_BOUNDARY,
  workflow: WORKFLOW,
  'runtime-assembly': RUNTIME_ASSEMBLY,
  studio: STUDIO,
  'studio-realtime': STUDIO_REALTIME,
  'catalog-scorecard': CATALOG_SCORECARD,
  'task-materialize': TASK_MATERIALIZE,
  'matrix-game': MATRIX_GAME,
};
