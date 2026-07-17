import {
  Boxes,
  ClipboardCheck,
  Database,
  Eye,
  Gauge,
  Play,
  Search,
  SlidersHorizontal,
  TerminalSquare,
} from 'lucide-react';

type SystemMapLocale = 'en' | 'zh';

const workflowCopy = {
  en: {
    aria: 'WorldFoundry end-to-end workflow',
    handoff:
      'Artifacts are the handoff: model runtimes do not own benchmark logic, and benchmark runners do not load model checkpoints.',
    stages: [
      {
        label: 'Discover',
        title: 'Choose a model and benchmark',
        detail: 'Catalog manifests expose stable IDs, capabilities, readiness, assets, and blockers.',
        Icon: Search,
      },
      {
        label: 'Prepare',
        title: 'Stage runtime and assets',
        detail: 'Resolve the conda profile, checkpoints, datasets, metric weights, and credentials.',
        Icon: Database,
      },
      {
        label: 'Run',
        title: 'Generate through one contract',
        detail: 'TUI, CLI, scripts, and Studio dispatch to a pipeline and operator.',
        Icon: Play,
      },
      {
        label: 'Inspect',
        title: 'Review normalized artifacts',
        detail: 'Videos, geometry, actions, traces, and metadata remain visible and reusable.',
        Icon: Eye,
      },
      {
        label: 'Evaluate',
        title: 'Produce reviewable evidence',
        detail: 'Metrics and official runners write reports, blockers, and a normalized scorecard.',
        Icon: ClipboardCheck,
      },
    ],
  },
  zh: {
    aria: 'WorldFoundry 端到端工作流',
    handoff:
      'Artifact 是两侧的交接面：模型 runtime 不承载 benchmark 逻辑，benchmark runner 也不直接加载模型 checkpoint。',
    stages: [
      {
        label: '发现',
        title: '选择模型与 benchmark',
        detail: 'Catalog manifest 给出稳定 ID、能力、就绪状态、所需资产与 blocker。',
        Icon: Search,
      },
      {
        label: '准备',
        title: '准备 runtime 与资产',
        detail: '解析 conda profile、checkpoint、dataset、metric 权重与凭据。',
        Icon: Database,
      },
      {
        label: '运行',
        title: '通过统一契约生成',
        detail: 'TUI、CLI、脚本与 Studio 最终调度到 pipeline 和 operator。',
        Icon: Play,
      },
      {
        label: '检查',
        title: '审阅归一化 artifact',
        detail: '视频、几何、动作、trace 与 metadata 都保持可见、可复用。',
        Icon: Eye,
      },
      {
        label: '评测',
        title: '产出可审查证据',
        detail: 'Metric 与官方 runner 写出报告、blocker 和归一化 scorecard。',
        Icon: ClipboardCheck,
      },
    ],
  },
} as const;

const architectureCopy = {
  en: {
    aria: 'WorldFoundry architecture layers',
    caption:
      'Each layer owns one kind of change. Serializable contracts keep model-specific code, benchmark-specific code, and user interfaces from leaking into one another.',
    layers: [
      {
        label: 'Surfaces',
        purpose: 'Entry points for people and agents',
        items: ['TUI', 'CLI', 'Studio', 'Python API', 'MCP'],
        Icon: TerminalSquare,
      },
      {
        label: 'Control plane',
        purpose: 'Runnable entries, dependencies, and dispatch',
        items: ['Model catalog', 'Benchmark catalog', 'Readiness', 'Run plans'],
        Icon: SlidersHorizontal,
      },
      {
        label: 'Execution',
        purpose: 'Where model inference and benchmark scoring happen',
        items: ['Pipeline', 'Operator', 'Model runner', 'Benchmark runner'],
        Icon: Gauge,
      },
      {
        label: 'Contracts & evidence',
        purpose: 'What crosses subsystem boundaries and survives a run',
        items: ['Request', 'Result', 'Artifact manifest', 'Scorecard'],
        Icon: Boxes,
      },
    ],
  },
  zh: {
    aria: 'WorldFoundry 架构分层',
    caption:
      '每一层只拥有一类变化。可序列化契约阻止模型专属代码、benchmark 专属代码和用户界面相互渗透。',
    layers: [
      {
        label: '使用界面',
        purpose: '面向人与 agent 的操作入口',
        items: ['TUI', 'CLI', 'Studio', 'Python API', 'MCP'],
        Icon: TerminalSquare,
      },
      {
        label: '控制平面',
        purpose: '可运行项、依赖与调度',
        items: ['模型目录', 'Benchmark 目录', 'Readiness', 'Run plan'],
        Icon: SlidersHorizontal,
      },
      {
        label: '执行层',
        purpose: '模型推理与 benchmark 打分真正发生的位置',
        items: ['Pipeline', 'Operator', '模型 runner', 'Benchmark runner'],
        Icon: Gauge,
      },
      {
        label: '契约与证据',
        purpose: '跨子系统流动并在 run 结束后保留的内容',
        items: ['Request', 'Result', 'Artifact manifest', 'Scorecard'],
        Icon: Boxes,
      },
    ],
  },
} as const;

export function WorldFoundryWorkflow({
  locale = 'en',
  variant = 'default',
}: {
  locale?: SystemMapLocale;
  variant?: 'default' | 'home';
}) {
  const copy = workflowCopy[locale];
  const isHome = variant === 'home';

  return (
    <figure
      className={`wf-system-workflow${isHome ? ' wf-system-workflow--home' : ''}`}
      aria-label={copy.aria}
    >
      <ol className="wf-system-workflow-track">
        {copy.stages.map(({ label, title, detail, Icon }, index) => (
          <li
            className="wf-system-workflow-stage"
            key={label}
            style={isHome ? ({ '--wf-stage-index': index } as React.CSSProperties) : undefined}
          >
            <div className="wf-system-workflow-stage-head">
              <div className="wf-system-workflow-index" aria-hidden="true">
                {String(index + 1).padStart(2, '0')}
              </div>
              <div className="wf-system-workflow-icon-wrap" aria-hidden="true">
                <Icon className="wf-system-workflow-icon" size={18} strokeWidth={1.8} />
              </div>
            </div>
            <p className="wf-system-workflow-label">{label}</p>
            <h3>{title}</h3>
            <p>{detail}</p>
          </li>
        ))}
      </ol>
      <figcaption>{copy.handoff}</figcaption>
    </figure>
  );
}

export function WorldFoundryArchitecture({ locale = 'en' }: { locale?: SystemMapLocale }) {
  const copy = architectureCopy[locale];

  return (
    <figure className="wf-system-architecture" aria-label={copy.aria}>
      <div className="wf-system-architecture-stack">
        {copy.layers.map(({ label, purpose, items, Icon }, index) => (
          <section className="wf-system-architecture-layer" key={label}>
            <div className="wf-system-architecture-heading">
              <span>{String(index + 1).padStart(2, '0')}</span>
              <Icon aria-hidden="true" size={20} strokeWidth={1.7} />
              <div>
                <h3>{label}</h3>
                <p>{purpose}</p>
              </div>
            </div>
            <ul>
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </section>
        ))}
      </div>
      <figcaption>{copy.caption}</figcaption>
    </figure>
  );
}
