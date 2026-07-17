import type { CSSProperties } from 'react';
import {
  ClipboardCheck,
  Database,
  Eye,
  Play,
  Search,
  type LucideIcon,
} from 'lucide-react';
import type { Locale } from '@/lib/i18n';

type WorkflowStage = {
  label: string;
  title: string;
  Icon: LucideIcon;
};

const copy = {
  en: {
    aria: 'WorldFoundry end-to-end workflow',
    stages: [
      { label: 'Discover', title: 'Model & benchmark', Icon: Search },
      { label: 'Prepare', title: 'Runtime & assets', Icon: Database },
      { label: 'Run', title: 'Unified generation', Icon: Play },
      { label: 'Inspect', title: 'Durable artifacts', Icon: Eye },
      { label: 'Evaluate', title: 'Reviewable evidence', Icon: ClipboardCheck },
    ],
  },
  zh: {
    aria: 'WorldFoundry 端到端工作流',
    stages: [
      { label: '发现', title: '模型与 benchmark', Icon: Search },
      { label: '准备', title: 'Runtime 与资产', Icon: Database },
      { label: '运行', title: '统一生成', Icon: Play },
      { label: '检查', title: '持久 artifact', Icon: Eye },
      { label: '评测', title: '可审查证据', Icon: ClipboardCheck },
    ],
  },
} as const satisfies Record<
  Locale,
  { aria: string; stages: readonly WorkflowStage[] }
>;

export function DocsWelcomeWorkflow({ locale }: { locale: Locale }) {
  const labels = copy[locale];

  return (
    <div className="pi-doc-welcome-workflow" aria-label={labels.aria}>
      <ol className="pi-doc-welcome-workflow-track">
        {labels.stages.map((stage, index) => (
          <li
            className="pi-doc-welcome-workflow-stage"
            key={stage.label}
            style={{ '--pi-workflow-index': index } as CSSProperties}
          >
            <article className="pi-doc-welcome-workflow-card">
              <div className="pi-doc-welcome-workflow-marker" aria-hidden="true">
                <span className="pi-doc-welcome-workflow-icon-wrap">
                  <span className="pi-doc-welcome-workflow-icon">
                    <stage.Icon size={22} strokeWidth={1.75} />
                  </span>
                  <span className="pi-doc-welcome-workflow-index">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                </span>
              </div>
              <p className="pi-doc-welcome-workflow-label">{stage.label}</p>
              <p className="pi-doc-welcome-workflow-title">{stage.title}</p>
            </article>
          </li>
        ))}
      </ol>
    </div>
  );
}
