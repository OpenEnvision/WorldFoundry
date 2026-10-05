import type { CSSProperties } from 'react';
import Link from 'next/link';
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
  href: string;
  label: string;
  title: string;
  Icon: LucideIcon;
};

const STAGES = [
  { href: '/guides/supported-models', Icon: Search },
  { href: '/guides/tui', Icon: Database },
  { href: '/guides/studio', Icon: Play },
  { href: '/guides/inference', Icon: Eye },
  { href: '/evaluation', Icon: ClipboardCheck },
] as const;

const copy = {
  en: {
    aria: 'WorldFoundry end-to-end workflow',
    stages: [
      { label: 'Discover', title: 'Model & benchmark' },
      { label: 'TUI', title: 'Runtime & assets' },
      { label: 'Studio', title: 'Unified generation' },
      { label: 'Inspect', title: 'Durable artifacts' },
      { label: 'Evaluate', title: 'Reviewable evidence' },
    ],
  },
  zh: {
    aria: 'WorldFoundry 端到端工作流',
    stages: [
      { label: '发现', title: '模型与 benchmark' },
      { label: 'TUI', title: 'Runtime 与资产' },
      { label: 'Studio', title: '统一生成' },
      { label: '检查', title: '持久 artifact' },
      { label: '评测', title: '可审查证据' },
    ],
  },
} as const satisfies Record<
  Locale,
  { aria: string; stages: readonly { label: string; title: string }[] }
>;

function docsHref(locale: Locale, path: string) {
  return locale === 'zh' ? `/zh/docs${path}` : `/docs${path}`;
}

export function DocsWelcomeWorkflow({ locale }: { locale: Locale }) {
  const labels = copy[locale];
  const stages: WorkflowStage[] = labels.stages.map((stage, index) => ({
    ...stage,
    ...STAGES[index],
  }));

  return (
    <div className="pi-doc-welcome-workflow">
      <ol className="pi-doc-welcome-workflow-track" aria-label={labels.aria}>
        <li className="pi-doc-welcome-workflow-tick" aria-hidden="true">
          <span className="pi-doc-welcome-workflow-tick-glint" />
        </li>
        {stages.map((stage, index) => (
          <li
            className="pi-doc-welcome-workflow-stage"
            data-wf-stage={index}
            key={stage.href}
            style={{ '--pi-workflow-index': index } as CSSProperties}
          >
            <Link className="pi-doc-welcome-workflow-card" href={docsHref(locale, stage.href)}>
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
            </Link>
          </li>
        ))}
      </ol>
    </div>
  );
}
