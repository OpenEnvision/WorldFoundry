import Link from 'next/link';

import { upstreamAcknowledgements } from '@/lib/upstream-acknowledgements';
import type { Locale } from '@/lib/i18n';

const copy = {
  en: {
    intro:
      'WorldFoundry integrates and adapts upstream world-model, video-generation, perception, reconstruction, embodied-action, and evaluation projects. Each catalog entry keeps its own source and license metadata.',
    infrastructure: 'Infrastructure foundations',
    footer:
      'For per-model or per-benchmark license terms, checkpoint provenance, and readiness notes, see the individual catalog entries and runtime profiles.',
    catalogModels: 'Model catalog',
    catalogBenchmarks: 'Benchmark Hub',
  },
  zh: {
    intro:
      'WorldFoundry 集成并适配了上游世界模型、视频生成、感知、重建、具身动作与评测项目。每个 catalog 条目都会保留各自的来源与许可证元数据。',
    infrastructure: '基础设施依赖',
    footer:
      '如需查看单个模型或 benchmark 的许可证、checkpoint 来源与 readiness 说明，请查阅对应 catalog 条目与 runtime profile。',
    catalogModels: '模型目录',
    catalogBenchmarks: 'Benchmark Hub',
  },
} as const satisfies Record<
  Locale,
  {
    intro: string;
    infrastructure: string;
    footer: string;
    catalogModels: string;
    catalogBenchmarks: string;
  }
>;

export function DocsWelcomeAcknowledgements({ locale }: { locale: Locale }) {
  const labels = copy[locale];
  const { infrastructure } = upstreamAcknowledgements;

  return (
    <div className="pi-doc-welcome-ack">
      <p>{labels.intro}</p>

      <h3>{labels.infrastructure}</h3>
      <ul className="pi-doc-welcome-ack-infra">
        {infrastructure.map((entry) => (
          <li key={entry.url}>
            <a href={entry.url} rel="noreferrer" target="_blank">
              {entry.name}
            </a>
            <span>{locale === 'zh' ? entry.summary_zh : entry.summary}</span>
          </li>
        ))}
      </ul>

      <p className="pi-doc-welcome-ack-footer">
        {labels.footer}{' '}
        <Link href={locale === 'zh' ? '/zh/docs/guides/supported-models' : '/docs/guides/supported-models'}>
          {labels.catalogModels}
        </Link>
        {' · '}
        <Link href={locale === 'zh' ? '/zh/docs/evaluation/benchmark-hub' : '/docs/evaluation/benchmark-hub'}>
          {labels.catalogBenchmarks}
        </Link>
      </p>
    </div>
  );
}
