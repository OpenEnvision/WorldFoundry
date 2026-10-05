import type { ReactNode } from 'react';

import {
  metricDocsPath,
  metricQuickNavGroups,
  metricQuickNavGroupsForPage,
  metricQuickNavLabels,
  metricQuickNavPageMeta,
  type MetricQuickNavPageId,
} from '@/lib/metric-quick-nav';
import type { Locale } from '@/lib/i18n';

type MetricQuickNavProps = {
  locale: Locale;
  page?: MetricQuickNavPageId;
  variant?: 'page' | 'hub';
  placement?: 'inline' | 'rail';
};

export function MetricQuickNav({
  locale,
  page,
  variant = page ? 'page' : 'hub',
  placement = 'inline',
}: MetricQuickNavProps) {
  const labels = metricQuickNavLabels[locale];
  const isRail = placement === 'rail';

  if (!isRail) {
    return null;
  }

  if (variant === 'hub') {
    const referencePath = metricDocsPath(locale, '/evaluation/metrics/reference');
    const title = labels.hubTitle;
    const body = (
      <div className="pi-metric-quick-nav-body">
        <div className="pi-metric-quick-nav-sections">
          <span className="pi-metric-quick-nav-section-item">
            <a href={referencePath}>
              {locale === 'zh' ? 'Registry 与参考' : 'Registry & reference'}
            </a>
          </span>
        </div>

        <div className="pi-metric-quick-nav-groups">
          {(Object.keys(metricQuickNavPageMeta) as MetricQuickNavPageId[]).map((pageId) => {
            const meta = metricQuickNavPageMeta[pageId];
            const groups = metricQuickNavGroupsForPage(pageId);
            const count = groups.reduce((sum, group) => sum + group.items.length, 0);

            return (
              <section className="pi-metric-quick-nav-group" key={pageId}>
                <h3 className="pi-metric-quick-nav-group-title">
                  <a href={metricDocsPath(locale, meta.slug)}>{meta.label[locale]}</a>
                </h3>
                <p className="pi-metric-quick-nav-group-meta">
                  {count} {locale === 'zh' ? '个指标' : 'metrics'}
                </p>
              </section>
            );
          })}
        </div>
      </div>
    );

    return wrapNav({
      isRail,
      title,
      count: (Object.keys(metricQuickNavPageMeta) as MetricQuickNavPageId[]).length,
      countLabel: labels.summary,
      locale,
      body,
      unit: locale === 'zh' ? '个页面' : 'pages',
    });
  }

  const groups = page ? metricQuickNavGroupsForPage(page) : metricQuickNavGroups;
  const metricCount = groups.reduce((count, group) => count + group.items.length, 0);
  const basePath = page ? metricDocsPath(locale, metricQuickNavPageMeta[page].slug) : '';
  const title = labels.title;
  const body = (
    <div className="pi-metric-quick-nav-body">
      <div className="pi-metric-quick-nav-sections">
        <span className="pi-metric-quick-nav-section-item">
          <a href={metricDocsPath(locale, '/evaluation/metrics')}>
            {locale === 'zh' ? '指标概览' : 'Metrics overview'}
          </a>
        </span>
      </div>

      <div className="pi-metric-quick-nav-groups">
        {groups.map((group) => (
          <section className="pi-metric-quick-nav-group" key={group.id}>
            <h3 className="pi-metric-quick-nav-group-title">{group.label[locale]}</h3>
            <ul className="pi-metric-quick-nav-list">
              {group.items.map((item) => (
                <li key={item.id[locale]}>
                  <a href={`${basePath}#${item.id[locale]}`}>{item.label[locale]}</a>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );

  return wrapNav({ isRail, title, count: metricCount, countLabel: labels.summary, locale, body });
}

function wrapNav({
  title,
  count,
  locale,
  body,
  unit,
}: {
  isRail: boolean;
  title: string;
  count: number;
  countLabel: string;
  locale: Locale;
  body: ReactNode;
  unit?: string;
}) {
  const meta = `${count} ${unit ?? (locale === 'zh' ? '个 id' : 'ids')}`;

  return (
    <nav className="pi-metric-quick-nav is-rail" id="metric-quick-nav" aria-label={title}>
      <span className="pi-metric-quick-nav-title">
        {title}
        <small>{meta}</small>
      </span>
      {body}
    </nav>
  );
}
