import architectureMetaEn from '@/content/docs/maintainers/architecture/meta.json';
import architectureMetaZh from '@/content/docs/maintainers/architecture/meta.zh.json';
import benchmarkHubMeta from '@/content/docs/evaluation/benchmark-hub/meta.json';
import metricsMetaEn from '@/content/docs/evaluation/metrics/meta.json';
import metricsMetaZh from '@/content/docs/evaluation/metrics/meta.zh.json';
import type { Locale } from '@/lib/i18n';
import { source } from '@/lib/source';
import { docsNavGroups, getNavPageLabel } from '@/lib/docs-navigation';

type DocsPage = NonNullable<ReturnType<typeof source.getPage>>;

export type DocsPaginationLink = {
  url: string;
  label: string;
};

function isMetaSeparator(entry: string) {
  return entry.startsWith('---') && entry.endsWith('---');
}

function getBenchmarkHubChildSlugs(): readonly string[][] {
  return benchmarkHubMeta.pages
    .filter((entry) => entry !== 'index' && !isMetaSeparator(entry))
    .map((entry) => ['evaluation', 'benchmark-hub', entry]);
}

function getMetricsMeta(locale: Locale) {
  return locale === 'zh' ? metricsMetaZh : metricsMetaEn;
}

function getMetricsChildSlugs(locale: Locale): readonly string[][] {
  return getMetricsMeta(locale).pages
    .filter((entry) => entry !== 'index' && !isMetaSeparator(entry))
    .map((entry) => ['evaluation', 'metrics', entry]);
}

function getArchitectureMeta(locale: Locale) {
  return locale === 'zh' ? architectureMetaZh : architectureMetaEn;
}

function getArchitectureChildSlugs(locale: Locale): readonly string[][] {
  return getArchitectureMeta(locale).pages
    .filter((entry) => entry !== 'index' && !isMetaSeparator(entry))
    .map((entry) => ['maintainers', 'architecture', entry]);
}

export function getDocsPaginationPages(locale: Locale): DocsPage[] {
  const pages: DocsPage[] = [];
  const seen = new Set<string>();

  const appendPage = (slugs: readonly string[]) => {
    const key = slugs.join('/');
    if (seen.has(key)) return;
    const page = source.getPage([...slugs], locale);
    if (!page) return;
    seen.add(key);
    pages.push(page);
  };

  for (const group of docsNavGroups) {
    for (const slugs of group.slugs) {
      appendPage(slugs);

      if (slugs.join('/') === 'evaluation/benchmark-hub') {
        for (const childSlugs of getBenchmarkHubChildSlugs()) {
          appendPage(childSlugs);
        }
      }

      if (slugs.join('/') === 'evaluation/metrics') {
        for (const childSlugs of getMetricsChildSlugs(locale)) {
          appendPage(childSlugs);
        }
      }

      if (slugs.join('/') === 'maintainers/architecture') {
        for (const childSlugs of getArchitectureChildSlugs(locale)) {
          appendPage(childSlugs);
        }
      }
    }
  }

  return pages;
}

function toPaginationLink(page: DocsPage, locale: Locale): DocsPaginationLink {
  return {
    url: page.url,
    label: getNavPageLabel(page.slugs, locale, page.data.title),
  };
}

export function getDocsPagination(currentSlugs: readonly string[], locale: Locale) {
  const pages = getDocsPaginationPages(locale);
  const currentKey = currentSlugs.join('/');
  const index = pages.findIndex((page) => page.slugs.join('/') === currentKey);

  if (index === -1) {
    return { prev: undefined, next: undefined };
  }

  return {
    prev: index > 0 ? toPaginationLink(pages[index - 1], locale) : undefined,
    next: index < pages.length - 1 ? toPaginationLink(pages[index + 1], locale) : undefined,
  };
}
