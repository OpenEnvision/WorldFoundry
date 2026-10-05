import { getNavPageLabel } from '@/lib/docs-navigation';
import type { Locale } from '@/lib/i18n';
import { source } from '@/lib/source';

export type DocsRelatedLink = {
  url: string;
  label: string;
};

const BENCHMARK_HUB_RELATED_SLUGS = [
  ['evaluation'],
  ['evaluation', 'benchmark-hub'],
  ['evaluation', 'metrics'],
  ['evaluation', 'metrics', 'scorers'],
  ['reference', 'environments'],
  ['reference', 'cli'],
  ['guides', 'local-assets'],
] as const;

export function isBenchmarkHubSection(slugs: readonly string[]) {
  return slugs[0] === 'evaluation' && slugs[1] === 'benchmark-hub';
}

export function getDocsRelatedLinks(slugs: readonly string[], locale: Locale): DocsRelatedLink[] {
  if (!isBenchmarkHubSection(slugs)) return [];

  const currentKey = slugs.join('/');

  return BENCHMARK_HUB_RELATED_SLUGS.flatMap((entrySlugs) => {
    const key = entrySlugs.join('/');
    if (key === currentKey) return [];

    const page = source.getPage([...entrySlugs], locale);
    if (!page) return [];

    return [
      {
        url: page.url,
        label: getNavPageLabel(page.slugs, locale, page.data.title),
      },
    ];
  });
}
