import type { TOCItemType } from 'fumadocs-core/toc';
import type { Locale } from '@/lib/i18n';

const SECTION_TITLES: Record<Locale, Array<{ title: string; url: string }>> = {
  en: [
    { title: 'Dimensions', url: '#benchmark-dimensions' },
    { title: 'Data composition', url: '#benchmark-data' },
    { title: 'Metrics', url: '#benchmark-metrics' },
    { title: 'Leaderboard', url: '#benchmark-leaderboard' },
  ],
  zh: [
    { title: '评测维度', url: '#benchmark-dimensions' },
    { title: '数据组成', url: '#benchmark-data' },
    { title: '指标', url: '#benchmark-metrics' },
    { title: 'Leaderboard', url: '#benchmark-leaderboard' },
  ],
};

function plainTitle(title: TOCItemType['title']) {
  if (typeof title === 'string') return title;
  return '';
}

export function isBenchmarkDetailPage(slugs: readonly string[]) {
  return (
    slugs[0] === 'evaluation' &&
    slugs[1] === 'benchmark-hub' &&
    slugs.length === 3 &&
    slugs[2] !== 'runtime-environments'
  );
}

export function enrichBenchmarkToc(
  slugs: readonly string[],
  toc: TOCItemType[],
  locale: Locale,
): TOCItemType[] {
  if (!isBenchmarkDetailPage(slugs)) return toc;

  const extras: TOCItemType[] = SECTION_TITLES[locale].map((item) => ({
    title: item.title,
    url: item.url,
    depth: 2,
  }));

  const aboutIndex = toc.findIndex((item) => /^(about|简介)$/i.test(plainTitle(item.title).trim()));
  if (aboutIndex >= 0) {
    return [...toc.slice(0, aboutIndex + 1), ...extras, ...toc.slice(aboutIndex + 1)];
  }
  return [...extras, ...toc];
}
