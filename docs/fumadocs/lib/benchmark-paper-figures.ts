import { readSiteJson } from '@/lib/read-site-json';
import type { Locale } from '@/lib/i18n';

export type BenchmarkPaperFigureRole = 'teaser' | 'overview' | 'results';

export type BenchmarkPaperFigureEntry = {
  benchmarkId: string;
  /** Official paper teaser (Fig. 1 / author collage). Never a PDF first page. */
  teaserSrc?: string;
  /** Official method / task overview. Never a PDF first page. */
  src?: string;
  /** Official quantitative results / leaderboard figure. */
  resultsSrc?: string;
  alt?: string;
  altZh?: string;
  caption?: string;
  captionZh?: string;
  teaserAlt?: string;
  teaserAltZh?: string;
  teaserCaption?: string;
  teaserCaptionZh?: string;
  resultsAlt?: string;
  resultsAltZh?: string;
  resultsCaption?: string;
  resultsCaptionZh?: string;
};

export type BenchmarkPaperFigure = {
  role: BenchmarkPaperFigureRole;
  src: string;
  alt: string;
  caption: string;
};

type BenchmarkPaperFigureData = {
  version: number;
  figures: Record<string, BenchmarkPaperFigureEntry>;
};

const data = readSiteJson<BenchmarkPaperFigureData>('lib/benchmark-paper-figures.json');

export function getBenchmarkPaperFigureEntry(
  benchmarkId: string,
): BenchmarkPaperFigureEntry | undefined {
  return data.figures[benchmarkId];
}

function localizedField(
  entry: BenchmarkPaperFigureEntry,
  locale: Locale,
  field:
    | 'alt'
    | 'caption'
    | 'teaserAlt'
    | 'teaserCaption'
    | 'resultsAlt'
    | 'resultsCaption',
): string {
  const zhKey = `${field}Zh` as const;
  const localized =
    locale === 'zh' ? entry[zhKey] || entry[field] : entry[field] || entry[zhKey];
  return localized?.trim() || '';
}

export function getBenchmarkPaperFigures(
  benchmarkId: string,
  locale: Locale = 'en',
  roles?: BenchmarkPaperFigureRole[],
): BenchmarkPaperFigure[] {
  const entry = data.figures[benchmarkId];
  if (!entry) return [];
  const figures: BenchmarkPaperFigure[] = [];
  if (entry.teaserSrc) {
    const caption =
      localizedField(entry, locale, 'teaserCaption') ||
      (locale === 'zh' ? '论文官方 Teaser。' : 'Official paper teaser.');
    figures.push({
      role: 'teaser',
      src: entry.teaserSrc,
      alt: localizedField(entry, locale, 'teaserAlt') || caption,
      caption,
    });
  }
  if (entry.src) {
    const caption =
      localizedField(entry, locale, 'caption') ||
      (locale === 'zh' ? '论文主图。' : 'Official paper figure.');
    figures.push({
      role: 'overview',
      src: entry.src,
      alt: localizedField(entry, locale, 'alt') || caption,
      caption,
    });
  }
  if (entry.resultsSrc) {
    const caption =
      localizedField(entry, locale, 'resultsCaption') ||
      (locale === 'zh' ? '论文官方结果图。' : 'Official paper results figure.');
    figures.push({
      role: 'results',
      src: entry.resultsSrc,
      alt: localizedField(entry, locale, 'resultsAlt') || caption,
      caption,
    });
  }
  if (!roles || roles.length === 0) return figures;
  return figures.filter((figure) => roles.includes(figure.role));
}

export function getBenchmarkLeaderboardFigures(
  benchmarkId: string,
  locale: Locale = 'en',
): BenchmarkPaperFigure[] {
  return getBenchmarkPaperFigures(benchmarkId, locale, ['results']);
}

export function getBenchmarkEmbedFallbackFigures(
  benchmarkId: string,
  locale: Locale = 'en',
): BenchmarkPaperFigure[] {
  const results = getBenchmarkPaperFigures(benchmarkId, locale, ['results']);
  if (results.length > 0) return results;
  const overview = getBenchmarkPaperFigures(benchmarkId, locale, ['overview']);
  if (overview.length > 0) return overview;
  return getBenchmarkPaperFigures(benchmarkId, locale, ['teaser']);
}

export function getBenchmarkHeroFigures(
  benchmarkId: string,
  locale: Locale = 'en',
): BenchmarkPaperFigure[] {
  const board = getBenchmarkLeaderboardFigures(benchmarkId, locale);
  const teasers = getBenchmarkPaperFigures(benchmarkId, locale, ['teaser']);
  const boardSrc = new Set(board.map((figure) => figure.src));
  return teasers.filter((figure) => !boardSrc.has(figure.src));
}

export function hasBenchmarkPaperFigures(benchmarkId: string): boolean {
  return getBenchmarkPaperFigures(benchmarkId).length > 0;
}
