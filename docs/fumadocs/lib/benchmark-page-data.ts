import pageData from '@/lib/benchmark-page-data.json';
import type { Locale } from '@/lib/i18n';

export type BenchmarkDimension = {
  id: string;
  name: string;
  nameZh: string;
  kind: string;
  description: string;
  descriptionZh: string;
};

export type BenchmarkDataSource = {
  kind: string;
  id: string;
  license: string;
};

export type BenchmarkDataComposition = {
  summary: string;
  summaryZh: string;
  sizeLabel: string;
  sizeLabelZh: string;
  count: number | null;
  unit: string;
  sources: BenchmarkDataSource[];
  splits: string[];
  modalities: string[];
  license: string;
  notes: string[];
  notesZh: string[];
  notApplicable: boolean;
};

export type BenchmarkPageMetric = {
  id: string;
  name: string;
  description: string;
  descriptionZh: string;
  higherIsBetter: boolean;
  primary: boolean;
  leaderboardKey: string;
};

export type BenchmarkLeaderboardEntry = {
  rank?: number;
  model: string;
  score: string;
  metric?: string;
  note?: string;
};

export type BenchmarkLeaderboard = {
  ingested: boolean;
  source: string;
  sourceZh: string;
  officialUrl: string;
  entries: BenchmarkLeaderboardEntry[];
};

export type BenchmarkPageEntry = {
  id: string;
  name: string;
  projectPage: string;
  paperUrl: string;
  dimensions: BenchmarkDimension[];
  data: BenchmarkDataComposition;
  metrics: BenchmarkPageMetric[];
  leaderboard: BenchmarkLeaderboard;
};

type BenchmarkPageFile = {
  generatedFrom: string;
  benchmarkCount: number;
  leaderboardIngestedIds: string[];
  leaderboardPlaceholderCount: number;
  pages: Record<string, BenchmarkPageEntry>;
};

const data = pageData as BenchmarkPageFile;

export const benchmarkPageEntries = data.pages;

export const benchmarkLeaderboardStats = {
  ingestedIds: data.leaderboardIngestedIds,
  ingestedCount: data.leaderboardIngestedIds.length,
  placeholderCount: data.leaderboardPlaceholderCount,
  total: data.benchmarkCount,
};

export function getBenchmarkPageEntry(id: string): BenchmarkPageEntry | undefined {
  return data.pages[id];
}

export function localizedDimensionName(item: BenchmarkDimension, locale: Locale) {
  return locale === 'zh' ? item.nameZh || item.name : item.name;
}

export function localizedDimensionDescription(item: BenchmarkDimension, locale: Locale) {
  return locale === 'zh' ? item.descriptionZh || item.description : item.description;
}

export function localizedMetricDescription(item: BenchmarkPageMetric, locale: Locale) {
  return locale === 'zh' ? item.descriptionZh || item.description : item.description;
}

export function localizedDataSummary(entry: BenchmarkPageEntry, locale: Locale) {
  const block = entry.data;
  return locale === 'zh' ? block.summaryZh || block.summary : block.summary;
}

export function localizedSizeLabel(entry: BenchmarkPageEntry, locale: Locale) {
  const block = entry.data;
  return locale === 'zh' ? block.sizeLabelZh || block.sizeLabel : block.sizeLabel;
}
