import catalogStatus from '@/lib/benchmark-catalog-status.json';
import type { Locale } from '@/lib/i18n';

export type BenchmarkBadgeKind = 'integrated' | 'normalizer' | 'planned' | 'blocked';

export type BenchmarkCatalogEntry = {
  name: string;
  aliases: string[];
  domains: string[];
  metrics: string[];
  summary: string;
  summaryZh: string;
  integrationStatus: string;
  verificationStatus: string;
  runtimeKind: string;
  requiresApi: boolean;
  leaderboardValid: boolean;
  badges: BenchmarkBadgeKind[];
  category: string;
  categoryZh: string;
  logoKey?: string;
};

const catalog = catalogStatus as Record<string, BenchmarkCatalogEntry>;

export type BenchmarkCatalogItem = BenchmarkCatalogEntry & { id: string };

export const benchmarkCatalogEntries: BenchmarkCatalogItem[] = Object.entries(catalog).map(
  ([id, entry]) => ({ id, ...entry }),
);

export function getBenchmarkCatalogEntry(benchmarkId: string): BenchmarkCatalogEntry | undefined {
  return catalog[benchmarkId];
}

export function getBenchmarkBadges(benchmarkId: string): BenchmarkBadgeKind[] {
  return catalog[benchmarkId]?.badges ?? [];
}

export const benchmarkBadgeLabels: Record<Locale, Record<BenchmarkBadgeKind, string>> = {
  en: {
    integrated: 'Integrated',
    normalizer: 'Normalizer',
    planned: 'Planned',
    blocked: 'Blocked',
  },
  zh: {
    integrated: '已接入',
    normalizer: '归一化',
    planned: '规划中',
    blocked: '阻塞',
  },
};

export function getBenchmarkBadgeLabel(kind: BenchmarkBadgeKind, locale: Locale) {
  return benchmarkBadgeLabels[locale][kind];
}

export const benchmarkHubSectionLabels: Record<string, Record<Locale, string>> = {
  'Embodied AI': { en: 'Embodied AI', zh: '具身 AI' },
  'Video Generation': { en: 'Video Generation', zh: '视频生成' },
  'World Models': { en: 'World Models', zh: '世界模型' },
};

export function getBenchmarkHubSectionLabel(section: string, locale: Locale) {
  return benchmarkHubSectionLabels[section]?.[locale] ?? section;
}
