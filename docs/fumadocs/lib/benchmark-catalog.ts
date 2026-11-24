import catalogStatus from '@/lib/benchmark-catalog-status.json';

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

export const benchmarkCatalogEntries: BenchmarkCatalogItem[] = Object.entries(catalog)
  .map(([id, entry]) => ({ id, ...entry }));

export function getBenchmarkCatalogEntry(benchmarkId: string): BenchmarkCatalogEntry | undefined {
  return catalog[benchmarkId];
}
