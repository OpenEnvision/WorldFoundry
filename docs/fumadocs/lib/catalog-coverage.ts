import catalogCoverageData from '@/lib/catalog-coverage-data.json';

export type CatalogEntry = {
  id: string;
  name: string;
  status?: string | null;
  /** Variant / catalog ids collapsed into this display row. */
  aliases?: string[];
};

export type CatalogFamily = {
  id: string;
  label: string;
  blurb: string;
  count: number;
  entries: CatalogEntry[];
};

export type CatalogBenchmarkGroup = {
  id: string;
  label: string;
  count: number;
  entries: CatalogEntry[];
};

export type CatalogCoverageData = {
  modelsTotal: number;
  modelsListed?: number;
  benchmarksTotal: number;
  modelFamilies: CatalogFamily[];
  benchmarkGroups: CatalogBenchmarkGroup[];
};

export const catalogCoverage = catalogCoverageData as CatalogCoverageData;

/** Benchmark Hub pages currently missing from docs (catalog still lists them). */
export const benchmarkHubMissingIds = new Set(['worldreasonbench', 'wrbench']);
