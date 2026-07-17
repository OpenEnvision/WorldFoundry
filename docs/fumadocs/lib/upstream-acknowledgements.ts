import upstreamAcknowledgementsData from '@/lib/upstream-acknowledgements-data.json';

export type UpstreamEntry = {
  name: string;
  url: string;
  repo?: string;
};

export type UpstreamFamily = {
  id: string;
  label: string;
  labelZh: string;
  count: number;
  entries: UpstreamEntry[];
};

export type UpstreamInfrastructure = {
  name: string;
  url: string;
  summary: string;
  summary_zh: string;
};

export type UpstreamAcknowledgementsData = {
  modelsTotal: number;
  benchmarksTotal: number;
  infrastructure: UpstreamInfrastructure[];
  modelFamilies: UpstreamFamily[];
  benchmarkGroups: UpstreamFamily[];
};

export const upstreamAcknowledgements =
  upstreamAcknowledgementsData as UpstreamAcknowledgementsData;
