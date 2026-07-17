import type { Locale } from '@/lib/i18n';

export type DocsNavGroupId =
  | 'api'
  | 'overview'
  | 'start'
  | 'inference'
  | 'training'
  | 'evaluation'
  | 'integration'
  | 'tools'
  | 'architecture'
  | 'maintainers';

export type DocsSlug = readonly string[];

export type DocsNavGroup = {
  id: DocsNavGroupId;
  slugs: readonly DocsSlug[];
};

export type DocsChromeLabels = {
  docs: string;
  home: string;
  kicker: string;
  language: string;
  markdown: string;
  nav: string;
  navGroups: Record<DocsNavGroupId, string>;
  openEnvision: string;
  onThisPage: string;
  previousPage: string;
  nextPage: string;
  relatedPages: string;
  expandMetricsList: string;
  collapseMetricsList: string;
  expandArchitectureList: string;
  collapseArchitectureList: string;
  expandApiList: string;
  collapseApiList: string;
  sidebar: string;
  editPage: string;
  source: string;
  lastUpdated: string;
  openMenu: string;
  closeMenu: string;
};

/** Sidebar labels keyed by slug path (e.g. `guides/inference`). Falls back to page title. */
export type DocsNavPageLabels = Partial<Record<string, string>>;

export const docsNavPageLabels: Record<Locale, DocsNavPageLabels> = {
  en: {
    'api-reference': 'API overview',
    'api-reference/core': 'Core API guide',
    'api-reference/core-attention': 'Core: attention',
    'api-reference/core-configuration': 'Core: configuration',
    'api-reference/core-io-media': 'Core: I/O & media',
    'api-reference/core-model-loading': 'Core: model loading',
    'api-reference/core-distributed': 'Core: distributed',
    'api-reference/core-runtime': 'Core: inference runtime',
    'api-reference/core-nn-math': 'Core: neural net & math',
    'api-reference/core-acceleration-memory': 'Core: acceleration & memory',
    'api-reference/core-foundations': 'Core: foundations',
    'api-reference/contracts': 'Contracts & artifacts',
    'api-reference/models': 'Models & runners',
    'api-reference/metrics-tasks': 'Metrics & tasks',
    'api-reference/runs': 'Runs & benchmarks',
    'api-reference/reporting': 'Reporting',
    'api-reference/runtime': 'Runtime & assets',
    '': 'Introduction',
    'overview/design': 'Design',
    'overview/capabilities': "What's included",
    'overview/why-worldfoundry': 'Why WorldFoundry',
    'reference/environments': 'Environment',
    'guides/inference': 'Run inference',
    'guides/supported-models': 'Models',
    'guides/local-assets': 'Local assets',
    'guides/tui': 'TUI',
    'reference/cli': 'CLI',
    'guides/studio': 'Studio',
    evaluation: 'Overview',
    'evaluation/benchmark-hub': 'Benchmark Hub',
    'evaluation/benchmark-hub/runtime-environments': 'Runtime environments',
    'evaluation/metrics': 'Metrics',
    'evaluation/metrics/scorers': 'Scorers & quality',
    'evaluation/metrics/distribution': 'Distribution',
    'evaluation/metrics/perceptual': 'Perceptual',
    'evaluation/metrics/editing': 'Editing',
    'evaluation/metrics/reference': 'Registry',
    'evaluation/embodied-official-runtime': 'Embodied setup',
    'maintainers/plan': 'Plan',
  },
  zh: {
    'api-reference': 'API 概览',
    'api-reference/core': 'Core API 指南',
    'api-reference/core-attention': 'Core：注意力',
    'api-reference/core-configuration': 'Core：配置',
    'api-reference/core-io-media': 'Core：I/O 与媒体',
    'api-reference/core-model-loading': 'Core：模型加载',
    'api-reference/core-distributed': 'Core：分布式',
    'api-reference/core-runtime': 'Core：推理 Runtime',
    'api-reference/core-nn-math': 'Core：神经网络与数学',
    'api-reference/core-acceleration-memory': 'Core：加速与内存',
    'api-reference/core-foundations': 'Core：基础能力',
    'api-reference/contracts': '契约与 artifact',
    'api-reference/models': '模型与 runner',
    'api-reference/metrics-tasks': 'Metric 与 task',
    'api-reference/runs': 'Run 与 benchmark',
    'api-reference/reporting': '报告与证据',
    'api-reference/runtime': 'Runtime 与资产',
    '': '简介',
    'overview/design': '设计',
    'overview/capabilities': '包含什么',
    'overview/why-worldfoundry': '为什么使用',
    'reference/environments': '环境配置',
    'guides/inference': '运行推理',
    'guides/supported-models': '模型',
    'guides/local-assets': '本地资产',
    'guides/tui': 'TUI',
    'reference/cli': 'CLI',
    'guides/studio': 'Studio',
    evaluation: '概览',
    'evaluation/benchmark-hub': 'Benchmark Hub',
    'evaluation/benchmark-hub/runtime-environments': '运行环境矩阵',
    'evaluation/metrics': '指标',
    'evaluation/metrics/scorers': 'Scorer 与质量',
    'evaluation/metrics/distribution': '分布指标',
    'evaluation/metrics/perceptual': '感知成对',
    'evaluation/metrics/editing': '编辑',
    'evaluation/metrics/reference': 'Registry',
    'evaluation/embodied-official-runtime': 'Embodied 环境',
    'maintainers/plan': '规划',
  },
};

export function getNavPageLabel(slugs: readonly string[], locale: Locale, fallback: string) {
  const key = slugs.join('/');
  return docsNavPageLabels[locale][key] ?? fallback;
}

export const docsNavGroups = [
  {
    id: 'overview',
    slugs: [
      [],
      ['overview', 'design'],
      ['overview', 'capabilities'],
      ['overview', 'why-worldfoundry'],
    ],
  },
  {
    id: 'start',
    slugs: [
      ['quickstart'],
      ['reference', 'environments'],
      ['guides', 'local-assets'],
      ['guides', 'tui'],
      ['reference', 'cli'],
    ],
  },
  {
    id: 'inference',
    slugs: [
      ['guides', 'inference'],
      ['guides', 'supported-models'],
      ['guides', 'studio'],
    ],
  },

  {
    id: 'evaluation',
    slugs: [
      ['evaluation'],
      ['evaluation', 'benchmark-hub'],
      ['evaluation', 'benchmark-hub', 'runtime-environments'],
      ['evaluation', 'metrics'],
      ['evaluation', 'embodied-official-runtime'],
    ],
  },
  {
    id: 'api',
    slugs: [
      ['api-reference'],
      ['api-reference', 'core'],
      ['api-reference', 'core-attention'],
      ['api-reference', 'core-configuration'],
      ['api-reference', 'core-io-media'],
      ['api-reference', 'core-model-loading'],
      ['api-reference', 'core-distributed'],
      ['api-reference', 'core-runtime'],
      ['api-reference', 'core-nn-math'],
      ['api-reference', 'core-acceleration-memory'],
      ['api-reference', 'core-foundations'],
      ['api-reference', 'contracts'],
      ['api-reference', 'models'],
      ['api-reference', 'metrics-tasks'],
      ['api-reference', 'runs'],
      ['api-reference', 'reporting'],
      ['api-reference', 'runtime'],
    ],
  },
  {
    id: 'integration',
    slugs: [
      ['guides', 'add-model'],
      ['guides', 'add-benchmark'],
    ],
  },
  {
    id: 'tools',
    slugs: [],
  },
  {
    id: 'architecture',
    slugs: [['maintainers', 'architecture']],
  },
  {
    id: 'maintainers',
    slugs: [
      ['maintainers', 'contributing'],
      ['maintainers', 'plan'],
    ],
  },
] as const satisfies readonly DocsNavGroup[];

export const docsLabels: Record<Locale, DocsChromeLabels> = {
  en: {
    docs: 'Docs',
    home: 'Home',
    kicker: 'WorldFoundry docs',
    language: 'Language',
    markdown: 'Markdown',
    nav: 'Main navigation',
    navGroups: {
      api: 'API Reference',
      architecture: 'Architecture',
      evaluation: 'Evaluation',
      inference: 'Inference',
      integration: 'Integration',
      maintainers: 'Maintainers',
      overview: 'Understand',
      start: 'Get started',
      tools: 'Tools',
      training: 'Training',
    },
    openEnvision: 'OpenEnvision',
    onThisPage: 'On this page',
    previousPage: 'Previous',
    nextPage: 'Next',
    relatedPages: 'Related',
    expandMetricsList: 'Expand metrics list',
    collapseMetricsList: 'Collapse metrics list',
    expandArchitectureList: 'Expand architecture list',
    collapseArchitectureList: 'Collapse architecture list',
    expandApiList: 'Expand API reference list',
    collapseApiList: 'Collapse API reference list',
    sidebar: 'Documentation',
    editPage: 'Edit this page',
    source: 'Source',
    lastUpdated: 'Last updated',
    openMenu: 'Menu',
    closeMenu: 'Close',
  },
  zh: {
    docs: '文档',
    home: '首页',
    kicker: 'WorldFoundry 文档',
    language: '语言',
    markdown: 'Markdown',
    nav: '主导航',
    navGroups: {
      api: 'API 参考',
      architecture: '架构',
      evaluation: '评测',
      inference: '推理',
      integration: '接入',
      maintainers: '维护者',
      overview: '理解项目',
      start: '开始使用',
      tools: '工具',
      training: '训练',
    },
    openEnvision: 'OpenEnvision',
    onThisPage: '本页内容',
    previousPage: '上一页',
    nextPage: '下一页',
    relatedPages: '相关页面',
    expandMetricsList: '展开指标列表',
    collapseMetricsList: '收起指标列表',
    expandArchitectureList: '展开架构列表',
    collapseArchitectureList: '收起架构列表',
    expandApiList: '展开 API 参考列表',
    collapseApiList: '收起 API 参考列表',
    sidebar: '文档',
    editPage: '编辑此页',
    source: '源码',
    lastUpdated: '最后更新',
    openMenu: '菜单',
    closeMenu: '关闭',
  },
};

const tableDenseDocsPages = new Set([
  'evaluation/benchmark-hub',
  'guides/supported-models',
]);

export function isTableDenseDocsPage(slugs: readonly string[]) {
  return tableDenseDocsPages.has(slugs.join('/'));
}

export function isBenchmarkHubDocsPage(slugs: readonly string[]) {
  return slugs[0] === 'evaluation' && slugs[1] === 'benchmark-hub';
}

export function isMetricsDocsPage(slugs: readonly string[]) {
  return slugs[0] === 'evaluation' && slugs[1] === 'metrics';
}

export { getBenchmarkHubSectionLabel } from '@/lib/benchmark-catalog';
