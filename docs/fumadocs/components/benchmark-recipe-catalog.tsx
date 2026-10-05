'use client';

import Link from 'next/link';
import { ArrowRight, Search } from 'lucide-react';
import { type CSSProperties, useDeferredValue, useMemo, useState } from 'react';

import {
  benchmarkCatalogEntries,
  type BenchmarkCatalogItem,
} from '@/lib/benchmark-catalog';
import { getBenchmarkPageEntry } from '@/lib/benchmark-page-data';
import { BenchmarkIdentityMark } from '@/components/benchmark-identity-mark';

type Locale = 'en' | 'zh';
type BenchmarkCategory = 'Embodied AI' | 'Video Generation' | 'World Models';
type StatusGroup = 'verified' | 'integrated' | 'normalizer' | 'planned' | 'blocked';
type StatusFilter = 'all' | StatusGroup;
type JudgeFilter = 'all' | 'local' | 'hosted';
type SortMode = 'readiness' | 'name';

const PAGE_SIZE = 48;
const CATEGORY_ORDER: BenchmarkCategory[] = ['Embodied AI', 'Video Generation', 'World Models'];
const CATEGORY_COUNTS = CATEGORY_ORDER.map((id) => ({
  id,
  count: benchmarkCatalogEntries.filter((entry) => entry.category === id).length,
}));
const STATUS_RANK: Record<StatusGroup, number> = {
  verified: 0,
  integrated: 1,
  normalizer: 2,
  planned: 3,
  blocked: 4,
};

const copy = {
  en: {
    coverage: 'catalog data from repository manifests',
    search: 'Search by benchmark, metric, domain, or alias…',
    families: 'Benchmark families',
    all: 'All',
    results: 'benchmarks',
    status: 'Status',
    allStatuses: 'All statuses',
    verified: 'Verified evidence',
    integrated: 'Integrated',
    normalizer: 'Normalizer',
    planned: 'Planned',
    blocked: 'Blocked',
    judge: 'Judge',
    allJudges: 'All judge modes',
    local: 'Local / no API',
    hosted: 'Hosted API',
    sort: 'Sort',
    readiness: 'Readiness',
    name: 'Name A–Z',
    benchmark: 'Benchmark',
    metrics: 'Primary metrics',
    runtime: 'Execution',
    verification: 'Verification',
    service: 'Judge',
    notRecorded: 'Not recorded',
    pending: 'Pending',
    none: 'No benchmarks matched these filters.',
    more: 'Show more benchmarks',
    axes: 'dimensions',
    metricsCount: 'metrics',
    noBoard: 'No official board',
    boardIngested: 'Official results',
    categories: {
      'Embodied AI': 'Embodied AI',
      'Video Generation': 'Video Generation',
      'World Models': 'World Models',
    },
  },
  zh: {
    coverage: '目录数据来自仓库 manifest',
    search: '按 benchmark、指标、领域或别名搜索…',
    families: 'Benchmark 分类',
    all: '全部',
    results: '个 benchmark',
    status: '状态',
    allStatuses: '全部状态',
    verified: '已有验证证据',
    integrated: '已接入',
    normalizer: '仅归一化',
    planned: '规划中',
    blocked: '阻塞',
    judge: '评审方式',
    allJudges: '全部评审方式',
    local: '本地 / 无 API',
    hosted: '托管 API',
    sort: '排序',
    readiness: '按就绪度',
    name: '按名称 A–Z',
    benchmark: 'Benchmark',
    metrics: '主要指标',
    runtime: '执行方式',
    verification: '验证状态',
    service: 'Judge',
    notRecorded: '未记录',
    pending: '待验证',
    none: '没有符合当前筛选条件的 benchmark。',
    more: '显示更多 benchmark',
    axes: '个维度',
    metricsCount: '项指标',
    noBoard: '无官方榜',
    boardIngested: '已收录成绩',
    categories: {
      'Embodied AI': '具身 AI',
      'Video Generation': '视频生成',
      'World Models': '世界模型',
    },
  },
} as const;

function statusGroup(entry: BenchmarkCatalogItem): StatusGroup {
  if (entry.badges.includes('blocked')) return 'blocked';
  if (entry.badges.includes('planned')) return 'planned';
  if (entry.badges.includes('normalizer')) return 'normalizer';
  if (entry.verificationStatus.includes('verified')) return 'verified';
  return 'integrated';
}

function includesQuery(entry: BenchmarkCatalogItem, query: string) {
  if (!query) return true;
  const page = getBenchmarkPageEntry(entry.id);
  const haystack = [
    entry.id,
    entry.name,
    entry.category,
    entry.categoryZh,
    entry.summary,
    entry.summaryZh,
    entry.integrationStatus,
    entry.verificationStatus,
    entry.runtimeKind,
    ...entry.aliases,
    ...entry.domains,
    ...entry.metrics,
    ...(page?.dimensions.flatMap((item) => [item.id, item.name, item.nameZh]) ?? []),
    page?.data.summary,
    page?.data.summaryZh,
    page?.data.sizeLabel,
  ]
    .join(' ')
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

function benchmarkHref(entry: BenchmarkCatalogItem, locale: Locale) {
  const prefix = locale === 'zh' ? '/zh' : '';
  return `${prefix}/docs/evaluation/benchmark-hub/${entry.id}`;
}

function humanize(value: string) {
  if (!value) return '';
  return value
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function runtimeLabel(entry: BenchmarkCatalogItem, locale: Locale) {
  const value = entry.runtimeKind;
  if (!value) return copy[locale].notRecorded;
  if (value === 'native_closed_loop_simulator') return locale === 'zh' ? '闭环仿真' : 'Closed-loop sim';
  if (value === 'external_official_results_runner') return locale === 'zh' ? '官方结果导入' : 'Official results';
  if (value.includes('judge')) return locale === 'zh' ? 'Judge runtime' : 'Judge runtime';
  if (value.includes('importer')) return locale === 'zh' ? 'Artifact 导入' : 'Artifact import';
  if (value.includes('official')) return locale === 'zh' ? '官方 runtime' : 'Official runtime';
  return humanize(value);
}

function rowFacts(entry: BenchmarkCatalogItem, locale: Locale, status: StatusGroup) {
  return [runtimeLabel(entry, locale), copy[locale][status]].filter(Boolean).slice(0, 2);
}

export function BenchmarkRecipeCatalog({ locale = 'en' }: { locale?: Locale }) {
  const t = copy[locale];
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState<'all' | BenchmarkCategory>('all');
  const [status, setStatus] = useState<StatusFilter>('all');
  const [judge, setJudge] = useState<JudgeFilter>('all');
  const [sort, setSort] = useState<SortMode>('readiness');
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const deferredQuery = useDeferredValue(query.trim());

  const results = useMemo(() => {
    const filtered = benchmarkCatalogEntries.filter((entry) => {
      if (category !== 'all' && entry.category !== category) return false;
      if (status !== 'all' && statusGroup(entry) !== status) return false;
      if (judge === 'hosted' && !entry.requiresApi) return false;
      if (judge === 'local' && entry.requiresApi) return false;
      return includesQuery(entry, deferredQuery);
    });

    return filtered.sort((left, right) => {
      if (sort === 'name') return left.name.localeCompare(right.name);
      const statusDifference = STATUS_RANK[statusGroup(left)] - STATUS_RANK[statusGroup(right)];
      return statusDifference || left.name.localeCompare(right.name);
    });
  }, [category, deferredQuery, judge, sort, status]);

  const visibleResults = results.slice(0, visibleCount);
  const resetVisible = () => setVisibleCount(PAGE_SIZE);

  return (
    <div className="wf-recipe-catalog wf-benchmark-catalog">
      <div className="wf-recipe-catalog-coverage">
        <span>{benchmarkCatalogEntries.length} benchmarks</span>
        <span aria-hidden="true">·</span>
        <span>{t.coverage}</span>
      </div>

      <label className="wf-recipe-search">
        <Search aria-hidden="true" size={20} strokeWidth={1.6} />
        <span className="sr-only">{t.search}</span>
        <input
          type="search"
          value={query}
          placeholder={t.search}
          autoComplete="off"
          onChange={(event) => {
            setQuery(event.target.value);
            resetVisible();
          }}
        />
      </label>

      <div className="wf-recipe-family-tabs" role="tablist" aria-label={t.families}>
        <button
          type="button"
          role="tab"
          aria-selected={category === 'all'}
          className={category === 'all' ? 'is-active' : undefined}
          onClick={() => {
            setCategory('all');
            resetVisible();
          }}
        >
          {t.all}
          <span>{benchmarkCatalogEntries.length}</span>
        </button>
        {CATEGORY_COUNTS.map((item) => (
          <button
            type="button"
            role="tab"
            aria-selected={category === item.id}
            className={category === item.id ? 'is-active' : undefined}
            key={item.id}
            onClick={() => {
              setCategory(item.id);
              resetVisible();
            }}
          >
            {t.categories[item.id]}
            <span>{item.count}</span>
          </button>
        ))}
      </div>

      <div className="wf-recipe-catalog-toolbar">
        <p aria-live="polite">
          <strong>{results.length}</strong> {t.results}
        </p>
        <div>
          <label>
            <span>{t.status}</span>
            <select
              value={status}
              onChange={(event) => {
                setStatus(event.target.value as StatusFilter);
                resetVisible();
              }}
            >
              <option value="all">{t.allStatuses}</option>
              <option value="verified">{t.verified}</option>
              <option value="integrated">{t.integrated}</option>
              <option value="normalizer">{t.normalizer}</option>
              <option value="planned">{t.planned}</option>
              <option value="blocked">{t.blocked}</option>
            </select>
          </label>
          <label>
            <span>{t.judge}</span>
            <select
              value={judge}
              onChange={(event) => {
                setJudge(event.target.value as JudgeFilter);
                resetVisible();
              }}
            >
              <option value="all">{t.allJudges}</option>
              <option value="local">{t.local}</option>
              <option value="hosted">{t.hosted}</option>
            </select>
          </label>
          <label>
            <span>{t.sort}</span>
            <select value={sort} onChange={(event) => setSort(event.target.value as SortMode)}>
              <option value="readiness">{t.readiness}</option>
              <option value="name">{t.name}</option>
            </select>
          </label>
        </div>
      </div>

      <div className="wf-recipe-table wf-benchmark-table wf-recipe-list" role="list" aria-label="Benchmark recipes">
        {visibleResults.length > 0 ? (
          visibleResults.map((entry, index) => {
            const entryStatus = statusGroup(entry);
            const facts = rowFacts(entry, locale, entryStatus);
            return (
              <Link
                className="wf-recipe-row wf-benchmark-row"
                href={benchmarkHref(entry, locale)}
                role="listitem"
                key={entry.id}
                style={{ '--wf-row-index': Math.min(index, 12) } as CSSProperties}
              >
                <span className="wf-recipe-row-model">
                  <BenchmarkIdentityMark
                    id={entry.id}
                    name={entry.name}
                    category={entry.category}
                    logoKey={entry.logoKey}
                    size="medium"
                  />
                  <span className="wf-recipe-row-identity">
                    <span>
                      <strong>{entry.name}</strong>
                      <em>{entry.id}</em>
                    </span>
                    <small>{locale === 'zh' ? entry.summaryZh : entry.summary}</small>
                    {facts.length > 0 ? (
                      <span className="wf-recipe-row-facts">
                        {facts.map((fact) => (
                          <span key={fact}>{fact}</span>
                        ))}
                      </span>
                    ) : null}
                  </span>
                </span>
                <ArrowRight aria-hidden="true" size={17} strokeWidth={1.5} />
              </Link>
            );
          })
        ) : (
          <p className="wf-recipe-empty">{t.none}</p>
        )}
      </div>

      {visibleCount < results.length ? (
        <button
          className="wf-recipe-show-more"
          type="button"
          onClick={() => setVisibleCount((count) => count + PAGE_SIZE)}
        >
          {t.more}
          <span>{Math.min(PAGE_SIZE, results.length - visibleCount)}</span>
        </button>
      ) : null}
    </div>
  );
}
