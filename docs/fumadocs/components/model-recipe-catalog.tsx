'use client';

import Link from 'next/link';
import { ArrowRight, Search } from 'lucide-react';
import { type CSSProperties, useDeferredValue, useMemo, useState } from 'react';

import { ModelIdentityMark } from '@/components/model-identity-mark';
import { modelRecipeIndex } from '@/lib/model-recipe-index';
import { formatCudaLabel } from '@/lib/runtime-labels';
import type {
  ModelRecipeIndexEntry,
  ModelRecipeStatusGroup,
} from '@/lib/model-recipe-types';

type Locale = 'en' | 'zh';
type RuntimeFilter = 'all' | 'dedicated' | 'unified' | 'unrecorded';
type SortMode = 'readiness' | 'name';

const PAGE_SIZE = 60;
const STATUS_RANK: Record<ModelRecipeStatusGroup, number> = {
  verified: 0,
  integrated: 1,
  runtime_ported: 2,
  profile: 3,
  planned: 4,
  blocked: 5,
};

const copy = {
  en: {
    coverage: 'runtime data from repository manifests',
    search: 'Search by model, task, provider, or alias…',
    all: 'All',
    results: 'recipes',
    status: 'Status',
    allStatuses: 'All statuses',
    runtime: 'Runtime',
    allRuntimes: 'All runtimes',
    dedicated: 'Dedicated environment',
    unified: 'Unified environment',
    unrecorded: 'Runtime not recorded',
    sort: 'Sort',
    readiness: 'Readiness',
    name: 'Name A–Z',
    model: 'Model',
    tasks: 'Tasks',
    python: 'Python',
    cuda: 'CUDA',
    none: 'No recipes matched these filters.',
    more: 'Show more recipes',
  },
  zh: {
    coverage: '运行时数据来自仓库 manifest',
    search: '按模型、任务、提供方或别名搜索…',
    all: '全部',
    results: '个配方',
    status: '状态',
    allStatuses: '全部状态',
    runtime: '运行时',
    allRuntimes: '全部运行时',
    dedicated: '独立环境',
    unified: '统一环境',
    unrecorded: '未记录运行时',
    sort: '排序',
    readiness: '按就绪度',
    name: '按名称 A–Z',
    model: '模型',
    tasks: '任务',
    python: 'Python',
    cuda: 'CUDA',
    none: '没有符合当前筛选条件的模型配方。',
    more: '显示更多模型配方',
  },
} as const;

function includesQuery(recipe: ModelRecipeIndexEntry, query: string) {
  if (!query) return true;
  const haystack = [
    recipe.id,
    recipe.name,
    recipe.provider,
    recipe.summary,
    ...recipe.aliases,
    ...recipe.tasks,
  ]
    .join(' ')
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

function recipeHref(recipe: ModelRecipeIndexEntry, locale: Locale) {
  const prefix = locale === 'zh' ? '/zh' : '';
  return `${prefix}/docs/guides/supported-models/${recipe.id}`;
}

function taskFactLabel(task: string) {
  const spaced = task.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim();
  if (!spaced) return task;
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function recipeFacts(recipe: ModelRecipeIndexEntry, t: (typeof copy)[Locale]) {
  const facts = [
    recipe.tasks[0] ? taskFactLabel(recipe.tasks[0]) : '',
    [recipe.runtime.python ? `${t.python} ${recipe.runtime.python}` : '', formatCudaLabel(recipe.runtime.cudaLabel)]
      .filter(Boolean)
      .join(' · '),
  ].filter(Boolean);
  return facts.slice(0, 2);
}

export function ModelRecipeCatalog({ locale = 'en' }: { locale?: Locale }) {
  const t = copy[locale];
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const [status, setStatus] = useState<'all' | ModelRecipeStatusGroup>('all');
  const [runtime, setRuntime] = useState<RuntimeFilter>('all');
  const [sort, setSort] = useState<SortMode>('readiness');
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const deferredQuery = useDeferredValue(query.trim());

  const results = useMemo(() => {
    const filtered = modelRecipeIndex.recipes.filter((recipe) => {
      if (category !== 'all' && recipe.category !== category) return false;
      if (status !== 'all' && recipe.status.group !== status) return false;
      if (runtime !== 'all' && recipe.runtime.environmentKind !== runtime) return false;
      return includesQuery(recipe, deferredQuery);
    });

    return filtered.sort((left, right) => {
      if (sort === 'name') return left.name.localeCompare(right.name);
      const statusDifference = STATUS_RANK[left.status.group] - STATUS_RANK[right.status.group];
      return statusDifference || left.name.localeCompare(right.name);
    });
  }, [category, deferredQuery, runtime, sort, status]);

  const visibleResults = results.slice(0, visibleCount);
  const resetVisible = () => setVisibleCount(PAGE_SIZE);

  return (
    <div className="wf-recipe-catalog">
      <div className="wf-recipe-catalog-coverage">
        <span>{modelRecipeIndex.total} models</span>
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

      <div className="wf-recipe-family-tabs" role="tablist" aria-label="Model families">
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
          <span>{modelRecipeIndex.total}</span>
        </button>
        {modelRecipeIndex.categories.map((item) => (
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
            {locale === 'zh' ? item.label_zh : item.label}
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
                setStatus(event.target.value as 'all' | ModelRecipeStatusGroup);
                resetVisible();
              }}
            >
              <option value="all">{t.allStatuses}</option>
              <option value="verified">Runner verified</option>
              <option value="integrated">Integrated</option>
              <option value="runtime_ported">Runtime ported</option>
              <option value="profile">Profile only</option>
              <option value="planned">Planned</option>
              <option value="blocked">Blocked</option>
            </select>
          </label>
          <label>
            <span>{t.runtime}</span>
            <select
              value={runtime}
              onChange={(event) => {
                setRuntime(event.target.value as RuntimeFilter);
                resetVisible();
              }}
            >
              <option value="all">{t.allRuntimes}</option>
              <option value="dedicated">{t.dedicated}</option>
              <option value="unified">{t.unified}</option>
              <option value="unrecorded">{t.unrecorded}</option>
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

      <div className="wf-recipe-table wf-model-table wf-recipe-list" role="list" aria-label="Model recipes">
        {visibleResults.length > 0 ? (
          visibleResults.map((recipe, index) => {
            const facts = recipeFacts(recipe, t);
            return (
            <Link
              className="wf-recipe-row"
              href={recipeHref(recipe, locale)}
              role="listitem"
              key={recipe.id}
              style={{ '--wf-row-index': Math.min(index, 12) } as CSSProperties}
            >
              <span className="wf-recipe-row-model">
                <ModelIdentityMark
                  id={recipe.id}
                  name={recipe.name}
                  provider={recipe.provider}
                  category={recipe.category}
                  size="medium"
                />
                <span className="wf-recipe-row-identity">
                  <span>
                    <strong>{recipe.name}</strong>
                    <em>{recipe.provider}</em>
                  </span>
                  <small>{recipe.summary}</small>
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
