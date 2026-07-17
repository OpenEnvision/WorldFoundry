'use client';

import Link from 'next/link';
import { ArrowRight, Search } from 'lucide-react';
import { type CSSProperties, useDeferredValue, useMemo, useState } from 'react';

import { ModelIdentityMark } from '@/components/model-identity-mark';
import { modelRecipeIndex } from '@/lib/model-recipe-index';
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
    checkpoint: 'Checkpoint',
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
    checkpoint: 'Checkpoint',
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

function runtimeLabel(recipe: ModelRecipeIndexEntry, locale: Locale) {
  const t = copy[locale];
  if (recipe.runtime.environmentKind === 'dedicated') return t.dedicated;
  if (recipe.runtime.environmentKind === 'unified') return t.unified;
  if (recipe.runtime.profileId) return locale === 'zh' ? '仅 Profile' : 'Profile only';
  return t.unrecorded;
}

function revisionLabel(recipe: ModelRecipeIndexEntry) {
  if (!recipe.checkpoint) return '—';
  if (recipe.checkpoint.revision) return recipe.checkpoint.revision.slice(0, 9);
  return recipe.checkpoint.id;
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

      <div className="wf-recipe-table" role="table" aria-label="Model recipes">
        <div className="wf-recipe-table-head" role="row">
          <span role="columnheader">{t.model}</span>
          <span role="columnheader">{t.tasks}</span>
          <span role="columnheader">{t.runtime}</span>
          <span role="columnheader">{t.python}</span>
          <span role="columnheader">{t.cuda}</span>
          <span role="columnheader">{t.checkpoint}</span>
          <span aria-hidden="true" />
        </div>

        {visibleResults.length > 0 ? (
          <div role="rowgroup">
            {visibleResults.map((recipe, index) => (
              <Link
                className="wf-recipe-row"
                href={recipeHref(recipe, locale)}
                role="row"
                key={recipe.id}
                style={{ '--wf-row-index': Math.min(index, 12) } as CSSProperties}
              >
                <span className="wf-recipe-row-model" role="cell">
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
                  </span>
                </span>
                <span className="wf-recipe-row-tasks" role="cell">
                  {recipe.tasks.slice(0, 2).map((task) => (
                    <code key={task}>{task}</code>
                  ))}
                  {recipe.tasks.length > 2 ? <small>+{recipe.tasks.length - 2}</small> : null}
                </span>
                <span className="wf-recipe-row-runtime" role="cell">
                  <span className={`wf-recipe-status wf-recipe-status-${recipe.status.group}`}>
                    {recipe.status.label}
                  </span>
                  <small>{runtimeLabel(recipe, locale)}</small>
                </span>
                <code role="cell">{recipe.runtime.python ?? '—'}</code>
                <code role="cell">{recipe.runtime.cudaLabel?.replace('CUDA ', '') ?? '—'}</code>
                <code className="wf-recipe-row-revision" role="cell" title={recipe.checkpoint?.id}>
                  {revisionLabel(recipe)}
                </code>
                <ArrowRight aria-hidden="true" role="cell" size={17} strokeWidth={1.5} />
              </Link>
            ))}
          </div>
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
