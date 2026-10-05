'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

import { ApiCopyButton } from '@/components/api-copy-button';
import { ModelIdentityMark } from '@/components/model-identity-mark';
import {
  dedicatedEnvExceptions,
  dedicatedEnvSetupCommand,
  type DedicatedEnvEntry,
  type DedicatedEnvGroup,
} from '@/lib/dedicated-env-exceptions';
import { modelRecipeIndex } from '@/lib/model-recipe-index';

type Locale = 'en' | 'zh';
type GroupFilter = 'all' | DedicatedEnvGroup;

const GROUPS: DedicatedEnvGroup[] = ['visual', 'embodied', 'prepare'];

const copy = {
  en: {
    count: 'dedicated environments',
    hint: 'Click a row for the setup command and notes.',
    all: 'All',
    visual: 'Visual & world',
    embodied: 'Embodied & JAX',
    prepare: 'Prepare-only',
    expand: 'Expand all',
    collapse: 'Collapse all',
    setup: 'Setup',
    notes: 'Notes',
    copy: 'Copy',
    copied: 'Copied',
  },
  zh: {
    count: '个专用环境',
    hint: '点开一行查看安装命令和说明。',
    all: '全部',
    visual: '视觉与世界',
    embodied: '具身与 JAX',
    prepare: '仅准备',
    expand: '全部展开',
    collapse: '全部收起',
    setup: '安装',
    notes: '说明',
    copy: '复制',
    copied: '已复制',
  },
} as const;

function recipeIdentity(modelId: string) {
  const recipe = modelRecipeIndex.recipes.find((item) => item.id === modelId);
  return {
    id: modelId,
    name: recipe?.name ?? modelId,
    provider: recipe?.provider ?? '',
    category: recipe?.category ?? 'world_models',
  };
}

function groupLabel(group: DedicatedEnvGroup, locale: Locale) {
  return copy[locale][group];
}

function groupCount(group: DedicatedEnvGroup) {
  return dedicatedEnvExceptions.filter((entry) => entry.group === group).length;
}

function DedicatedEnvRow({
  entry,
  locale,
  open,
  onToggle,
}: {
  entry: DedicatedEnvEntry;
  locale: Locale;
  open: boolean;
  onToggle: (id: string, nextOpen: boolean) => void;
}) {
  const t = copy[locale];
  const identity = recipeIdentity(entry.models[0]);
  const command = dedicatedEnvSetupCommand(entry.setupModel);

  return (
    <details
      className="pi-env-entry"
      open={open}
      onToggle={(event) => onToggle(entry.id, event.currentTarget.open)}
    >
      <summary>
        <span className="pi-env-entry-chevron" aria-hidden="true" />
        <ModelIdentityMark
          id={identity.id}
          name={identity.name}
          provider={identity.provider}
          category={identity.category}
          size="small"
        />
        <span className="pi-env-entry-main">
          <span className="pi-env-entry-models">
            {entry.models.map((model) => (
              <code key={model}>{model}</code>
            ))}
          </span>
          <code className="pi-env-entry-env" title={entry.env}>
            {entry.env}
          </code>
        </span>
      </summary>
      <div className="pi-env-entry-body">
        <div className="pi-env-entry-command">
          <div>
            <span>{t.setup}</span>
            <ApiCopyButton value={command} label={t.copy} doneLabel={t.copied} />
          </div>
          <code>{command}</code>
        </div>
        <p>
          <strong>{t.notes}</strong>
          {entry.notes[locale]}
        </p>
      </div>
    </details>
  );
}

export function DedicatedEnvCatalog({ locale = 'en' }: { locale?: Locale }) {
  const t = copy[locale];
  const [group, setGroup] = useState<GroupFilter>('all');
  const [openIds, setOpenIds] = useState<Set<string>>(() => new Set());
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
  }, []);

  const visible = useMemo(
    () =>
      dedicatedEnvExceptions.filter((entry) => group === 'all' || entry.group === group),
    [group],
  );

  const visibleGroups = group === 'all' ? GROUPS : [group];
  const allOpen = visible.every((entry) => openIds.has(entry.id));

  function setOpen(id: string, nextOpen: boolean) {
    if (!mountedRef.current) return;
    setOpenIds((current) => {
      const already = current.has(id);
      if (already === nextOpen) return current;
      const next = new Set(current);
      if (nextOpen) next.add(id);
      else next.delete(id);
      return next;
    });
  }

  function toggleVisible(nextOpen: boolean) {
    setOpenIds((current) => {
      const next = new Set(current);
      for (const entry of visible) {
        if (nextOpen) next.add(entry.id);
        else next.delete(entry.id);
      }
      return next;
    });
  }

  return (
    <div className="pi-env-catalog not-prose">
      <div className="pi-env-catalog-toolbar">
        <p>
          <strong>{dedicatedEnvExceptions.length}</strong> {t.count}
          <span>{t.hint}</span>
        </p>
        <button type="button" onClick={() => toggleVisible(!allOpen)}>
          {allOpen ? t.collapse : t.expand}
        </button>
      </div>

      <div className="pi-env-catalog-filters" role="tablist" aria-label={t.count}>
        <button
          type="button"
          role="tab"
          aria-selected={group === 'all'}
          className={group === 'all' ? 'is-active' : undefined}
          onClick={() => setGroup('all')}
        >
          {t.all}
          <span>{dedicatedEnvExceptions.length}</span>
        </button>
        {GROUPS.map((item) => (
          <button
            type="button"
            role="tab"
            aria-selected={group === item}
            className={group === item ? 'is-active' : undefined}
            key={item}
            onClick={() => setGroup(item)}
          >
            {groupLabel(item, locale)}
            <span>{groupCount(item)}</span>
          </button>
        ))}
      </div>

      <div className="pi-env-catalog-list">
        {visibleGroups.map((item) => {
          const entries = visible.filter((entry) => entry.group === item);
          if (entries.length === 0) return null;

          return (
            <section className="pi-env-group" key={item}>
              {group === 'all' ? (
                <h3>
                  {groupLabel(item, locale)}
                  <span>{entries.length}</span>
                </h3>
              ) : null}
              {entries.map((entry) => (
                <DedicatedEnvRow
                  entry={entry}
                  key={entry.id}
                  locale={locale}
                  open={openIds.has(entry.id)}
                  onToggle={setOpen}
                />
              ))}
            </section>
          );
        })}
      </div>
    </div>
  );
}
