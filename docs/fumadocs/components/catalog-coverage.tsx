'use client';

import { useDeferredValue, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { ArrowRight, Search } from 'lucide-react';

import {
  benchmarkHubMissingIds,
  catalogCoverage,
  type CatalogEntry,
} from '@/lib/catalog-coverage';

type Surface = 'models' | 'benchmarks';

function matchesQuery(entry: CatalogEntry, query: string) {
  if (!query) return true;
  const q = query.toLowerCase();
  if (entry.name.toLowerCase().includes(q) || entry.id.toLowerCase().includes(q)) return true;
  return (entry.aliases ?? []).some((alias) => alias.toLowerCase().includes(q));
}

function entryTitle(entry: CatalogEntry) {
  const aliases = entry.aliases?.filter((alias) => alias !== entry.id) ?? [];
  const parts = [entry.id, entry.status ? `status: ${entry.status}` : null];
  if (aliases.length) parts.push(`variants: ${aliases.join(', ')}`);
  return parts.filter(Boolean).join(' · ');
}

function EntryRow({
  entry,
  href,
}: {
  entry: CatalogEntry;
  href?: string;
}) {
  const body = (
    <>
      <span className="wf-catalog-entry-name">{entry.name}</span>
      <span className="wf-catalog-entry-id">{entry.id}</span>
    </>
  );

  if (href) {
    return (
      <li>
        <Link className="wf-catalog-entry" href={href} title={entryTitle(entry)}>
          {body}
          <ArrowRight aria-hidden="true" className="wf-catalog-entry-arrow" size={14} strokeWidth={1.8} />
        </Link>
      </li>
    );
  }

  return (
    <li>
      <div className="wf-catalog-entry" title={entryTitle(entry)}>
        {body}
      </div>
    </li>
  );
}

export function CatalogCoverage() {
  const { modelsTotal, benchmarksTotal, modelFamilies, benchmarkGroups } = catalogCoverage;
  const [surface, setSurface] = useState<Surface>('models');
  const [familyId, setFamilyId] = useState(modelFamilies[0]?.id ?? 'video');
  const [groupId, setGroupId] = useState(benchmarkGroups[0]?.id ?? 'video');
  const [query, setQuery] = useState('');
  const deferredQuery = useDeferredValue(query.trim());
  const isSearching = deferredQuery.length > 0;

  useEffect(() => {
    if (!isSearching) return;
    // Keep surface coherent while searching: prefer the side with matches.
    const modelHits = modelFamilies.some((family) =>
      family.entries.some((entry) => matchesQuery(entry, deferredQuery)),
    );
    const benchHits = benchmarkGroups.some((group) =>
      group.entries.some((entry) => matchesQuery(entry, deferredQuery)),
    );
    if (surface === 'models' && !modelHits && benchHits) setSurface('benchmarks');
    if (surface === 'benchmarks' && !benchHits && modelHits) setSurface('models');
  }, [deferredQuery, isSearching, modelFamilies, benchmarkGroups, surface]);

  const activeFamily = modelFamilies.find((family) => family.id === familyId) ?? modelFamilies[0];
  const activeGroup = benchmarkGroups.find((group) => group.id === groupId) ?? benchmarkGroups[0];

  const modelResults = useMemo(() => {
    if (isSearching) {
      return modelFamilies.flatMap((family) =>
        family.entries
          .filter((entry) => matchesQuery(entry, deferredQuery))
          .map((entry) => ({ entry, familyLabel: family.label })),
      );
    }
    return (activeFamily?.entries ?? []).map((entry) => ({
      entry,
      familyLabel: activeFamily?.label ?? '',
    }));
  }, [isSearching, deferredQuery, modelFamilies, activeFamily]);

  const benchmarkResults = useMemo(() => {
    if (isSearching) {
      return benchmarkGroups.flatMap((group) =>
        group.entries
          .filter((entry) => matchesQuery(entry, deferredQuery))
          .map((entry) => ({ entry, groupLabel: group.label })),
      );
    }
    return (activeGroup?.entries ?? []).map((entry) => ({
      entry,
      groupLabel: activeGroup?.label ?? '',
    }));
  }, [isSearching, deferredQuery, benchmarkGroups, activeGroup]);

  const results = surface === 'models' ? modelResults : benchmarkResults;
  const resultCount = results.length;

  return (
    <div className="wf-catalog">
      <div className="wf-catalog-lede">
        <div className="wf-catalog-totals" aria-label="Catalog totals">
          <p>
            <strong>{modelsTotal}</strong> models
            <span aria-hidden="true"> · </span>
            <strong>{benchmarksTotal}</strong> benchmarks
          </p>
        </div>
      </div>

      <div className="wf-catalog-finder">
        <div className="wf-catalog-finder-bar">
          <div className="wf-catalog-surface" role="tablist" aria-label="Catalog surface">
            <button
              type="button"
              role="tab"
              aria-selected={surface === 'models'}
              className={surface === 'models' ? 'is-active' : undefined}
              onClick={() => setSurface('models')}
            >
              Models
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={surface === 'benchmarks'}
              className={surface === 'benchmarks' ? 'is-active' : undefined}
              onClick={() => setSurface('benchmarks')}
            >
              Benchmarks
            </button>
          </div>

          <label className="wf-catalog-search">
            <Search aria-hidden="true" size={15} strokeWidth={1.8} />
            <span className="sr-only">Search catalog</span>
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={
                surface === 'models'
                  ? 'Search models by name or id…'
                  : 'Search benchmarks by name or id…'
              }
              autoComplete="off"
            />
          </label>
        </div>

        <div className="wf-catalog-finder-body">
          {!isSearching ? (
            <nav className="wf-catalog-rail" aria-label={surface === 'models' ? 'Model families' : 'Benchmark groups'}>
              {surface === 'models'
                ? modelFamilies.map((family) => (
                    <button
                      type="button"
                      key={family.id}
                      className={family.id === familyId ? 'is-active' : undefined}
                      aria-current={family.id === familyId ? 'true' : undefined}
                      onClick={() => setFamilyId(family.id)}
                    >
                      <span className="wf-catalog-rail-label">{family.label}</span>
                      <span className="wf-catalog-rail-count">{family.count}</span>
                      <span className="wf-catalog-rail-blurb">{family.blurb}</span>
                    </button>
                  ))
                : benchmarkGroups.map((group) => (
                    <button
                      type="button"
                      key={group.id}
                      className={group.id === groupId ? 'is-active' : undefined}
                      aria-current={group.id === groupId ? 'true' : undefined}
                      onClick={() => setGroupId(group.id)}
                    >
                      <span className="wf-catalog-rail-label">{group.label}</span>
                      <span className="wf-catalog-rail-count">{group.count}</span>
                    </button>
                  ))}
            </nav>
          ) : null}

          <div className="wf-catalog-pane">
            <header className="wf-catalog-pane-header">
              <div>
                <p className="wf-catalog-pane-kicker">
                  {isSearching
                    ? 'Search results'
                    : surface === 'models'
                      ? activeFamily?.label
                      : activeGroup?.label}
                </p>
                <h3>
                  {isSearching
                    ? `Matches for “${deferredQuery}”`
                    : surface === 'models'
                      ? activeFamily?.blurb
                      : 'Open a page in the Benchmark Hub for readiness and runners.'}
                </h3>
              </div>
              <p className="wf-catalog-pane-meta" aria-live="polite">
                {resultCount} {surface === 'models' ? 'model' : 'benchmark'}
                {resultCount === 1 ? '' : 's'}
              </p>
            </header>

            {resultCount === 0 ? (
              <p className="wf-catalog-empty">Nothing matched. Try another name or id.</p>
            ) : (
              <ul className="wf-catalog-entries">
                {results.map(({ entry }) => (
                  <EntryRow
                    key={`${surface}-${entry.id}`}
                    entry={entry}
                    href={
                      surface === 'models'
                        ? `/docs/guides/supported-models/${entry.aliases?.[0] ?? entry.id}`
                        : !benchmarkHubMissingIds.has(entry.id)
                          ? `/docs/evaluation/benchmark-hub/${entry.id}`
                          : undefined
                    }
                  />
                ))}
              </ul>
            )}

            <footer className="wf-catalog-pane-footer">
              {surface === 'models' ? (
                <Link href="/docs/guides/supported-models" className="wf-home-text-link">
                  Open model docs
                  <ArrowRight aria-hidden="true" size={15} strokeWidth={1.8} />
                </Link>
              ) : (
                <Link href="/docs/evaluation/benchmark-hub" className="wf-home-text-link">
                  Open Benchmark Hub
                  <ArrowRight aria-hidden="true" size={15} strokeWidth={1.8} />
                </Link>
              )}
            </footer>
          </div>
        </div>
      </div>
    </div>
  );
}
