'use client';
import {
  SearchDialog,
  SearchDialogClose,
  SearchDialogContent,
  SearchDialogHeader,
  SearchDialogIcon,
  SearchDialogInput,
  SearchDialogList,
  SearchDialogOverlay,
  type SharedProps,
} from 'fumadocs-ui/components/dialog/search';
import { createContentHighlighter, type SortedResult } from 'fumadocs-core/search';
import { oramaStaticClient } from 'fumadocs-core/search/client/orama-static';
import { create } from '@orama/orama';
import { I18nProvider } from 'fumadocs-ui/contexts/i18n';
import { usePathname } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';
import { i18n, localeNames, type Locale } from '@/lib/i18n';
import { stripBasePath, withBasePath } from '@/lib/site-path';

type SearchDocument = {
  id: string;
  page_id: string;
  type: SortedResult['type'];
  content: string;
  breadcrumbs?: string[];
  url: string;
};

type SearchIndex = {
  type: 'i18n';
  data: Record<
    string,
    {
      docs: {
        docs: Record<string, SearchDocument>;
      };
    }
  >;
};

let staticSearchIndex: Promise<SearchIndex> | undefined;

function initOrama() {
  return create({
    schema: { _: 'string' },
    // https://docs.orama.com/docs/orama-js/supported-languages
    language: 'english',
  });
}

function normalizeForLooseSearch(value: string) {
  return value.toLocaleLowerCase().replace(/\s+/g, '');
}

async function loadSearchIndex() {
  staticSearchIndex ??= fetch(withBasePath('/api/search') ?? '/api/search').then((res) => {
    if (!res.ok) throw new Error('Failed to fetch search index.');
    return res.json() as Promise<SearchIndex>;
  });

  return staticSearchIndex;
}

async function searchChineseFallback(query: string): Promise<SortedResult[]> {
  const normalizedQuery = normalizeForLooseSearch(query);

  if (!normalizedQuery) return [];

  const index = await loadSearchIndex();
  const documents = Object.values(index.data.zh?.docs.docs ?? {});
  const pageDocuments = new Map(
    documents.filter((doc) => doc.type === 'page').map((doc) => [doc.page_id, doc]),
  );
  const highlighter = createContentHighlighter(query);
  const results: SortedResult[] = [];
  const emitted = new Set<string>();

  for (const doc of documents) {
    const searchable = normalizeForLooseSearch(`${doc.content} ${doc.breadcrumbs?.join(' ') ?? ''}`);

    if (!searchable.includes(normalizedQuery)) continue;

    const page = pageDocuments.get(doc.page_id) ?? doc;

    if (!emitted.has(page.page_id)) {
      results.push({
        id: page.page_id,
        type: 'page',
        content: highlighter.highlightMarkdown(page.content),
        breadcrumbs: page.breadcrumbs,
        url: page.url,
      });
      emitted.add(page.page_id);
    }

    if (doc.type !== 'page' && !emitted.has(doc.id)) {
      results.push({
        id: doc.id,
        type: doc.type,
        content: highlighter.highlightMarkdown(doc.content),
        breadcrumbs: doc.breadcrumbs,
        url: doc.url,
      });
      emitted.add(doc.id);
    }

    if (results.length >= 60) break;
  }

  return results;
}

function createSearchClient(locale: Locale) {
  const client = oramaStaticClient({
    from: withBasePath('/api/search') ?? '/api/search',
    initOrama,
    locale,
  });

  return {
    async search(query: string) {
      const results = await client.search(query);

      if (locale !== 'zh' || results.length > 0) return results;

      return searchChineseFallback(query);
    },
  };
}

function useDebouncedValue(value: string, delayMs: number) {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    if (delayMs === 0) {
      setDebounced(value);
      return;
    }

    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [delayMs, value]);

  return delayMs === 0 ? value : debounced;
}

function useDocsSearchEffect(client: ReturnType<typeof createSearchClient>) {
  const [search, setSearch] = useState('');
  const [results, setResults] = useState<SortedResult[] | 'empty'>('empty');
  const [error, setError] = useState<unknown>();
  const [isLoading, setIsLoading] = useState(false);
  const debouncedSearch = useDebouncedValue(search, 100);

  useEffect(() => {
    let cancelled = false;

    async function run() {
      if (debouncedSearch.length === 0) {
        if (!cancelled) {
          setResults('empty');
          setError(undefined);
          setIsLoading(false);
        }
        return;
      }

      setIsLoading(true);

      try {
        const data = await client.search(debouncedSearch);
        if (!cancelled) {
          setResults(data);
          setError(undefined);
        }
      } catch (err) {
        if (!cancelled) setError(err);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    void run();

    return () => {
      cancelled = true;
    };
  }, [client, debouncedSearch]);

  return useMemo(
    () => ({
      search,
      setSearch,
      query: {
        isLoading,
        data: results,
        error,
      },
    }),
    [search, isLoading, results, error],
  );
}

export default function DefaultSearchDialog(props: SharedProps) {
  const pathname = stripBasePath(usePathname());
  const locale: Locale = pathname === '/zh' || pathname.startsWith('/zh/') ? 'zh' : 'en';
  const searchClient = useMemo(() => createSearchClient(locale), [locale]);
  const { search, setSearch, query } = useDocsSearchEffect(searchClient);

  return (
    <I18nProvider
      locale={locale}
      locales={i18n.languages.map((item) => ({ locale: item, name: localeNames[item] }))}
      translations={locale === 'zh' ? { search: '搜索文档', searchNoResult: '没有找到结果' } : undefined}
    >
      <SearchDialog search={search} onSearchChange={setSearch} isLoading={query.isLoading} {...props}>
        <SearchDialogOverlay />
        <SearchDialogContent>
          <SearchDialogHeader>
            <SearchDialogIcon />
            <SearchDialogInput />
            <SearchDialogClose />
          </SearchDialogHeader>
          <SearchDialogList items={query.data !== 'empty' ? query.data : null} />
        </SearchDialogContent>
      </SearchDialog>
    </I18nProvider>
  );
}
