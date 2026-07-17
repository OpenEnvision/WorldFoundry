import apiReference from '@/generated/python-api.json';
import { createElement, type ReactNode } from 'react';
import type { TOCItemType } from 'fumadocs-core/toc';

type ApiSymbol = {
  name: string;
  kind: string;
  methods: Array<{ name: string; kind: string }>;
};

type ApiReferenceData = {
  groups: Record<string, string[]>;
  symbols: Record<string, ApiSymbol>;
};

const data = apiReference as ApiReferenceData;

const KIND_ABBR: Record<string, string> = {
  class: 'cls',
  function: 'func',
  protocol: 'prot',
  method: 'meth',
  property: 'prop',
  classmethod: 'cmeth',
  staticmethod: 'smeth',
};

export function symbolAnchor(symbol: string) {
  return `api-${symbol.replace(/[^a-zA-Z0-9_-]+/g, '-')}`;
}

export function methodAnchor(symbol: string, methodName: string) {
  return `${symbolAnchor(symbol)}--${methodName}`;
}

function TocBadge({ kind }: { kind: string }) {
  const label = KIND_ABBR[kind] ?? kind.slice(0, 4);
  return createElement(
    'span',
    {
      className: `wf-toc-symbol wf-toc-symbol-${kind}`,
      'aria-hidden': true,
    },
    label,
  );
}

function tocTitle(kind: string, name: string): ReactNode {
  return createElement(
    'span',
    { className: 'wf-toc-entry' },
    createElement(TocBadge, { kind }),
    createElement('code', null, name),
  );
}

function plainTitle(title: TOCItemType['title']): string {
  if (typeof title === 'string' || typeof title === 'number') return String(title);
  if (Array.isArray(title)) {
    return title.map((part) => plainTitle(part as TOCItemType['title'])).join('');
  }
  if (title && typeof title === 'object' && 'props' in title) {
    const props = (title as { props?: { children?: unknown } }).props;
    if (props?.children !== undefined) {
      return plainTitle(props.children as TOCItemType['title']);
    }
  }
  return '';
}

function buildSymbolTocItems(symbols: string[]): TOCItemType[] {
  const apiItems: TOCItemType[] = [];
  for (const qualified of symbols) {
    const entry = data.symbols[qualified];
    if (!entry) continue;
    apiItems.push({
      title: tocTitle(entry.kind, entry.name),
      url: `#${symbolAnchor(qualified)}`,
      depth: 3,
    });
    for (const method of entry.methods) {
      if (method.name.startsWith('_') && method.name !== '__call__') continue;
      apiItems.push({
        title: tocTitle(method.kind || 'method', method.name),
        url: `#${methodAnchor(qualified, method.name)}`,
        depth: 4,
      });
    }
  }
  return apiItems;
}

function decorateProseToc(page: string, toc: TOCItemType[]): TOCItemType[] {
  const symbols = data.groups[page];
  if (!symbols?.length) return toc;

  const byName = new Map(
    symbols.map((qualified) => [data.symbols[qualified]?.name, data.symbols[qualified]]),
  );

  return toc.map((item) => {
    const text = plainTitle(item.title).replace(/`/g, '').trim();
    const entry = byName.get(text);
    if (!entry) return item;
    return {
      ...item,
      title: tocTitle(entry.kind, entry.name),
    };
  });
}

/**
 * Merge generated API symbols into the page TOC (vLLM / mkdocstrings-style).
 *
 * Core group pages render symbols in React, so MDX TOC misses them — inject
 * after "Complete reference". Evaluation pages already have MDX headings; only
 * decorate those titles with kind badges.
 */
export function enrichApiReferenceToc(
  slugs: readonly string[],
  toc: TOCItemType[],
): TOCItemType[] {
  if (slugs[0] !== 'api-reference' || slugs.length < 2) {
    return toc;
  }

  const page = slugs[1];
  const symbols = data.groups[page];
  if (!symbols?.length) {
    return toc;
  }

  // Evaluation / hand-written pages: headings already in MDX.
  if (!page.startsWith('core-')) {
    return decorateProseToc(page, toc);
  }

  const apiItems = buildSymbolTocItems(symbols);
  if (apiItems.length === 0) return toc;

  const completeIndex = toc.findIndex((item) =>
    /complete reference|完整参考/i.test(plainTitle(item.title)),
  );

  if (completeIndex >= 0) {
    return [
      ...toc.slice(0, completeIndex + 1),
      ...apiItems,
      ...toc.slice(completeIndex + 1),
    ];
  }

  return [
    ...toc,
    {
      title: 'Reference',
      url: apiItems[0].url,
      depth: 2,
    },
    ...apiItems,
  ];
}

export function isApiReferenceDocsPage(slugs: readonly string[]) {
  return slugs[0] === 'api-reference';
}
