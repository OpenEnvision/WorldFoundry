'use client';

import type { TOCItemType } from 'fumadocs-core/toc';
import { useMemo } from 'react';

import { enrichApiReferenceToc } from '@/lib/docs-api-toc';

export function useDocsTocItems(slugs: readonly string[], items: TOCItemType[]) {
  return useMemo(() => enrichApiReferenceToc(slugs, items), [items, slugs]);
}
