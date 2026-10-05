'use client';

import type { TOCItemType } from 'fumadocs-core/toc';

export function useDocsTocItems(_slugs: readonly string[], items: TOCItemType[]) {
  return items;
}
