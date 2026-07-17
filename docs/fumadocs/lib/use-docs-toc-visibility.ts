'use client';

import type { TOCItemType } from 'fumadocs-core/toc';
import { useMemo } from 'react';

import { computeDocsTocVisibility } from '@/lib/docs-toc-visibility';
import { useDocsTocScrollSpyActiveId } from '@/lib/docs-toc-scroll-spy';

export function useDocsTocVisibility(items: TOCItemType[]) {
  const activeId = useDocsTocScrollSpyActiveId();

  return useMemo(() => computeDocsTocVisibility(items, activeId), [activeId, items]);
}
