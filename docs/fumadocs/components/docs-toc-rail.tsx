'use client';

import { ScrollProvider, type TOCItemType } from 'fumadocs-core/toc';
import { useRef } from 'react';

import { DocsTocLinks } from '@/components/docs-toc-links';
import { DocsTocScrollSpyProvider } from '@/lib/docs-toc-scroll-spy';
import { useDocsTocItems } from '@/lib/use-docs-toc-items';

type DocsTocRailProps = {
  title: string;
  items: TOCItemType[];
  slugs: readonly string[];
  pageKey: string;
};

export function DocsTocRail({ title, items, slugs, pageKey }: DocsTocRailProps) {
  const linksRef = useRef<HTMLDivElement>(null);
  const toc = useDocsTocItems(slugs, items);

  if (toc.length === 0) return null;

  return (
    <DocsTocScrollSpyProvider key={pageKey} items={toc}>
      <ScrollProvider containerRef={linksRef}>
        <aside className="pi-doc-right-rail" aria-label={title}>
          <nav className="pi-doc-toc">
            <span>{title}</span>
            <DocsTocLinks items={toc} linksRef={linksRef} />
          </nav>
        </aside>
      </ScrollProvider>
    </DocsTocScrollSpyProvider>
  );
}
