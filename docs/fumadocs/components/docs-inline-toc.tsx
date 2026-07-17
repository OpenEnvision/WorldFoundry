'use client';

import { ScrollProvider, type TOCItemType } from 'fumadocs-core/toc';
import { useRef } from 'react';

import { DocsTocLinks } from '@/components/docs-toc-links';
import { DocsTocScrollSpyProvider } from '@/lib/docs-toc-scroll-spy';
import { useDocsTocItems } from '@/lib/use-docs-toc-items';
import { useMediaQuery } from '@/lib/use-media-query';

type DocsInlineTocProps = {
  title: string;
  items: TOCItemType[];
  slugs: readonly string[];
  pageKey: string;
};

export function DocsInlineToc({ title, items, slugs, pageKey }: DocsInlineTocProps) {
  const wide = useMediaQuery('(min-width: 1280px)');
  const linksRef = useRef<HTMLDivElement>(null);
  const toc = useDocsTocItems(slugs, items);

  if (wide || toc.length === 0) return null;

  return (
    <DocsTocScrollSpyProvider key={pageKey} items={toc}>
      <ScrollProvider containerRef={linksRef}>
        <details className="pi-doc-inline-toc">
          <summary>{title}</summary>
          <DocsTocLinks items={toc} linksRef={linksRef} className="pi-doc-toc-links pi-doc-inline-toc-links" />
        </details>
      </ScrollProvider>
    </DocsTocScrollSpyProvider>
  );
}
