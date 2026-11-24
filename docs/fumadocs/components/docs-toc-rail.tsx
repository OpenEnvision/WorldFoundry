'use client';

import { ScrollProvider, type TOCItemType } from 'fumadocs-core/toc';
import { AlignLeft } from 'lucide-react';
import { type ReactNode, useRef } from 'react';

import { DocsTocLinks } from '@/components/docs-toc-links';
import { DocsTocScrollSpyProvider } from '@/lib/docs-toc-scroll-spy';

type DocsTocRailProps = {
  title: string;
  items: TOCItemType[];
  pageKey: string;
  children?: ReactNode;
};

export function DocsTocRail({ title, items, pageKey, children }: DocsTocRailProps) {
  const linksRef = useRef<HTMLDivElement>(null);

  if (items.length === 0 && !children) return null;

  if (items.length === 0) {
    return (
      <aside className="pi-doc-right-rail" aria-label={title}>
        {children}
      </aside>
    );
  }

  return (
    <DocsTocScrollSpyProvider key={pageKey} items={items}>
      <ScrollProvider containerRef={linksRef}>
        <aside className="pi-doc-right-rail" aria-label={title}>
          <nav className="pi-doc-toc">
            <span className="pi-doc-toc-title">
              <AlignLeft aria-hidden="true" size={14} strokeWidth={1.8} />
              {title}
            </span>
            <DocsTocLinks items={items} linksRef={linksRef} />
          </nav>
          {children}
        </aside>
      </ScrollProvider>
    </DocsTocScrollSpyProvider>
  );
}
