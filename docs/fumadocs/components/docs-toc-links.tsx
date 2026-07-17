'use client';

import type { TOCItemType } from 'fumadocs-core/toc';
import type { CSSProperties, RefObject } from 'react';

import { DocsTocLink } from '@/components/docs-toc-link';
import { useDocsTocVisibility } from '@/lib/use-docs-toc-visibility';

type DocsTocLinksProps = {
  items: TOCItemType[];
  linksRef: RefObject<HTMLDivElement | null>;
  className?: string;
};

export function DocsTocLinks({ items, linksRef, className = 'pi-doc-toc-links' }: DocsTocLinksProps) {
  const { visible, branchActive } = useDocsTocVisibility(items);

  return (
    <div className={className} ref={linksRef}>
      {items.map((item, index) => {
        if (!visible.has(index)) return null;

        return (
          <DocsTocLink
            href={item.url}
            key={`${item.url}-${index}`}
            scrollContainerRef={linksRef}
            branchActive={branchActive.has(index)}
            style={
              {
                '--toc-indent': `${Math.max(0, item.depth - 2) * 12}px`,
              } as CSSProperties
            }
          >
            {item.title}
          </DocsTocLink>
        );
      })}
    </div>
  );
}
