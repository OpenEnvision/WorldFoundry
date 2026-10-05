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

function tocItemLabel(item: TOCItemType): string {
  const labeled = item as TOCItemType & { label?: string };
  if (typeof labeled.label === 'string' && labeled.label) return labeled.label;
  return titleText(item.title);
}

function titleText(title: TOCItemType['title']): string {
  if (typeof title === 'string' || typeof title === 'number') return String(title);
  if (Array.isArray(title)) {
    return title
      .map((part) => titleText(part as TOCItemType['title']))
      .filter(Boolean)
      .join(' ');
  }
  if (title && typeof title === 'object' && 'props' in title) {
    const props = (title as { props?: { title?: string; children?: unknown } }).props;
    if (typeof props?.title === 'string' && props.title) return props.title;
    if (props?.children !== undefined) {
      return titleText(props.children as TOCItemType['title']);
    }
  }
  return '';
}

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
            label={tocItemLabel(item) || undefined}
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
