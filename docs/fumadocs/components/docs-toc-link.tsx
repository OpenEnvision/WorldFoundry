'use client';

import type { CSSProperties, ReactNode } from 'react';
import { useEffect, useRef } from 'react';

import { useDocsTocActiveId } from '@/lib/docs-toc-scroll-spy';

type DocsTocLinkProps = {
  href: string;
  children: ReactNode;
  style?: CSSProperties;
  scrollContainerRef?: React.RefObject<HTMLDivElement | null>;
  branchActive?: boolean;
};

export function DocsTocLink({
  href,
  children,
  style,
  scrollContainerRef,
  branchActive = false,
}: DocsTocLinkProps) {
  const active = useDocsTocActiveId(href);
  const ref = useRef<HTMLAnchorElement>(null);

  useEffect(() => {
    if (!active || !scrollContainerRef?.current || !ref.current) return;

    ref.current.scrollIntoView({
      block: 'nearest',
      inline: 'nearest',
    });
  }, [active, scrollContainerRef]);

  return (
    <a
      ref={ref}
      href={href}
      data-active={active ? 'true' : 'false'}
      data-branch-active={branchActive ? 'true' : 'false'}
      style={style}
    >
      {children}
    </a>
  );
}
