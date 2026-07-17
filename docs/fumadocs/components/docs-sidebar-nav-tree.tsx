'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { BenchmarkBadge } from '@/components/benchmark-badge';
import type { Locale } from '@/lib/i18n';
import {
  isSidebarItemActive,
  type SidebarNavItem,
  type SidebarNavPage,
} from '@/lib/docs-sidebar-shared';

type DocsSidebarNavTreeProps = {
  hub: SidebarNavPage;
  items: SidebarNavItem[];
  currentUrl: string;
  locale: Locale;
  defaultOpen: boolean;
  expandLabel: string;
  collapseLabel: string;
  panelId: string;
  showBadges?: boolean;
};

export function DocsSidebarNavTree({
  hub,
  items,
  currentUrl,
  locale,
  defaultOpen,
  expandLabel,
  collapseLabel,
  panelId,
  showBadges = false,
}: DocsSidebarNavTreeProps) {
  const [open, setOpen] = useState(defaultOpen);
  const panelRef = useRef<HTMLDivElement>(null);
  const hubActive = isSidebarItemActive(hub, currentUrl);

  useEffect(() => {
    setOpen(defaultOpen);
  }, [defaultOpen]);

  useEffect(() => {
    if (!open || !panelRef.current) return;

    const activeLink = panelRef.current.querySelector<HTMLElement>('.pi-doc-link-active');
    activeLink?.scrollIntoView({ block: 'nearest' });
  }, [open, currentUrl]);

  return (
    <div className={['pi-doc-benchmark-tree', open ? 'pi-doc-benchmark-tree-open' : ''].filter(Boolean).join(' ')}>
      <div className="pi-doc-benchmark-tree-head">
        <Link
          href={hub.link.url}
          className={['pi-doc-link', hubActive ? 'pi-doc-link-active' : ''].filter(Boolean).join(' ')}
          aria-current={hubActive ? 'page' : undefined}
        >
          <span className="pi-doc-link-title">{hub.link.label}</span>
        </Link>
        <button
          type="button"
          className="pi-doc-benchmark-tree-toggle"
          aria-expanded={open}
          aria-controls={panelId}
          aria-label={open ? collapseLabel : expandLabel}
          onClick={() => setOpen((value) => !value)}
        >
          <span aria-hidden="true">{open ? '−' : '+'}</span>
        </button>
      </div>
      {open ? (
        <div className="pi-doc-benchmark-tree-panel" id={panelId} ref={panelRef}>
          {items.map((item) => {
            if (item.type === 'divider') {
              return (
                <p className="pi-doc-sidebar-divider" key={`divider-${item.label}`}>
                  {item.label}
                </p>
              );
            }

            const active = isSidebarItemActive(item, currentUrl);

            return (
              <Link
                href={item.link.url}
                className={['pi-doc-link', 'pi-doc-link-deep', active ? 'pi-doc-link-active' : '']
                  .filter(Boolean)
                  .join(' ')}
                aria-current={active ? 'page' : undefined}
                key={item.link.url}
              >
                <span className="pi-doc-link-row">
                  <span className="pi-doc-link-title">{item.link.label}</span>
                  {showBadges && item.badges.length > 0 ? (
                    <span className="pi-doc-link-badges">
                      {item.badges.map((badge) => (
                        <BenchmarkBadge kind={badge.kind} locale={locale} key={badge.kind} />
                      ))}
                    </span>
                  ) : null}
                </span>
              </Link>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
