'use client';

import type { TOCItemType } from 'fumadocs-core/toc';
import { usePathname } from 'next/navigation';
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useSyncExternalStore,
  type ReactNode,
} from 'react';

// Viewport line used to pick the current section while scrolling.
const ACTIVE_LINE = 120;
// Headings slightly below the line still count (API symbol blocks sit below h2 spacing).
const ACTIVE_SLACK = 56;

export const DOCS_MAIN_SCROLL_EVENT = 'docs-main-scroll';

type DocsTocScrollSpyContextValue = {
  activeId: string | null;
};

const DocsTocScrollSpyContext = createContext<DocsTocScrollSpyContextValue>({
  activeId: null,
});

function getHeadingId(url: string) {
  return url.startsWith('#') ? url.slice(1) : null;
}

function getScrollRoot() {
  return document.querySelector<HTMLElement>('.pi-doc-main');
}

export function resolveActiveHeading(ids: string[]) {
  let active = ids[0] ?? null;
  let bestTop = Number.NEGATIVE_INFINITY;

  for (const id of ids) {
    const element = document.getElementById(id);
    if (!element) continue;

    const top = element.getBoundingClientRect().top;
    if (top <= ACTIVE_LINE + ACTIVE_SLACK && top > bestTop) {
      bestTop = top;
      active = id;
    }
  }

  return active;
}

function subscribeToHeadingUpdates(onStoreChange: () => void) {
  const schedule = () => window.requestAnimationFrame(onStoreChange);

  schedule();
  const timers = [120, 320, 640, 1200].map((delay) => window.setTimeout(schedule, delay));

  const main = getScrollRoot();
  main?.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener(DOCS_MAIN_SCROLL_EVENT, schedule);
  window.addEventListener('wheel', schedule, { passive: true, capture: true });
  window.addEventListener('resize', schedule);
  window.addEventListener('hashchange', schedule);
  window.addEventListener('load', schedule);

  const content = document.querySelector<HTMLElement>('.pi-doc-content');
  let observer: MutationObserver | null = null;
  if (content) {
    observer = new MutationObserver(() => {
      schedule();
    });
    observer.observe(content, { childList: true, subtree: true });
  }

  return () => {
    timers.forEach((timer) => window.clearTimeout(timer));
    main?.removeEventListener('scroll', schedule);
    window.removeEventListener(DOCS_MAIN_SCROLL_EVENT, schedule);
    window.removeEventListener('wheel', schedule, { capture: true });
    window.removeEventListener('resize', schedule);
    window.removeEventListener('hashchange', schedule);
    window.removeEventListener('load', schedule);
    observer?.disconnect();
  };
}

function useActiveHeading(ids: string[]) {
  const getSnapshot = useCallback(() => {
    if (ids.length === 0) return null;
    return resolveActiveHeading(ids);
  }, [ids]);

  return useSyncExternalStore(subscribeToHeadingUpdates, getSnapshot, () => null);
}

export function DocsTocScrollSpyProvider({
  items,
  children,
}: {
  items: TOCItemType[];
  children: ReactNode;
}) {
  const pathname = usePathname();
  const ids = useMemo(
    () =>
      items
        .map((item) => getHeadingId(item.url))
        .filter((id): id is string => Boolean(id)),
    [items],
  );
  const activeId = useActiveHeading(ids);

  // Force a refresh after client-side TOC enrichment or hash scroll settles.
  useEffect(() => {
    const timers = [0, 120, 400].map((delay) =>
      window.setTimeout(() => {
        window.dispatchEvent(new Event(DOCS_MAIN_SCROLL_EVENT));
      }, delay),
    );
    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [pathname, ids]);

  const value = useMemo(() => ({ activeId }), [activeId]);

  return (
    <DocsTocScrollSpyContext.Provider value={value}>{children}</DocsTocScrollSpyContext.Provider>
  );
}

export function useDocsTocActiveId(href: string) {
  const { activeId } = useContext(DocsTocScrollSpyContext);
  const id = getHeadingId(href);
  return Boolean(id && activeId === id);
}

export function useDocsTocScrollSpyActiveId() {
  return useContext(DocsTocScrollSpyContext).activeId;
}
