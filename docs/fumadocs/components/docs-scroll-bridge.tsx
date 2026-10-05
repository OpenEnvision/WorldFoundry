'use client';

import { DOCS_MAIN_SCROLL_EVENT } from '@/lib/docs-toc-scroll-spy';
import { usePathname } from 'next/navigation';
import { useEffect } from 'react';

function notifyMainScroll(main: HTMLElement) {
  main.dispatchEvent(new Event('scroll', { bubbles: false }));
  window.dispatchEvent(new Event(DOCS_MAIN_SCROLL_EVENT));
}

function findScrollContainer(node: Node | null): HTMLElement | null {
  let current: Node | null = node;
  while (current && current instanceof HTMLElement) {
    const overflowY = getComputedStyle(current).overflowY;
    const canScroll =
      (overflowY === 'auto' || overflowY === 'scroll') &&
      current.scrollHeight > current.clientHeight + 1;
    if (canScroll) return current;
    current = current.parentElement;
  }
  return null;
}

function forwardWheelToMain(panel: HTMLElement, main: HTMLElement, event: WheelEvent) {
  // The sidebar/right-rail panels are `overflow: hidden` flex columns; the real
  // scroll happens on an inner container (e.g. `.pi-doc-list`). Forwarding to the
  // panel itself made `canScroll` always false, which preventDefault'd the wheel
  // and starved the inner list. Resolve the actual scrollable ancestor instead.
  const scroller = findScrollContainer(event.target instanceof Node ? event.target : null);

  if (!scroller) {
    main.scrollTop += event.deltaY;
    notifyMainScroll(main);
    event.preventDefault();
    return;
  }

  const atTop = scroller.scrollTop <= 0;
  const atBottom = scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 1;

  // Only hand the wheel to the main column once the inner list has reached an edge.
  if ((event.deltaY < 0 && atTop) || (event.deltaY > 0 && atBottom)) {
    main.scrollTop += event.deltaY;
    notifyMainScroll(main);
    event.preventDefault();
  }
}

function scrollMainToHash(main: HTMLElement, hash: string) {
  const id = decodeURIComponent(hash.replace(/^#/, ''));
  if (!id) return;

  const target = document.getElementById(id);
  if (!target || !main.contains(target)) return;

  const offset = 16;
  const mainRect = main.getBoundingClientRect();
  const targetRect = target.getBoundingClientRect();
  main.scrollTop += targetRect.top - mainRect.top - offset;
}

export function DocsScrollBridge() {
  const pathname = usePathname();

  useEffect(() => {
    const shell = document.querySelector<HTMLElement>('.pi-doc-shell');
    const main = document.querySelector<HTMLElement>('.pi-doc-main');
    if (!shell || !main) return;

    const panels = Array.from(
      document.querySelectorAll<HTMLElement>('.pi-doc-sidebar, .pi-doc-right-rail'),
    );

    const onShellWheel = (event: WheelEvent) => {
      if (!(event.target instanceof Node) || main.contains(event.target)) {
        return;
      }

      for (const panel of panels) {
        if (panel.contains(event.target)) {
          forwardWheelToMain(panel, main, event);
          return;
        }
      }

      main.scrollTop += event.deltaY;
      notifyMainScroll(main);
      event.preventDefault();
    };

    const onAnchorClick = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Element)) return;

      const anchor = target.closest<HTMLAnchorElement>('a[href^="#"]');
      if (!anchor || !shell.contains(anchor)) return;

      const hash = anchor.getAttribute('href');
      if (!hash || hash === '#') return;

      event.preventDefault();
      history.pushState(null, '', hash);
      scrollMainToHash(main, hash);
      notifyMainScroll(main);
    };

    const onHashChange = () => {
      scrollMainToHash(main, window.location.hash);
      notifyMainScroll(main);
    };

    const panelCleanups = panels.map((panel) => {
      const onPanelWheel = (event: WheelEvent) => forwardWheelToMain(panel, main, event);
      panel.addEventListener('wheel', onPanelWheel, { passive: false });
      return () => panel.removeEventListener('wheel', onPanelWheel);
    });

    shell.addEventListener('wheel', onShellWheel, { passive: false });
    shell.addEventListener('click', onAnchorClick);
    window.addEventListener('hashchange', onHashChange);

    if (window.location.hash) {
      requestAnimationFrame(() => {
        scrollMainToHash(main, window.location.hash);
        notifyMainScroll(main);
        window.setTimeout(() => notifyMainScroll(main), 120);
        window.setTimeout(() => notifyMainScroll(main), 400);
      });
    } else {
      main.scrollTop = 0;
      notifyMainScroll(main);
    }

    return () => {
      shell.removeEventListener('wheel', onShellWheel);
      shell.removeEventListener('click', onAnchorClick);
      window.removeEventListener('hashchange', onHashChange);
      panelCleanups.forEach((cleanup) => cleanup());
    };
  }, [pathname]);

  return null;
}
