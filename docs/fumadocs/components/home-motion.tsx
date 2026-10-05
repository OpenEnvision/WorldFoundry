'use client';

import { useLayoutEffect } from 'react';

const SELECTOR = '.wf-home-reveal, .pi-doc-welcome-workflow';

function supportsViewTimeline() {
  return typeof CSS !== 'undefined' && CSS.supports('animation-timeline: view()');
}

function reveal(el: Element) {
  el.removeAttribute('data-wf-pending');
  el.setAttribute('data-wf-inview', '');
}

export function HomeMotion() {
  useLayoutEffect(() => {
    const nodes = Array.from(document.querySelectorAll(SELECTOR));
    if (nodes.length === 0) return undefined;

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (reduce.matches || supportsViewTimeline()) {
      nodes.forEach(reveal);
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          reveal(entry.target);
          observer.unobserve(entry.target);
        }
      },
      { rootMargin: '0px 0px -8% 0px', threshold: 0.16 },
    );

    nodes.forEach((el) => {
      el.setAttribute('data-wf-pending', '');
      observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  return null;
}
