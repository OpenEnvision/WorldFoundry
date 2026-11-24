'use client';

import { type CSSProperties, type MouseEvent, useEffect, useRef, useState } from 'react';

type HomeSection = {
  id: string;
  label: string;
};

// Anchors map to the `aria-labelledby` heading ids already rendered by the
// landing sections, so the rail needs no markup changes to stay in sync.
const sections: HomeSection[] = [
  { id: 'wf-configure-title', label: 'Configure a run' },
  { id: 'wf-pillars-title', label: 'Why WorldFoundry' },
  { id: 'wf-catalog-title', label: 'Catalog' },
  { id: 'wf-taxonomy-title', label: 'Capability progression' },
  { id: 'wf-capabilities-title', label: 'End to end' },
  { id: 'wf-workflow-title', label: 'How it works' },
  { id: 'wf-community-title', label: 'Community' },
  { id: 'wf-resources-title', label: 'Resources' },
];

export function HomeSectionNav() {
  const [active, setActive] = useState(0);
  const [visible, setVisible] = useState(false);
  const frame = useRef(0);

  useEffect(() => {
    const main = document.querySelector<HTMLElement>('.wf-home-main');
    const targets = sections.map((section) => document.getElementById(section.id));

    const measure = () => {
      frame.current = 0;
      const viewport = window.innerHeight || 1;

      // Show once the content region clears the hero, hide again near the footer.
      if (main) {
        const rect = main.getBoundingClientRect();
        setVisible(rect.top <= viewport * 0.35 && rect.bottom >= viewport * 0.45);
      }

      // Active = last heading that has crossed the activation line.
      const line = viewport * 0.33;
      let next = 0;
      for (let index = 0; index < targets.length; index += 1) {
        const element = targets[index];
        if (element && element.getBoundingClientRect().top - line <= 0) next = index;
      }
      setActive(next);
    };

    const schedule = () => {
      if (frame.current) return;
      frame.current = window.requestAnimationFrame(measure);
    };

    measure();
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    return () => {
      if (frame.current) window.cancelAnimationFrame(frame.current);
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
    };
  }, []);

  const handleClick = (event: MouseEvent<HTMLAnchorElement>, id: string) => {
    const element = document.getElementById(id);
    if (!element) return;
    event.preventDefault();
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    element.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
    window.history.replaceState(null, '', `#${id}`);
  };

  const progress = sections.length > 1 ? active / (sections.length - 1) : 0;

  return (
    <nav className="wf-home-secnav" aria-label="On this page" data-visible={visible ? 'true' : 'false'}>
      <ul className="wf-home-secnav-list" style={{ '--wf-secnav-progress': progress } as CSSProperties}>
        {sections.map((section, index) => {
          const state = index === active ? 'active' : index < active ? 'past' : 'upcoming';
          return (
            <li className="wf-home-secnav-item" key={section.id}>
              <a
                className="wf-home-secnav-link"
                href={`#${section.id}`}
                data-state={state}
                aria-current={index === active ? 'true' : undefined}
                onClick={(event) => handleClick(event, section.id)}
              >
                <span className="wf-home-secnav-dot" aria-hidden="true" />
                <span className="wf-home-secnav-label">{section.label}</span>
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
