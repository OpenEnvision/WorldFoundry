'use client';

import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

type DocsMobileNavProps = {
  openLabel: string;
  closeLabel: string;
};

function getShell() {
  return document.querySelector<HTMLElement>('.pi-doc-shell');
}

export function DocsMobileNavToggle({ openLabel, closeLabel }: DocsMobileNavProps) {
  const [open, setOpen] = useState(false);
  const [portalRoot, setPortalRoot] = useState<HTMLElement | null>(null);

  useEffect(() => {
    const shell = getShell();
    const header = shell?.querySelector<HTMLElement>('.pi-doc-header');
    if (!shell || !header) return;
    setPortalRoot(shell);
    const measure = () => {
      shell.style.setProperty('--pi-doc-header-height', `${header.getBoundingClientRect().height}px`);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(header);
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    const desktop = window.matchMedia('(min-width: 901px)');
    const onResize = () => { if (desktop.matches) setOpen(false); };
    desktop.addEventListener('change', onResize);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      observer.disconnect();
      shell.style.removeProperty('--pi-doc-header-height');
      desktop.removeEventListener('change', onResize);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, []);

  useEffect(() => {
    const shell = getShell();
    if (!shell) return;
    const previousOverflow = document.body.style.overflow;

    if (open) {
      shell.classList.add('pi-doc-nav-open');
      document.body.style.overflow = 'hidden';
    } else {
      shell.classList.remove('pi-doc-nav-open');
      document.body.style.overflow = previousOverflow;
    }

    return () => {
      shell.classList.remove('pi-doc-nav-open');
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  useEffect(() => {
    const shell = getShell();
    if (!shell) return;

    const close = () => setOpen(false);
    shell.querySelectorAll<HTMLAnchorElement>('.pi-doc-sidebar a').forEach((link) => {
      link.addEventListener('click', close);
    });

    return () => {
      shell.querySelectorAll<HTMLAnchorElement>('.pi-doc-sidebar a').forEach((link) => {
        link.removeEventListener('click', close);
      });
    };
  }, [open]);

  return (
    <>
      <button
        type="button"
        className="pi-doc-menu-button"
        aria-expanded={open}
        aria-controls="pi-doc-sidebar"
        onClick={() => setOpen((value) => !value)}
      >
        {open ? closeLabel : openLabel}
      </button>
      {open && portalRoot ? createPortal(
        <button
          type="button"
          className="pi-doc-nav-backdrop"
          aria-label={closeLabel}
          onClick={() => setOpen(false)}
        />,
        portalRoot,
      ) : null}
    </>
  );
}
