import Link from 'next/link';
import type { ReactNode } from 'react';

import { SiteNav } from '@/components/site-nav';
import { SiteSearchTrigger } from '@/components/site-search-trigger';
import { WorldFoundryWordmarkLink } from '@/components/worldfoundry-wordmark';
import { WORLDFOUNDRY_GITHUB_REPO } from '@/lib/site-links';

type BlogShellProps = {
  children: ReactNode;
  footerLabel?: string;
};

export function BlogShell({ children, footerLabel = 'Blog' }: BlogShellProps) {
  return (
    <main className="pi-home-shell wf-home-shell">
      <header className="pi-header pi-doc-header wf-home-site-header">
        <div className="pi-doc-header-inner flex flex-wrap items-center justify-between w-full">
          <div className="pi-doc-header-brand">
            <WorldFoundryWordmarkLink variant="compact" />
          </div>
          <div className="pi-doc-header-tools ml-auto">
            <SiteNav active="blog" />
            <SiteSearchTrigger />
            <div className="pi-language-switch" aria-label="Language">
              <Link href="/blog" aria-current="true">
                English
              </Link>
              <Link href="/zh/docs">中文</Link>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8 md:py-12">
        {children}

        <footer className="pi-footer">
          <p>{footerLabel}</p>
          <div>
            <Link href="/docs">Docs</Link>
            <a href={WORLDFOUNDRY_GITHUB_REPO} rel="noreferrer" target="_blank">
              Community
            </a>
          </div>
        </footer>
      </div>
    </main>
  );
}
