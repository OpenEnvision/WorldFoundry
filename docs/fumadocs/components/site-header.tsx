import Link from 'next/link';
import type { ReactNode } from 'react';

import { SiteGitHubLink } from '@/components/site-github-link';
import { SiteNav, type SiteNavItemId } from '@/components/site-nav';
import { SiteSearchTrigger } from '@/components/site-search-trigger';
import { SiteThemeSwitch } from '@/components/site-theme-switch';
import { WorldFoundryWordmarkLink } from '@/components/worldfoundry-wordmark';

export type SiteHeaderLanguageLink = {
  href: string;
  label: string;
  current?: boolean;
};

export type SiteHeaderProps = {
  variant: 'solid' | 'hero';
  active: SiteNavItemId;
  languageLinks: SiteHeaderLanguageLink[];
  languageAriaLabel?: string;
  navAriaLabel?: string;
  docsHref?: string;
  docsLabel?: string;
  homeLabel?: string;
  blogLabel?: string;
  eventsLabel?: string;
  communityLabel?: string;
  openEnvisionLabel?: string;
  searchLabel?: string;
  brandLeading?: ReactNode;
  beforeInner?: ReactNode;
  wordmarkClassName?: string;
  headerClassName?: string;
};

function joinClasses(...values: Array<string | false | undefined>) {
  return values.filter(Boolean).join(' ');
}

export function SiteHeader({
  variant,
  active,
  languageLinks,
  languageAriaLabel = 'Language',
  navAriaLabel,
  docsHref,
  docsLabel,
  homeLabel,
  blogLabel,
  eventsLabel,
  communityLabel,
  openEnvisionLabel,
  searchLabel,
  brandLeading,
  beforeInner,
  wordmarkClassName,
  headerClassName,
}: SiteHeaderProps) {
  const isHero = variant === 'hero';

  return (
    <header
      className={joinClasses(
        isHero ? 'wf-home-site-header' : 'pi-header pi-doc-header wf-home-site-header',
        headerClassName,
      )}
    >
      {beforeInner}
      <div
        className={joinClasses(
          'wf-home-site-header-inner',
          !isHero && 'pi-doc-header-inner',
          'flex flex-wrap items-center justify-between w-full',
        )}
      >
        <div className="pi-doc-header-brand">
          {brandLeading}
          <WorldFoundryWordmarkLink variant="compact" className={wordmarkClassName} />
        </div>
        <div className="wf-home-site-header-tools ml-auto">
          <SiteNav
            active={active}
            ariaLabel={navAriaLabel}
            docsHref={docsHref}
            docsLabel={docsLabel}
            homeLabel={homeLabel}
            blogLabel={blogLabel}
            eventsLabel={eventsLabel}
            communityLabel={communityLabel}
            openEnvisionLabel={openEnvisionLabel}
          />
          <SiteSearchTrigger label={searchLabel} />
          <SiteGitHubLink />
          <SiteThemeSwitch />
          <div className="pi-language-switch" aria-label={languageAriaLabel}>
            {languageLinks.map((item) => (
              <Link
                href={item.href}
                aria-current={item.current ? 'true' : undefined}
                key={`${item.href}-${item.label}`}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </header>
  );
}
