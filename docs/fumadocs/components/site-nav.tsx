import Link from 'next/link';

import { WORLDFOUNDRY_SLACK_INVITE } from '@/lib/site-links';

export type SiteNavItemId =
  | 'home'
  | 'docs'
  | 'models'
  | 'benchmarks'
  | 'blog'
  | 'events'
  | 'community'
  | 'openenvision';

type SiteNavItem = {
  id: SiteNavItemId;
  href: string;
  label: string;
  external?: boolean;
};

type SiteNavProps = {
  active: SiteNavItemId;
  ariaLabel?: string;
  className?: string;
  docsHref?: string;
  docsLabel?: string;
  homeLabel?: string;
  blogLabel?: string;
  eventsLabel?: string;
  communityLabel?: string;
  openEnvisionLabel?: string;
};

export function SiteNav({
  active,
  ariaLabel = 'Main navigation',
  className = 'pi-nav',
  docsHref = '/docs',
  docsLabel = 'Docs',
  homeLabel = 'Home',
  blogLabel = 'Blog',
  eventsLabel = 'Events',
  communityLabel = 'Community',
  openEnvisionLabel = 'OpenEnvision',
}: SiteNavProps) {
  const items: SiteNavItem[] = [
    { id: 'home', href: '/', label: homeLabel },
    { id: 'docs', href: docsHref, label: docsLabel },
    { id: 'blog', href: '/blog', label: blogLabel },
    { id: 'events', href: '/events', label: eventsLabel },
    {
      id: 'community',
      href: WORLDFOUNDRY_SLACK_INVITE,
      label: communityLabel,
      external: true,
    },
    { id: 'openenvision', href: '/openenvision', label: openEnvisionLabel },
  ];

  return (
    <nav className={className} aria-label={ariaLabel}>
      {items.map((item) =>
        item.external ? (
          <a href={item.href} key={item.id} rel="noreferrer" target="_blank">
            {item.label}
          </a>
        ) : (
          <Link href={item.href} aria-current={active === item.id ? 'page' : undefined} key={item.id}>
            {item.label}
          </Link>
        ),
      )}
    </nav>
  );
}
