import Link from 'next/link';

import { WorldFoundryWordmarkLink } from '@/components/worldfoundry-wordmark';
import {
  OPENENVISION_AWESOME_WORLD_MODELING,
  OPENENVISION_BLOGXIV_SITE,
  OPENENVISION_ORG,
  WORLDFOUNDRY_GITHUB_DISCUSSIONS,
  WORLDFOUNDRY_GITHUB_ISSUES,
  WORLDFOUNDRY_GITHUB_REPO,
  WORLDFOUNDRY_SLACK_INVITE,
} from '@/lib/site-links';

type FooterLink = {
  label: string;
  href: string;
  external?: boolean;
};

type FooterColumn = {
  title: string;
  links: FooterLink[];
};

const footerColumns: FooterColumn[] = [
  {
    title: 'Documentation',
    links: [
      { label: 'Introduction', href: '/docs' },
      { label: 'Quickstart', href: '/docs/quickstart' },
      { label: 'Model catalog', href: '/docs/guides/supported-models' },
      { label: 'Benchmark Hub', href: '/docs/evaluation/benchmark-hub' },
      { label: 'API reference', href: '/docs/api-reference' },
    ],
  },
  {
    title: 'Product',
    links: [
      { label: 'Studio', href: '/docs/guides/studio' },
      { label: 'CLI', href: '/docs/reference/cli' },
      { label: 'TUI', href: '/docs/guides/tui' },
      { label: 'Run inference', href: '/docs/guides/inference' },
      { label: 'Evaluation', href: '/docs/evaluation' },
    ],
  },
  {
    title: 'Community',
    links: [
      { label: 'Slack', href: WORLDFOUNDRY_SLACK_INVITE, external: true },
      { label: 'GitHub Issues', href: WORLDFOUNDRY_GITHUB_ISSUES, external: true },
      { label: 'GitHub Discussions', href: WORLDFOUNDRY_GITHUB_DISCUSSIONS, external: true },
      { label: 'Contributing', href: '/docs/maintainers/contributing' },
    ],
  },
  {
    title: 'Ecosystem',
    links: [
      { label: 'OpenEnvision', href: '/openenvision' },
      { label: 'Blog', href: '/blog' },
      { label: 'Events', href: '/events' },
      { label: 'Awesome World Modeling', href: OPENENVISION_AWESOME_WORLD_MODELING, external: true },
      { label: 'BlogrXiv', href: OPENENVISION_BLOGXIV_SITE, external: true },
    ],
  },
];

function FooterLinkItem({ label, href, external }: FooterLink) {
  if (external) {
    return (
      <a href={href} target="_blank" rel="noreferrer">
        {label}
      </a>
    );
  }

  return <Link href={href}>{label}</Link>;
}

export function SiteFooter() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="wf-site-footer" aria-label="Site footer">
      <div className="wf-site-footer-inner">
        <div className="wf-site-footer-meta">
          <WorldFoundryWordmarkLink variant="compact" className="wf-site-footer-wordmark" />
          <p className="wf-site-footer-copyright">
            Copyright © 2024–{currentYear}{' '}
            <a href={OPENENVISION_ORG} target="_blank" rel="noreferrer">
              OpenEnvision
            </a>
            . All rights reserved.
          </p>
          <p className="wf-site-footer-tagline">
            Unified world model inference and evaluation infrastructure.
          </p>
        </div>

        <div className="wf-site-footer-columns">
          {footerColumns.map((column) => (
            <div className="wf-site-footer-column" key={column.title}>
              <h2>{column.title}</h2>
              <ul>
                {column.links.map((link) => (
                  <li key={link.href}>
                    <FooterLinkItem {...link} />
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </footer>
  );
}
