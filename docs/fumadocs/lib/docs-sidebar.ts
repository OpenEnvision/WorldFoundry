import apiReferenceMetaEn from '@/content/docs/api-reference/meta.json';
import apiReferenceMetaZh from '@/content/docs/api-reference/meta.zh.json';
import architectureMetaEn from '@/content/docs/maintainers/architecture/meta.json';
import architectureMetaZh from '@/content/docs/maintainers/architecture/meta.zh.json';
import metricsMetaEn from '@/content/docs/evaluation/metrics/meta.json';
import metricsMetaZh from '@/content/docs/evaluation/metrics/meta.zh.json';
import { docsNavGroups, getNavPageLabel } from '@/lib/docs-navigation';
import type { Locale } from '@/lib/i18n';
import {
  isSidebarItemActive,
  type SidebarNavItem,
  type SidebarNavPage,
  type SidebarPageLink,
} from '@/lib/docs-sidebar-shared';
import { source } from '@/lib/source';

export type {
  SidebarBadge,
  SidebarNavDivider,
  SidebarNavItem,
  SidebarNavPage,
  SidebarPageLink,
} from '@/lib/docs-sidebar-shared';
export { isSidebarItemActive };

type DocsPage = NonNullable<ReturnType<typeof source.getPage>>;

export type SidebarMetricsTree = {
  type: 'metrics-tree';
  hub: SidebarNavPage;
  items: SidebarNavItem[];
};

export type SidebarArchitectureTree = {
  type: 'architecture-tree';
  hub: SidebarNavPage;
  items: SidebarNavItem[];
};

export type SidebarApiTree = {
  type: 'api-tree';
  hub: SidebarNavPage;
  items: SidebarNavItem[];
};

export type SidebarNavEntry =
  | SidebarNavItem
  | SidebarMetricsTree
  | SidebarArchitectureTree
  | SidebarApiTree;

export type SidebarNavGroup = {
  id: (typeof docsNavGroups)[number]['id'];
  items: SidebarNavEntry[];
};

function toSidebarPageLink(page: DocsPage, locale: Locale): SidebarPageLink {
  return {
    url: page.url,
    label: getNavPageLabel(page.slugs, locale, page.data.title),
  };
}

function isMetaSeparator(entry: string) {
  return entry.startsWith('---') && entry.endsWith('---');
}

function parseMetaSeparator(entry: string) {
  return entry.slice(3, -3).trim();
}

function getMetricsMeta(locale: Locale) {
  return locale === 'zh' ? metricsMetaZh : metricsMetaEn;
}

function getArchitectureMeta(locale: Locale) {
  return locale === 'zh' ? architectureMetaZh : architectureMetaEn;
}

function getArchitectureSidebarItems(locale: Locale): SidebarNavItem[] {
  const items: SidebarNavItem[] = [];

  for (const entry of getArchitectureMeta(locale).pages) {
    if (entry === 'index') continue;

    if (isMetaSeparator(entry)) {
      items.push({
        type: 'divider',
        label: parseMetaSeparator(entry),
      });
      continue;
    }

    const page = source.getPage(['maintainers', 'architecture', entry], locale);
    if (!page) continue;

    items.push({
      type: 'page',
      link: toSidebarPageLink(page, locale),
      depth: 2,
      badges: [],
    });
  }

  return items;
}

function getMetricsSidebarItems(locale: Locale): SidebarNavItem[] {
  const items: SidebarNavItem[] = [];

  for (const entry of getMetricsMeta(locale).pages) {
    if (entry === 'index') continue;

    if (isMetaSeparator(entry)) {
      items.push({
        type: 'divider',
        label: parseMetaSeparator(entry),
      });
      continue;
    }

    const page = source.getPage(['evaluation', 'metrics', entry], locale);
    if (!page) continue;

    items.push({
      type: 'page',
      link: toSidebarPageLink(page, locale),
      depth: 2,
      badges: [],
    });
  }

  return items;
}

function getApiReferenceMeta(locale: Locale) {
  return locale === 'zh' ? apiReferenceMetaZh : apiReferenceMetaEn;
}

function getApiReferenceSidebarItems(locale: Locale): SidebarNavItem[] {
  const items: SidebarNavItem[] = [];

  for (const entry of getApiReferenceMeta(locale).pages) {
    if (entry === 'index') continue;

    const page = source.getPage(['api-reference', entry], locale);
    if (!page) continue;

    items.push({
      type: 'page',
      link: toSidebarPageLink(page, locale),
      depth: 2,
      badges: [],
    });
  }

  return items;
}

export function getDocsSidebarGroups(locale: Locale): SidebarNavGroup[] {
  const metricsChildren = getMetricsSidebarItems(locale);
  const architectureChildren = getArchitectureSidebarItems(locale);
  const apiChildren = getApiReferenceSidebarItems(locale);

  return docsNavGroups
    .map((group) => {
      const items: SidebarNavEntry[] = [];

      for (const slugs of group.slugs) {
        const page = source.getPage([...slugs], locale);
        if (!page) continue;

        if (slugs.join('/') === 'api-reference') {
          items.push({
            type: 'api-tree',
            hub: {
              type: 'page',
              link: toSidebarPageLink(page, locale),
              depth: 0,
              badges: [],
            },
            items: apiChildren,
          });
          continue;
        }

        if (slugs[0] === 'api-reference') {
          // Children are rendered inside api-tree.
          continue;
        }

        if (slugs.join('/') === 'evaluation/metrics') {
          items.push({
            type: 'metrics-tree',
            hub: {
              type: 'page',
              link: toSidebarPageLink(page, locale),
              depth: 0,
              badges: [],
            },
            items: metricsChildren,
          });
          continue;
        }

        if (slugs.join('/') === 'maintainers/architecture') {
          items.push({
            type: 'architecture-tree',
            hub: {
              type: 'page',
              link: toSidebarPageLink(page, locale),
              depth: 0,
              badges: [],
            },
            items: architectureChildren,
          });
          continue;
        }

        const nested = page.slugs.length > 2;
        items.push({
          type: 'page',
          link: toSidebarPageLink(page, locale),
          depth: nested ? 1 : 0,
          badges: [],
        });
      }

      return { id: group.id, items };
    })
    .filter((group) => group.items.length > 0);
}

export function isMetricsSidebarOpen(slugs: readonly string[]) {
  return slugs[0] === 'evaluation' && slugs[1] === 'metrics';
}

export function isArchitectureSidebarOpen(slugs: readonly string[]) {
  return slugs[0] === 'maintainers' && slugs[1] === 'architecture';
}

/** Open only on nested API pages; keep the overview collapsed by default. */
export function isApiReferenceSidebarOpen(slugs: readonly string[]) {
  return slugs[0] === 'api-reference' && slugs.length > 1;
}

export function isSidebarGroupActive(group: SidebarNavGroup, currentUrl: string) {
  return group.items.some((item) => {
    if (
      item.type === 'metrics-tree' ||
      item.type === 'architecture-tree' ||
      item.type === 'api-tree'
    ) {
      if (item.hub.link.url === currentUrl) return true;
      return item.items.some((child) => child.type === 'page' && child.link.url === currentUrl);
    }

    if (item.type !== 'page') return false;
    if (item.link.url === currentUrl) return true;
    return (
      item.link.url.endsWith('/evaluation/benchmark-hub') &&
      currentUrl.startsWith(`${item.link.url}/`)
    );
  });
}
