import { getPageMarkdownUrl, source } from '@/lib/source';
import { SiteSearchTrigger } from '@/components/site-search-trigger';
import { SiteNav } from '@/components/site-nav';
import { notFound } from 'next/navigation';
import { getMDXComponents } from '@/components/mdx';
import { createRelativeLink } from 'fumadocs-ui/mdx';
import { getBenchmarkBadges } from '@/lib/benchmark-catalog';
import { getDocsBreadcrumbs } from '@/lib/docs-breadcrumb';
import {
  docsLabels,
  isBenchmarkHubDocsPage,
  isTableDenseDocsPage,
} from '@/lib/docs-navigation';
import { BenchmarkBadge } from '@/components/benchmark-badge';
import { DocsBreadcrumb } from '@/components/docs-breadcrumb';
import { DocsPagination } from '@/components/docs-pagination';
import { DocsRelatedLinks } from '@/components/docs-related-links';
import { DocsInlineToc } from '@/components/docs-inline-toc';
import { DocsMobileNavToggle } from '@/components/docs-mobile-nav';
import { DocsPageActions } from '@/components/docs-page-actions';
import { DocsReadingProgress } from '@/components/docs-reading-progress';
import { DocsScrollBridge } from '@/components/docs-scroll-bridge';
import { DocsSidebarApiTree } from '@/components/docs-sidebar-api-tree';
import { DocsSidebarArchitectureTree } from '@/components/docs-sidebar-architecture-tree';
import { DocsSidebarMetricsTree } from '@/components/docs-sidebar-metrics-tree';
import { DocsTocRail } from '@/components/docs-toc-rail';
import { DocsWelcomeHero } from '@/components/docs-welcome-hero';
import { WorldFoundryWordmarkLink } from '@/components/worldfoundry-wordmark';
import { getDocsLastUpdated } from '@/lib/docs-last-updated';
import { getDocsPagination } from '@/lib/docs-pagination';
import { getDocsRelatedLinks } from '@/lib/docs-related-links';
import {
  getDocsSidebarGroups,
  isApiReferenceSidebarOpen,
  isArchitectureSidebarOpen,
  isMetricsSidebarOpen,
  isSidebarGroupActive,
  isSidebarItemActive,
} from '@/lib/docs-sidebar';
import { gitConfig } from '@/lib/shared';
import Link from 'next/link';
import { defaultLocale, i18n, isLocale, localeNames, type Locale } from '@/lib/i18n';

function normalizeLocale(locale: string): Locale {
  if (!isLocale(locale)) notFound();
  return locale;
}

function getPageUrl(slugs: string[], locale: Locale) {
  const page = source.getPage(slugs, locale);

  if (page) {
    return page.url;
  }

  const path = slugs.length > 0 ? `/${slugs.join('/')}` : '';
  return locale === defaultLocale ? `/docs${path}` : `/${locale}/docs${path}`;
}

function getNavGroups(locale: Locale) {
  return getDocsSidebarGroups(locale);
}

export async function DocsPage({ slug, locale }: { slug: string[] | undefined; locale: string }) {
  const normalized = normalizeLocale(locale);
  const page = source.getPage(slug, normalized);
  if (!page) notFound();

  const t = docsLabels[normalized];
  const MDX = page.data.body;
  const markdownUrl = getPageMarkdownUrl(page).url;
  const sidebarGroups = getNavGroups(normalized);
  const githubUrl = `https://github.com/${gitConfig.user}/${gitConfig.repo}/blob/${gitConfig.branch}/docs/fumadocs/content/docs/${page.path}`;
  const docsHref = getPageUrl([], normalized);
  const benchmarkHubPage = isBenchmarkHubDocsPage(page.slugs);
  const usesWideTableLayout = isTableDenseDocsPage(page.slugs) || benchmarkHubPage;

  const toc = page.data.toc ?? [];
  const pagination = getDocsPagination(page.slugs, normalized);
  const relatedLinks = getDocsRelatedLinks(page.slugs, normalized);
  const breadcrumbs = getDocsBreadcrumbs(page.slugs, normalized, page.data.title);
  const pageBadges =
    page.slugs[0] === 'evaluation' && page.slugs[1] === 'benchmark-hub' && page.slugs.length === 3
      ? getBenchmarkBadges(page.slugs[2])
      : [];
  const lastUpdated = getDocsLastUpdated(page.path, normalized);
  const isWelcomePage = page.slugs.length === 0;
  const showToc = toc.length > 0;

  return (
    <main
      className={[
        'pi-doc-shell',
        isWelcomePage ? 'pi-doc-shell-welcome' : '',
        usesWideTableLayout ? 'pi-doc-shell-table-wide' : '',
        usesWideTableLayout && showToc ? 'pi-doc-shell-table-wide-has-toc' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      lang={normalized}
    >
      <DocsScrollBridge />
      <header className="pi-header pi-doc-header">
        <DocsReadingProgress />
        <div className="pi-doc-header-inner flex flex-wrap items-center justify-between w-full">
          <div className="pi-doc-header-brand">
            <DocsMobileNavToggle openLabel={t.openMenu} closeLabel={t.closeMenu} />
            <WorldFoundryWordmarkLink variant="compact" />
          </div>
          <div className="pi-doc-header-tools ml-auto">
            <SiteNav
              active={
                page.slugs[0] === 'guides' && page.slugs[1] === 'supported-models'
                  ? 'models'
                  : page.slugs[0] === 'evaluation' && page.slugs[1] === 'benchmark-hub'
                    ? 'benchmarks'
                    : 'docs'
              }
              ariaLabel={t.nav}
              docsHref={docsHref}
              docsLabel={t.docs}
              homeLabel={t.home}
              openEnvisionLabel={t.openEnvision}
            />
            <SiteSearchTrigger />
            <div className="pi-language-switch" aria-label={t.language}>
              {i18n.languages.map((item) => (
                <Link
                  href={getPageUrl(page.slugs, item)}
                  aria-current={item === normalized ? 'true' : undefined}
                  key={item}
                >
                  {localeNames[item]}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </header>

      <div className="pi-doc-frame">
        <aside className="pi-doc-sidebar" id="pi-doc-sidebar" aria-label={t.sidebar}>
          <p className="pi-doc-sidebar-title">{t.sidebar}</p>
          <nav className="pi-doc-list">
            {sidebarGroups.map((group) => {
              const groupActive = isSidebarGroupActive(group, page.url);

              return (
                <div
                  className={['pi-doc-nav-group', groupActive ? 'pi-doc-nav-group-active' : '']
                    .filter(Boolean)
                    .join(' ')}
                  key={group.id}
                >
                  <p className="pi-doc-section-title">
                    <span>{t.navGroups[group.id]}</span>
                  </p>
                  {group.items.map((item) => {
                    if (item.type === 'api-tree') {
                      return (
                        <DocsSidebarApiTree
                          hub={item.hub}
                          items={item.items}
                          currentUrl={page.url}
                          locale={normalized}
                          defaultOpen={isApiReferenceSidebarOpen(page.slugs)}
                          expandLabel={t.expandApiList}
                          collapseLabel={t.collapseApiList}
                          key="api-tree"
                        />
                      );
                    }

                    if (item.type === 'metrics-tree') {
                      return (
                        <DocsSidebarMetricsTree
                          hub={item.hub}
                          items={item.items}
                          currentUrl={page.url}
                          locale={normalized}
                          defaultOpen={isMetricsSidebarOpen(page.slugs)}
                          expandLabel={t.expandMetricsList}
                          collapseLabel={t.collapseMetricsList}
                          key="metrics-tree"
                        />
                      );
                    }

                    if (item.type === 'architecture-tree') {
                      return (
                        <DocsSidebarArchitectureTree
                          hub={item.hub}
                          items={item.items}
                          currentUrl={page.url}
                          locale={normalized}
                          defaultOpen={isArchitectureSidebarOpen(page.slugs)}
                          expandLabel={t.expandArchitectureList}
                          collapseLabel={t.collapseArchitectureList}
                          key="architecture-tree"
                        />
                      );
                    }

                    if (item.type === 'divider') {
                      return (
                        <p className="pi-doc-sidebar-divider" key={`divider-${item.label}`}>
                          {item.label}
                        </p>
                      );
                    }

                    const active = isSidebarItemActive(item, page.url);
                    const depthClass =
                      item.depth === 2
                        ? 'pi-doc-link-deep'
                        : item.depth === 1
                          ? 'pi-doc-link-child'
                          : '';

                    return (
                      <Link
                        href={item.link.url}
                        className={['pi-doc-link', depthClass, active ? 'pi-doc-link-active' : '']
                          .filter(Boolean)
                          .join(' ')}
                        aria-current={active ? 'page' : undefined}
                        key={item.link.url}
                      >
                        <span className="pi-doc-link-row">
                          <span className="pi-doc-link-title">{item.link.label}</span>
                          {item.badges.length > 0 ? (
                            <span className="pi-doc-link-badges">
                              {item.badges.map((badge) => (
                                <BenchmarkBadge kind={badge.kind} locale={normalized} key={badge.kind} />
                              ))}
                            </span>
                          ) : null}
                        </span>
                      </Link>
                    );
                  })}
                </div>
              );
            })}
          </nav>
        </aside>

        <div className="pi-doc-main">
          <article className="pi-doc-article">
            {isWelcomePage ? (
              <>
                <DocsWelcomeHero
                  description={page.data.description ?? ''}
                  locale={normalized}
                />
                <div className="pi-doc-article-inner pi-doc-welcome-body">
                  <DocsPageActions
                    editLabel={t.editPage}
                    githubUrl={githubUrl}
                    lastUpdated={lastUpdated}
                    lastUpdatedLabel={t.lastUpdated}
                    markdownLabel={t.markdown}
                    markdownUrl={markdownUrl}
                  />

                  {showToc ? (
                    <DocsInlineToc items={toc} pageKey={page.url} slugs={page.slugs} title={t.onThisPage} />
                  ) : null}

                  <div className="pi-doc-content">
                    <MDX
                      components={getMDXComponents({
                        a: createRelativeLink(source, page),
                      })}
                    />
                  </div>

                  <DocsPagination
                    next={pagination.next}
                    nextLabel={t.nextPage}
                    prev={pagination.prev}
                    previousLabel={t.previousPage}
                  />
                </div>
              </>
            ) : (
              <div className="pi-doc-article-inner">
                {breadcrumbs.length > 1 ? <DocsBreadcrumb items={breadcrumbs} /> : null}

                <div className="pi-doc-title-row">
                  <h1>{page.data.title}</h1>
                  {pageBadges.length > 0 ? (
                    <div className="pi-doc-title-badges">
                      {pageBadges.map((kind) => (
                        <BenchmarkBadge kind={kind} locale={normalized} key={kind} />
                      ))}
                    </div>
                  ) : null}
                </div>

                {page.data.description ? (
                  <p className="pi-doc-description">{page.data.description}</p>
                ) : null}

                <DocsPageActions
                  editLabel={t.editPage}
                  githubUrl={githubUrl}
                  lastUpdated={lastUpdated}
                  lastUpdatedLabel={t.lastUpdated}
                  markdownLabel={t.markdown}
                  markdownUrl={markdownUrl}
                />

                {showToc ? (
                  <DocsInlineToc items={toc} pageKey={page.url} slugs={page.slugs} title={t.onThisPage} />
                ) : null}

                <div className="pi-doc-content">
                  <MDX
                    components={getMDXComponents({
                      a: createRelativeLink(source, page),
                    })}
                  />
                </div>

                <DocsRelatedLinks links={relatedLinks} title={t.relatedPages} />

                <DocsPagination
                  next={pagination.next}
                  nextLabel={t.nextPage}
                  prev={pagination.prev}
                  previousLabel={t.previousPage}
                />
              </div>
            )}
          </article>
        </div>

        {showToc ? (
          <DocsTocRail key={page.url} items={toc} pageKey={page.url} slugs={page.slugs} title={t.onThisPage} />
        ) : null}
      </div>
    </main>
  );
}
