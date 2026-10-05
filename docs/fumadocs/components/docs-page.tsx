import { getPageMarkdownUrl, source } from '@/lib/source';
import { SiteHeader } from '@/components/site-header';
import { notFound, redirect } from 'next/navigation';
import { getMDXComponents } from '@/components/mdx';
import { ModelPageFigure, ModelPageNoDemoVideo } from '@/components/docs-video';
import { enrichApiReferenceToc, isApiReferenceDocsPage } from '@/lib/docs-api-toc';
import { enrichBenchmarkToc } from '@/lib/docs-benchmark-toc';
import { createRelativeLink } from 'fumadocs-ui/mdx';
import { getBenchmarkCatalogEntry } from '@/lib/benchmark-catalog';
import { getModelRecipeIndexEntry } from '@/lib/model-recipe-index';
import { BenchmarkBadge } from '@/components/benchmark-badge';
import { BenchmarkIdentityMark } from '@/components/benchmark-identity-mark';
import { ModelIdentityMark } from '@/components/model-identity-mark';
import { ModelPaperThumb } from '@/components/model-paper-thumb';
import { getModelPaperThumbSrc } from '@/lib/model-arch-diagrams';
import { getDocsBreadcrumbs } from '@/lib/docs-breadcrumb';
import { movedDocsDestination } from '@/lib/docs-moved';
import { docsLabels } from '@/lib/docs-navigation';
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
import { DocsSidebarIcon } from '@/components/docs-sidebar-icon';
import { DocsSidebarMetricsTree } from '@/components/docs-sidebar-metrics-tree';
import { DocsTocRail } from '@/components/docs-toc-rail';
import { MetricQuickNav } from '@/components/metric-quick-nav';
import { DocsWelcomeHero } from '@/components/docs-welcome-hero';
import { getDocsLastUpdated } from '@/lib/docs-last-updated';
import { getDocsPagination } from '@/lib/docs-pagination';
import { getDocsRelatedLinks } from '@/lib/docs-related-links';
import { resolveMetricQuickNavFromSlugs } from '@/lib/metric-quick-nav';
import {
  getDocsSidebarGroups,
  isApiReferenceSidebarOpen,
  isArchitectureSidebarOpen,
  isMetricsSidebarOpen,
  isSidebarGroupActive,
  isSidebarItemActive,
} from '@/lib/docs-sidebar';
import { getGithubContentUrls } from '@/lib/shared';
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

function redirectMovedDocsPage(slugs: string[] | undefined, locale: Locale) {
  const destination = movedDocsDestination(slugs, locale);
  if (destination) {
    redirect(destination);
  }
}

export async function DocsPage({ slug, locale }: { slug: string[] | undefined; locale: string }) {
  const normalized = normalizeLocale(locale);
  redirectMovedDocsPage(slug, normalized);
  const page = source.getPage(slug, normalized);
  if (!page) notFound();

  const t = docsLabels[normalized];
  const MDX = page.data.body;
  const markdownUrl = getPageMarkdownUrl(page).url;
  const sidebarGroups = getNavGroups(normalized);
  const githubUrl = getGithubContentUrls(page.path).blob;
  const docsHref = getPageUrl([], normalized);
  const apiHubPage = isApiReferenceDocsPage(page.slugs);

  const toc = enrichBenchmarkToc(
    page.slugs,
    enrichApiReferenceToc(page.slugs, page.data.toc ?? []),
    normalized,
  );
  const metricNav = resolveMetricQuickNavFromSlugs(page.slugs);
  const pagination = getDocsPagination(page.slugs, normalized);
  const relatedLinks = getDocsRelatedLinks(page.slugs, normalized);
  const breadcrumbs = getDocsBreadcrumbs(page.slugs, normalized, page.data.title);
  const benchmarkId =
    page.slugs[0] === 'evaluation' && page.slugs[1] === 'benchmark-hub' && page.slugs.length === 3
      ? page.slugs[2]
      : undefined;
  const benchmarkEntry = benchmarkId ? getBenchmarkCatalogEntry(benchmarkId) : undefined;
  const modelId =
    page.slugs[0] === 'guides' && page.slugs[1] === 'supported-models' && page.slugs.length === 3
      ? page.slugs[2]
      : undefined;
  const modelEntry = modelId ? getModelRecipeIndexEntry(modelId) : undefined;
  const lastUpdated = getDocsLastUpdated(page.path, normalized);
  const isWelcomePage = page.slugs.length === 0;
  const isModelRecipePage = Boolean(modelEntry);
  const showToc = !isWelcomePage && !isModelRecipePage && toc.length > 0;
  const showRail = !isModelRecipePage && (showToc || Boolean(metricNav));
  const metricQuickNav = metricNav ? (
    <MetricQuickNav
      locale={normalized}
      page={metricNav.page}
      variant={metricNav.variant}
      placement="rail"
    />
  ) : null;

  return (
    <main
      className={[
        'pi-doc-shell',
        isWelcomePage ? 'pi-doc-shell-welcome' : '',
        benchmarkId ? 'pi-doc-shell-benchmark-detail' : '',
        isModelRecipePage ? 'pi-doc-shell-model-detail' : '',
        apiHubPage ? 'pi-doc-shell-api-hub' : '',
        metricNav ? 'pi-doc-shell-has-metric-nav' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      lang={normalized}
    >
      <DocsScrollBridge />
      <SiteHeader
        variant="solid"
        active={
          page.slugs[0] === 'guides' && page.slugs[1] === 'supported-models'
            ? 'models'
            : page.slugs[0] === 'evaluation' && page.slugs[1] === 'benchmark-hub'
              ? 'benchmarks'
              : 'docs'
        }
        navAriaLabel={t.nav}
        docsHref={docsHref}
        docsLabel={t.docs}
        homeLabel={t.home}
        blogLabel={t.blog}
        eventsLabel={t.events}
        communityLabel={t.community}
        openEnvisionLabel={t.openEnvision}
        searchLabel={t.search}
        languageAriaLabel={t.language}
        beforeInner={<DocsReadingProgress />}
        brandLeading={
          <DocsMobileNavToggle openLabel={t.openMenu} closeLabel={t.closeMenu} />
        }
        languageLinks={i18n.languages.map((item) => ({
          href: getPageUrl(page.slugs, item),
          label: localeNames[item],
          current: item === normalized,
        }))}
      />

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
                          {item.depth === 0 ? (
                            <DocsSidebarIcon url={item.link.url} active={active} />
                          ) : null}
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
          <article
            className={['pi-doc-article', benchmarkId ? 'wf-benchmark-detail' : '']
              .filter(Boolean)
              .join(' ')}
          >
            {isWelcomePage ? (
              <>
                <DocsWelcomeHero
                  description={page.data.description ?? ''}
                  locale={normalized}
                />
                <div className="pi-doc-article-inner pi-doc-welcome-body">
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

                <div
                  className={[
                    'pi-doc-title-row',
                    modelId && getModelPaperThumbSrc(modelId) ? 'has-paper-thumb' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                >
                  <div className="pi-doc-title-heading">
                    {benchmarkId && benchmarkEntry ? (
                      <BenchmarkIdentityMark
                        id={benchmarkId}
                        name={benchmarkEntry.name}
                        category={benchmarkEntry.category}
                        logoKey={benchmarkEntry.logoKey}
                        size="large"
                      />
                    ) : modelEntry ? (
                      <ModelIdentityMark
                        id={modelEntry.id}
                        name={modelEntry.name}
                        provider={modelEntry.provider}
                        category={modelEntry.category}
                        size="large"
                      />
                    ) : null}
                    <h1>{page.data.title}</h1>
                  </div>
                  {modelId ? <ModelPaperThumb modelId={modelId} locale={normalized} /> : null}
                </div>

                {benchmarkId || !page.data.description ? null : (
                  <p className="pi-doc-description">{page.data.description}</p>
                )}

                <DocsPageActions
                  copyMarkdownCopiedLabel={t.askAiCopied}
                  copyMarkdownLabel={t.askAiCopy}
                  editLabel={t.editPage}
                  githubUrl={githubUrl}
                  lastUpdated={lastUpdated}
                  lastUpdatedLabel={t.lastUpdated}
                  markdownLabel={t.markdown}
                  markdownUrl={markdownUrl}
                  pageTitle={page.data.title}
                />

                {showToc ? (
                  <DocsInlineToc items={toc} pageKey={page.url} slugs={page.slugs} title={t.onThisPage} />
                ) : null}

                <div className="pi-doc-content">
                  <MDX
                    components={getMDXComponents({
                      a: createRelativeLink(source, page),
                      ...(modelId
                        ? {
                            Video: ModelPageNoDemoVideo,
                            video: ModelPageNoDemoVideo,
                            figure: ModelPageFigure,
                          }
                        : {}),
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

        {showRail ? (
          <DocsTocRail key={page.url} items={toc} pageKey={page.url} slugs={page.slugs} title={t.onThisPage}>
            {metricQuickNav}
          </DocsTocRail>
        ) : null}
      </div>
    </main>
  );
}
