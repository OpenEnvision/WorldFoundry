import { blog as blogPosts, docs } from 'collections/server';
import { loader } from 'fumadocs-core/source';
import { toFumadocsSource } from 'fumadocs-mdx/runtime/server';
import { defaultLocale, isDefaultLocale } from './i18n';
import { docsContentRoute, docsImageRoute, docsRoute } from './shared';
import { i18n } from './i18n';
import { withBasePath } from './site-path';

// See https://fumadocs.dev/docs/headless/source-api for more info
export const source = loader({
  baseUrl: docsRoute,
  i18n,
  source: docs.toFumadocsSource(),
  plugins: [],
});

export const blog = loader({
  baseUrl: '/blog',
  source: toFumadocsSource(blogPosts, []),
});

function getLocalizedSegments(page: (typeof source)['$inferPage'], leaf: string) {
  const segments = [...page.slugs, leaf];

  if (!isDefaultLocale(page.locale)) {
    segments.unshift(page.locale ?? defaultLocale);
  }

  return segments;
}

export function getPageImage(page: (typeof source)['$inferPage']) {
  const segments = getLocalizedSegments(page, 'image.png');

  return {
    segments,
    url: withBasePath(`${docsImageRoute}/${segments.join('/')}`) ?? `${docsImageRoute}/${segments.join('/')}`,
  };
}

export function getPageMarkdownUrl(page: (typeof source)['$inferPage']) {
  const segments = getLocalizedSegments(page, 'content.md');

  return {
    segments,
    url: withBasePath(`${docsContentRoute}/${segments.join('/')}`) ?? `${docsContentRoute}/${segments.join('/')}`,
  };
}

export async function getLLMText(page: (typeof source)['$inferPage']) {
  const processed = await page.data.getText('processed');

  return `# ${page.data.title} (${page.url})

${processed}`;
}
