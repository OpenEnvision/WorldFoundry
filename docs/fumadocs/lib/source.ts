import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { docs } from 'collections/server';
import { loader } from 'fumadocs-core/source';
import { defaultLocale, isDefaultLocale, isLocale } from './i18n';
import { docsImageRoute, docsRoute } from './shared';
import { i18n } from './i18n';
import { withBasePath } from './site-path';

// See https://fumadocs.dev/docs/headless/source-api for more info
export const source = loader({
  baseUrl: docsRoute,
  i18n,
  source: docs.toFumadocsSource(),
  plugins: [],
});

function getLocalizedSegments(page: (typeof source)['$inferPage'], leaf?: string) {
  const segments = leaf ? [...page.slugs, leaf] : [...page.slugs];

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

export function getPageMarkdownSlugs(page: (typeof source)['$inferPage']) {
  const segments = getLocalizedSegments(page);
  // A static export cannot write a page as a file and its descendants as a
  // directory at the same path. Give every markdown route a .md leaf.
  if (page.slugs.length === 0) segments.push('index.md');
  else segments[segments.length - 1] += '.md';
  return segments;
}

export function getPageMarkdownUrl(page: (typeof source)['$inferPage']) {
  // Public fumadocs Next.js URL: append `.md` to the page path.
  // next.config rewrites this to /llms.mdx/docs/... in `next dev`.
  const url = `${page.url}.md`;

  return {
    segments: getPageMarkdownSlugs(page),
    url: withBasePath(url) ?? url,
  };
}

export function resolveMarkdownSlug(slug: string[] | undefined) {
  const segments = [...(slug ?? [])];

  if (segments.at(-1) === 'index.md' || segments.at(-1) === 'content.md') {
    segments.pop();
  } else {
    const last = segments.at(-1);
    if (last && /\.mdx?$/i.test(last)) {
      segments[segments.length - 1] = last.replace(/\.mdx?$/i, '');
    }
  }

  const maybeLocale = segments[0];
  const locale = isLocale(maybeLocale) ? maybeLocale : defaultLocale;
  const pageSlugs = isLocale(maybeLocale) ? segments.slice(1) : segments;

  return { locale, pageSlugs };
}

export async function getPageSourceMarkdown(page: (typeof source)['$inferPage']) {
  const filePath = path.join(process.cwd(), 'content/docs', page.path);

  try {
    return await readFile(filePath, 'utf8');
  } catch {
    return getLLMText(page);
  }
}

export async function getLLMText(page: (typeof source)['$inferPage']) {
  const processed = await page.data.getText('processed');

  return `# ${page.data.title} (${page.url})

${processed}`;
}
