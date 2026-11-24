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

/**
 * Segments of the public markdown path `${page.url}.md`, e.g. `zh/docs/quickstart.md`.
 * The /llms.mdx route serves this layout so the build can move its static
 * output onto the public paths unchanged.
 */
export function getPageMarkdownSlugs(page: (typeof source)['$inferPage']) {
  const segments = page.url.split('/').filter(Boolean);
  segments[segments.length - 1] += '.md';
  return segments;
}

export function getPageMarkdownUrl(page: (typeof source)['$inferPage']) {
  const url = `${page.url}.md`;
  return { url: withBasePath(url) ?? url };
}

/** Inverse of `getPageMarkdownSlugs`; `.mdx` leaves are accepted as aliases. */
export function resolveMarkdownSlug(slug: readonly string[]) {
  const [first, ...rest] = slug;
  const locale = isLocale(first) ? first : defaultLocale;
  const segments = isLocale(first) ? rest : [...slug];
  const last = segments.at(-1);
  if (!last || !/\.mdx?$/i.test(last)) return undefined;

  segments[segments.length - 1] = last.replace(/\.mdx?$/i, '');
  const [root, ...pageSlugs] = segments;
  return `/${root}` === docsRoute ? { locale, pageSlugs } : undefined;
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
