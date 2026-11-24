import { getPageImage, source } from '@/lib/source';
import { notFound } from 'next/navigation';
import type { Metadata } from 'next';
import { defaultLocale, i18n, isLocale, type Locale } from '@/lib/i18n';

function normalizeLocale(locale: string): Locale {
  if (!isLocale(locale)) notFound();
  return locale;
}

export function generateDocsStaticParams(locale = defaultLocale) {
  const normalized = normalizeLocale(locale);

  return source.getPages(normalized).map((page) => ({
    slug: page.slugs,
  }));
}

export function generateLocalizedDocsStaticParams() {
  return i18n.languages.flatMap((locale) =>
    source.getPages(locale).map((page) => ({
      lang: locale,
      slug: page.slugs,
    })),
  );
}

export async function generateDocsMetadata(slug: string[] | undefined, locale: string): Promise<Metadata> {
  const normalized = normalizeLocale(locale);
  const page = source.getPage(slug, normalized);
  if (!page) notFound();

  return {
    title: page.data.title,
    description: page.data.description,
    openGraph: {
      images: getPageImage(page).url,
    },
  };
}
