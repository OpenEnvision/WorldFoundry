export const movedDocsDestinations: Record<string, { en: string; zh: string }> = {
  overview: { en: '/docs', zh: '/zh/docs' },
  'overview/world-models': { en: '/docs', zh: '/zh/docs' },
  'overview/capabilities': { en: '/docs/overview/design', zh: '/zh/docs/overview/design' },
  'overview/why-worldfoundry': { en: '/docs', zh: '/zh/docs' },
  cookbook: { en: '/docs/guides/inference', zh: '/zh/docs/guides/inference' },
  'cookbook/inference-recipes': { en: '/docs/guides/inference', zh: '/zh/docs/guides/inference' },
};

export function movedDocsDestination(slugs: readonly string[] | undefined, locale: 'en' | 'zh') {
  const destination = movedDocsDestinations[(slugs ?? []).join('/')];
  return destination ? destination[locale] : null;
}
