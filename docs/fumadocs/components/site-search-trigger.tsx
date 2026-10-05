'use client';

import { I18nProvider } from 'fumadocs-ui/contexts/i18n';
import { FullSearchTrigger } from 'fumadocs-ui/layouts/shared/slots/search-trigger';
import { usePathname } from 'next/navigation';

import { i18n, localeNames } from '@/lib/i18n';
import { stripBasePath } from '@/lib/site-path';

type SiteSearchTriggerProps = {
  label?: string;
};

function searchLabelFromPath(pathname: string) {
  const path = stripBasePath(pathname);
  return path === '/zh' || path.startsWith('/zh/') ? '搜索' : 'Search';
}

export function SiteSearchTrigger({ label }: SiteSearchTriggerProps) {
  const pathname = usePathname();
  const resolved = label ?? searchLabelFromPath(pathname);

  return (
    <I18nProvider
      locale="en"
      locales={i18n.languages.map((item) => ({ locale: item, name: localeNames[item] }))}
      translations={{
        search: resolved,
        'Search(search trigger)': resolved,
        'Open Search(search trigger)(aria-label)': resolved,
      }}
    >
      <FullSearchTrigger className="pi-site-search-trigger" hideIfDisabled />
    </I18nProvider>
  );
}
