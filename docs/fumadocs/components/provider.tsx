'use client';
import { HomeMotion } from '@/components/home-motion';
import SearchDialog from '@/components/search';
import { RootProvider } from 'fumadocs-ui/provider/next';
import { type ReactNode } from 'react';

/**
 * next-themes still renders an inline <script> for FOUC. React 19 warns about
 * executable scripts inside client components. Mark it as a data block so the
 * warning is skipped; real theme bootstrap lives in the server layout.
 * @see https://github.com/pacocoursey/next-themes/issues/385
 */
const THEME_SCRIPT_PROPS = { type: 'application/json' } as const;

export function Provider({ children }: { children: ReactNode }) {
  return (
    <RootProvider
      search={{ SearchDialog, preload: false }}
      theme={{ scriptProps: THEME_SCRIPT_PROPS }}
    >
      {children}
      <HomeMotion />
    </RootProvider>
  );
}
