'use client';

import { THEME_BOOTSTRAP_SCRIPT } from '@/lib/theme-bootstrap';
import { useServerInsertedHTML } from 'next/navigation';
import { useRef } from 'react';

/**
 * Injects the theme bootstrap into `<head>` during SSR only.
 * A raw `<script>` (or `next/script` beforeInteractive) stays in the React
 * tree and trips React 19's "scripts inside components never execute" warning.
 */
export function ThemeBootstrap() {
  const flushed = useRef(false);

  useServerInsertedHTML(() => {
    if (flushed.current) {
      return null;
    }
    flushed.current = true;
    return (
      <script
        id="wf-theme-bootstrap"
        dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP_SCRIPT }}
      />
    );
  });

  return null;
}
