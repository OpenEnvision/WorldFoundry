'use client';

import { DOCS_MAIN_SCROLL_EVENT, isMainScrollable } from '@/lib/docs-toc-scroll-spy';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';

export function DocsReadingProgress() {
  const pathname = usePathname();
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const main = document.querySelector<HTMLElement>('.pi-doc-main');
    if (!main) return;

    const update = () => {
      if (isMainScrollable(main)) {
        const scrollable = main.scrollHeight - main.clientHeight;
        setProgress(scrollable <= 0 ? 0 : Math.min(1, Math.max(0, main.scrollTop / scrollable)));
        return;
      }

      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(scrollable <= 0 ? 0 : Math.min(1, Math.max(0, window.scrollY / scrollable)));
    };

    update();
    main.addEventListener('scroll', update, { passive: true });
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener(DOCS_MAIN_SCROLL_EVENT, update);
    window.addEventListener('resize', update);

    return () => {
      main.removeEventListener('scroll', update);
      window.removeEventListener('scroll', update);
      window.removeEventListener(DOCS_MAIN_SCROLL_EVENT, update);
      window.removeEventListener('resize', update);
    };
  }, [pathname]);

  return (
    <div className="pi-doc-reading-progress" aria-hidden="true">
      <span style={{ width: `${Math.round(progress * 100)}%` }} />
    </div>
  );
}
