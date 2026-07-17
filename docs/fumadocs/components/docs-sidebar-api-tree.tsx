'use client';

import { DocsSidebarNavTree } from '@/components/docs-sidebar-nav-tree';
import type { Locale } from '@/lib/i18n';
import type { SidebarNavItem, SidebarNavPage } from '@/lib/docs-sidebar-shared';

type DocsSidebarApiTreeProps = {
  hub: SidebarNavPage;
  items: SidebarNavItem[];
  currentUrl: string;
  locale: Locale;
  defaultOpen: boolean;
  expandLabel: string;
  collapseLabel: string;
};

export function DocsSidebarApiTree(props: DocsSidebarApiTreeProps) {
  return (
    <DocsSidebarNavTree
      {...props}
      panelId="api-sidebar-panel"
      showBadges={false}
    />
  );
}
