import type { LucideIcon } from 'lucide-react';
import {
  BookOpen,
  Box,
  Braces,
  ChartColumn,
  Compass,
  FileCode2,
  FileText,
  Gauge,
  GitPullRequest,
  HardDrive,
  Layers,
  Library,
  Map,
  MonitorPlay,
  Network,
  Play,
  ShieldCheck,
  Timer,
  Plus,
  PlusCircle,
  Rocket,
  SquareTerminal,
  Terminal,
  Workflow,
} from 'lucide-react';

const SIDEBAR_ICONS: Record<string, LucideIcon> = {
  '': BookOpen,
  'overview/design': Compass,
  quickstart: Rocket,
  'reference/environments': Terminal,
  'guides/local-assets': HardDrive,
  'guides/tui': SquareTerminal,
  'reference/cli': FileCode2,
  'guides/inference': Play,
  'guides/inference-optimization': Timer,
  'guides/run-status': ShieldCheck,
  'guides/supported-models': Layers,
  'guides/studio': MonitorPlay,
  evaluation: Gauge,
  'evaluation/benchmark-hub': Library,
  'evaluation/benchmark-hub/runtime-environments': Workflow,
  'evaluation/metrics': ChartColumn,
  'evaluation/embodied-official-runtime': Box,
  'api-reference': Braces,
  'guides/add-model': PlusCircle,
  'guides/add-benchmark': Plus,
  'maintainers/architecture': Network,
  'maintainers/contributing': GitPullRequest,
  'maintainers/plan': Map,
};

export function sidebarPathKey(url: string): string {
  const path = url.split('?')[0] ?? url;
  return path
    .replace(/^https?:\/\/[^/]+/i, '')
    .replace(/^\/[^/]+\/docs-site/, '')
    .replace(/^\/zh(?=\/|$)/, '')
    .replace(/^\/docs\/?/, '')
    .replace(/\/$/, '');
}

export function DocsSidebarIcon({
  url,
  active = false,
}: {
  url: string;
  active?: boolean;
}) {
  const key = sidebarPathKey(url);
  const Icon = SIDEBAR_ICONS[key] ?? FileText;

  return (
    <Icon
      className={['pi-doc-link-icon', active ? 'is-active' : ''].filter(Boolean).join(' ')}
      aria-hidden="true"
      size={16}
      strokeWidth={1.8}
    />
  );
}
