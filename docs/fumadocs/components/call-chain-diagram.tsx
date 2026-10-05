import { ArchDiagram } from '@/components/arch-diagram';
import type { ArchLocale } from '@/lib/arch-diagrams';

export function CallChainDiagram({
  locale = 'en',
}: {
  locale?: ArchLocale;
}) {
  return <ArchDiagram id="workflow" locale={locale} />;
}
