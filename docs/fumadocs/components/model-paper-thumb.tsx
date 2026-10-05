import { DocsZoomImage } from '@/components/docs-zoom-image';
import {
  getModelArchDiagram,
  getModelPaperThumbSrc,
} from '@/lib/model-arch-diagrams';

type Locale = 'en' | 'zh';

const LABEL: Record<Locale, string> = {
  en: 'Paper',
  zh: '论文',
};

function resolveAlt(
  entry: ReturnType<typeof getModelArchDiagram>,
  locale: Locale,
  modelId: string,
): string {
  if (!entry) return modelId;
  const localized =
    locale === 'zh' ? entry.altZh || entry.alt : entry.alt || entry.altZh;
  return localized?.trim() || modelId;
}

export function ModelPaperThumb({
  modelId,
  locale = 'en',
}: {
  modelId: string;
  locale?: Locale;
}) {
  const src = getModelPaperThumbSrc(modelId);
  if (!src) return null;

  const entry = getModelArchDiagram(modelId);
  const alt = resolveAlt(entry, locale, modelId);

  return (
    <div className="wf-model-paper-thumb" aria-label={LABEL[locale]}>
      <DocsZoomImage className="wf-model-paper-thumb__image" src={src} alt={alt} />
    </div>
  );
}
