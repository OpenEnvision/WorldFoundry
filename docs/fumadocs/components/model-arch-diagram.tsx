import { DocsZoomImage } from '@/components/docs-zoom-image';
import {
  getModelArchDiagram,
  hasModelArchDiagramAsset,
  isPaperCoverSrc,
} from '@/lib/model-arch-diagrams';
import { withBasePath } from '@/lib/site-path';

type Locale = 'en' | 'zh';

const LABEL: Record<Locale, string> = {
  en: 'Architecture',
  zh: '架构',
};

const PLACEHOLDER: Record<Locale, { title: string; body: string }> = {
  en: {
    title: 'Diagram pending',
    body: 'Architecture figure not yet available for this model.',
  },
  zh: {
    title: '架构图待补充',
    body: '该模型暂无论文架构图。',
  },
};

function resolveText(
  entry: NonNullable<ReturnType<typeof getModelArchDiagram>>,
  locale: Locale,
  field: 'alt' | 'caption',
): string {
  const zhKey = `${field}Zh` as const;
  const localized =
    locale === 'zh'
      ? entry[zhKey] || entry[field]
      : entry[field] || entry[zhKey];
  return localized?.trim() || entry.modelId;
}

export function ModelArchDiagram({
  modelId,
  locale = 'en',
  showPlaceholder = false,
}: {
  modelId: string;
  locale?: Locale;
  showPlaceholder?: boolean;
}) {
  const entry = getModelArchDiagram(modelId);
  const hasAsset =
    hasModelArchDiagramAsset(modelId) &&
    Boolean(entry?.src) &&
    !isPaperCoverSrc(entry?.src);

  if (!hasAsset) {
    if (!showPlaceholder) return null;
    const copy = PLACEHOLDER[locale];
    return (
      <section
        className="wf-model-arch-diagram wf-model-arch-diagram--placeholder"
        aria-label={LABEL[locale]}
      >
        <figure className="wf-recipe-media is-diagram">
          <div className="wf-model-arch-diagram__frame">
            <span className="wf-model-arch-diagram__label">{copy.title}</span>
            <p>{copy.body}</p>
          </div>
          <figcaption>{LABEL[locale]}</figcaption>
        </figure>
      </section>
    );
  }

  const alt = resolveText(entry!, locale, 'alt');
  const caption = resolveText(entry!, locale, 'caption');

  return (
    <section className="wf-model-arch-diagram" aria-label={LABEL[locale]}>
      <figure className="wf-recipe-media is-diagram">
        <DocsZoomImage src={withBasePath(entry!.src!)} alt={alt} />
        <figcaption>{caption}</figcaption>
      </figure>
    </section>
  );
}
