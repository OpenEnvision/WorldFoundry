import { readSiteJson } from '@/lib/read-site-json';

export type ModelArchDiagramEntry = {
  modelId: string;
  /** True architecture / method figure for the article body. Never a paper first-page. */
  src?: string;
  /** Official paper teaser (Fig. 1 / author collage). Never a paper first-page. */
  teaserSrc?: string;
  /** Paper title-page / header crop, shown beside the model title. */
  paperSrc?: string;
  alt?: string;
  altZh?: string;
  caption?: string;
  captionZh?: string;
  teaserAlt?: string;
  teaserAltZh?: string;
  teaserCaption?: string;
  teaserCaptionZh?: string;
};

export type ModelPaperFigure = {
  role: 'teaser' | 'overview';
  src: string;
  alt: string;
  caption: string;
};

type ModelArchDiagramData = {
  version: number;
  diagrams: Record<string, ModelArchDiagramEntry>;
};

const data = readSiteJson<ModelArchDiagramData>('lib/model-arch-diagrams.json');

export function isPaperCoverSrc(src?: string): boolean {
  return Boolean(src?.toLowerCase().endsWith('/paper.png'));
}

export function getModelArchDiagram(modelId: string): ModelArchDiagramEntry | undefined {
  return data.diagrams[modelId];
}

export function getModelPaperThumbSrc(modelId: string): string | undefined {
  const entry = data.diagrams[modelId];
  if (!entry) return undefined;
  if (entry.paperSrc) return entry.paperSrc;
  if (isPaperCoverSrc(entry.src)) return entry.src;
  return undefined;
}

export function hasModelArchDiagramAsset(modelId: string): boolean {
  const src = data.diagrams[modelId]?.src;
  return Boolean(src) && !isPaperCoverSrc(src);
}

function localizedField(
  entry: ModelArchDiagramEntry,
  locale: 'en' | 'zh',
  field: 'alt' | 'caption' | 'teaserAlt' | 'teaserCaption',
): string {
  const zhKey = `${field}Zh` as const;
  const localized =
    locale === 'zh'
      ? entry[zhKey] || entry[field]
      : entry[field] || entry[zhKey];
  return localized?.trim() || '';
}

export function getModelPaperFigures(
  modelId: string,
  locale: 'en' | 'zh' = 'en',
): ModelPaperFigure[] {
  const entry = data.diagrams[modelId];
  if (!entry) return [];
  const figures: ModelPaperFigure[] = [];
  if (entry.teaserSrc && !isPaperCoverSrc(entry.teaserSrc)) {
    const caption =
      localizedField(entry, locale, 'teaserCaption') ||
      (locale === 'zh' ? '论文官方 Teaser。' : 'Official paper teaser.');
    figures.push({
      role: 'teaser',
      src: entry.teaserSrc,
      alt: localizedField(entry, locale, 'teaserAlt') || caption,
      caption,
    });
  }
  if (entry.src && !isPaperCoverSrc(entry.src)) {
    const caption =
      localizedField(entry, locale, 'caption') ||
      (locale === 'zh' ? '论文方法总览。' : 'Official paper method overview.');
    figures.push({
      role: 'overview',
      src: entry.src,
      alt: localizedField(entry, locale, 'alt') || caption,
      caption,
    });
  }
  return figures;
}

export function hasModelPaperFigures(modelId: string): boolean {
  return getModelPaperFigures(modelId).length > 0;
}
