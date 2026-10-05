import { DocsZoomImage } from '@/components/docs-zoom-image';
import {
  getBenchmarkPaperFigures,
  type BenchmarkPaperFigure,
  type BenchmarkPaperFigureRole,
} from '@/lib/benchmark-paper-figures';
import type { Locale } from '@/lib/i18n';
import { withBasePath } from '@/lib/site-path';

const LABEL: Record<Locale, string> = {
  en: 'Paper figures',
  zh: '论文配图',
};

export function BenchmarkPaperFigures({
  benchmarkId,
  locale = 'en',
  roles,
  figures,
}: {
  benchmarkId: string;
  locale?: Locale;
  roles?: BenchmarkPaperFigureRole[];
  figures?: BenchmarkPaperFigure[];
}) {
  const resolved = figures ?? getBenchmarkPaperFigures(benchmarkId, locale, roles);
  if (!resolved.length) return null;

  return (
    <section className="wf-model-paper-figures wf-benchmark-paper-figures" aria-label={LABEL[locale]}>
      {resolved.map((figure) => (
        <figure key={`${figure.role}:${figure.src}`} className="wf-model-paper-figure">
          <div className="wf-recipe-media is-diagram">
            <DocsZoomImage src={withBasePath(figure.src)} alt={figure.alt} />
          </div>
          {figure.caption ? <figcaption>{figure.caption}</figcaption> : null}
        </figure>
      ))}
    </section>
  );
}
