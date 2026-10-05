import { getModelRecipe } from '@/lib/model-recipes';
import { readerStatusLabel } from '@/lib/model-status-label';

type Locale = 'en' | 'zh';

const copy = {
  en: {
    heading: 'Variants',
    task: 'Task',
    profile: 'Runtime profile',
    binding: 'Pipeline binding',
    status: 'Status',
    empty: 'Not recorded',
  },
  zh: {
    heading: '变体',
    task: '任务',
    profile: 'Runtime profile',
    binding: 'Pipeline binding',
    status: '状态',
    empty: '未记录',
  },
} as const;

export function ModelVariantCards({
  modelId,
  locale = 'en',
}: {
  modelId: string;
  locale?: Locale;
}) {
  const recipe = getModelRecipe(modelId);
  const variants = recipe?.variants ?? [];
  if (variants.length < 2) return null;

  const t = copy[locale];

  return (
    <div className="pi-kv-catalog is-vars not-prose">
      <section className="pi-kv-group">
        <h3>{t.heading}</h3>
        <ul>
          {variants.map((variant) => (
            <li className="pi-kv-row" key={variant.id}>
              <div className="pi-kv-head">
                <div className="pi-kv-names">
                  <code>{variant.id}</code>
                </div>
                <span className="pi-kv-default">
                  <span>{t.status}</span>
                  {readerStatusLabel(variant.status, locale, t.empty)}
                </span>
              </div>
              <p>{variant.task || t.empty}</p>
              <p className="pi-kv-read">
                <span>{t.profile}</span>
                <code>{variant.runtimeProfile || t.empty}</code>
              </p>
              <p className="pi-kv-read">
                <span>{t.binding}</span>
                <code>{variant.pipelineBinding || t.empty}</code>
              </p>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
