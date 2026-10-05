import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

import { ModelCommandBuilder as ModelCommandBuilderClient } from '@/components/model-command-builder';
import { ModelIdentityMark } from '@/components/model-identity-mark';
import { ModelVariantCards } from '@/components/model-variant-cards';
import { getModelRecipe, getRelatedModelRecipes } from '@/lib/model-recipes';

type Locale = 'en' | 'zh';

export function ModelCommandBuilder({
  modelId,
  locale = 'en',
}: {
  modelId: string;
  locale?: Locale;
}) {
  const recipe = getModelRecipe(modelId);
  if (!recipe) return null;
  return (
    <>
      <ModelCommandBuilderClient recipe={recipe} locale={locale} />
      <ModelVariantCards modelId={modelId} locale={locale} />
    </>
  );
}

export function ModelRelatedRecipes({
  modelId,
  locale = 'en',
}: {
  modelId: string;
  locale?: Locale;
}) {
  const recipe = getModelRecipe(modelId);
  if (!recipe) return null;
  const related = getRelatedModelRecipes(recipe, 4);
  if (related.length === 0) return null;

  const basePath = `${locale === 'zh' ? '/zh' : ''}/docs/guides/supported-models`;
  const heading = locale === 'zh' ? '相关模型' : 'Related models';
  const details = locale === 'zh' ? '打开配方' : 'Open recipe';

  return (
    <section className="wf-recipe-related" aria-labelledby="wf-related-models">
      <h2 id="wf-related-models">{heading}</h2>
      <div>
        {related.map((item) => (
          <Link className="wf-recipe-related-card" href={`${basePath}/${item.id}`} key={item.id}>
            <div className="wf-recipe-related-identity">
              <ModelIdentityMark
                id={item.id}
                name={item.name}
                provider={item.provider}
                category={item.category}
                size="small"
              />
              <span>
                <span>{item.provider}</span>
                <strong>{item.name}</strong>
              </span>
            </div>
            <small>
              {details} <ArrowRight aria-hidden="true" size={12} />
            </small>
          </Link>
        ))}
      </div>
    </section>
  );
}
