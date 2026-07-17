import type { Metadata } from 'next';
import { notFound } from 'next/navigation';

import { ModelRecipePage } from '@/components/model-recipe-page';
import { defaultLocale, i18n, isLocale } from '@/lib/i18n';
import { getModelRecipe, getModelRecipeStaticParams } from '@/lib/model-recipes';

type Params = {
  lang: string;
  modelId: string;
};

export default async function Page({ params }: { params: Promise<Params> }) {
  const { lang, modelId } = await params;
  if (!isLocale(lang) || lang === defaultLocale) notFound();
  const recipe = getModelRecipe(modelId);
  if (!recipe) notFound();

  return <ModelRecipePage recipe={recipe} locale={lang} />;
}

export function generateStaticParams() {
  const locales = i18n.languages.filter((lang) => lang !== defaultLocale);
  const models = getModelRecipeStaticParams();
  return locales.flatMap((lang) => models.map(({ modelId }) => ({ lang, modelId })));
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { lang, modelId } = await params;
  if (!isLocale(lang) || lang === defaultLocale) return {};
  const recipe = getModelRecipe(modelId);
  if (!recipe) return {};
  return {
    title: `${recipe.name} 模型配方 | WorldFoundry`,
    description: recipe.summary,
  };
}

export const dynamicParams = false;
