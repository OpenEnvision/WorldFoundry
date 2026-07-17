import type { Metadata } from 'next';
import { notFound } from 'next/navigation';

import { ModelRecipePage } from '@/components/model-recipe-page';
import { getModelRecipe, getModelRecipeStaticParams } from '@/lib/model-recipes';

type Params = {
  modelId: string;
};

export default async function Page({ params }: { params: Promise<Params> }) {
  const { modelId } = await params;
  const recipe = getModelRecipe(modelId);
  if (!recipe) notFound();

  return <ModelRecipePage recipe={recipe} locale="en" />;
}

export function generateStaticParams() {
  return getModelRecipeStaticParams();
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { modelId } = await params;
  const recipe = getModelRecipe(modelId);
  if (!recipe) return {};
  return {
    title: `${recipe.name} model recipe | WorldFoundry`,
    description: recipe.summary,
  };
}

export const dynamicParams = false;
