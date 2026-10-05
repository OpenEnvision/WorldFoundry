import rawIndex from '@/lib/model-recipes-index.json';
import type { ModelRecipeIndexData } from '@/lib/model-recipe-types';

export const modelRecipeIndex = rawIndex as unknown as ModelRecipeIndexData;

const recipeById = new Map(modelRecipeIndex.recipes.map((recipe) => [recipe.id, recipe]));

export function getModelRecipeIndexEntry(modelId: string) {
  return recipeById.get(modelId);
}
