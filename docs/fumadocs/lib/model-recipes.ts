import type { ModelRecipe, ModelRecipeData } from '@/lib/model-recipe-types';
import { readSiteJson } from '@/lib/read-site-json';

export const modelRecipeData = readSiteJson<ModelRecipeData>('lib/model-recipes-data.json');

const recipeById = new Map(modelRecipeData.recipes.map((recipe) => [recipe.id, recipe]));

export function getModelRecipe(modelId: string) {
  return recipeById.get(modelId);
}

export function getRelatedModelRecipes(recipe: ModelRecipe, limit = 4) {
  return modelRecipeData.recipes
    .filter((candidate) => candidate.id !== recipe.id && candidate.category === recipe.category)
    .slice(0, limit);
}

export function getModelRecipeStaticParams() {
  return modelRecipeData.recipes.map((recipe) => ({ modelId: recipe.id }));
}
