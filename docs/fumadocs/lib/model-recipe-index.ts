import rawIndex from '@/lib/model-recipes-index.json';
import type { ModelRecipeIndexData } from '@/lib/model-recipe-types';

export const modelRecipeIndex = rawIndex as unknown as ModelRecipeIndexData;
