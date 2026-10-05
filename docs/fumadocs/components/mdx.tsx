import * as CalloutComponents from 'fumadocs-ui/components/callout';
import * as CardComponents from 'fumadocs-ui/components/card';
import * as AccordionComponents from 'fumadocs-ui/components/accordion';
import * as FilesComponents from 'fumadocs-ui/components/files';
import * as StepsComponents from 'fumadocs-ui/components/steps';
import { Banner } from 'fumadocs-ui/components/banner';
import { DynamicCodeBlock } from 'fumadocs-ui/components/dynamic-codeblock';
import { InlineTOC } from 'fumadocs-ui/components/inline-toc';
import { Tab, Tabs } from '@/components/docs-tabs';
import * as TabsComponents from 'fumadocs-ui/components/tabs';
import defaultMdxComponents from 'fumadocs-ui/mdx';
import { DocsVideo } from '@/components/docs-video';
import { DocsZoomImage } from '@/components/docs-zoom-image';
import { DocsMoved } from '@/components/docs-moved';
import { DocsWelcomeAcknowledgements } from '@/components/docs-welcome-acknowledgements';
import { BenchmarkPageSections } from '@/components/benchmark-page-sections';
import { BenchmarkPaperFigures } from '@/components/benchmark-paper-figures';
import { BenchmarkRecipeCatalog } from '@/components/benchmark-recipe-catalog';
import { DedicatedEnvCatalog } from '@/components/dedicated-env-catalog';
import { ArchDiagram } from '@/components/arch-diagram';
import { CallChainDiagram } from '@/components/call-chain-diagram';
import { MetricQuickNav } from '@/components/metric-quick-nav';
import { ModelArchDiagram } from '@/components/model-arch-diagram';
import { ModelPaperFigures } from '@/components/model-paper-figures';
import { KvCatalog } from '@/components/kv-catalog';
import { ModelCommandBuilder, ModelRelatedRecipes } from '@/components/model-page-widgets';
import { ModelRecipeCatalog } from '@/components/model-recipe-catalog';
import { ModelVariantCards } from '@/components/model-variant-cards';
import { RecipeRecords } from '@/components/recipe-records';
import { ModelRecipeHeader } from '@/components/model-recipe-header';
import {
  PythonApiCatalog,
  PythonApiGroupReference,
  PythonApiReference,
} from '@/components/python-api-reference';
import { StudioRealtimeEnvVars } from '@/components/studio-realtime-env-vars';
import { StudioVisualizerGallery } from '@/components/studio-visualizer-gallery';
import { TeaserImage } from '@/components/teaser-image';
import { MathBlock, MathInline } from '@/components/math';
import { WorldModelProgression } from '@/components/world-model-progression';
import {
  WorldFoundryArchitecture,
  WorldFoundryWorkflow,
} from '@/components/worldfoundry-system-map';
import { withBasePath } from '@/lib/site-path';
import { TypeTable } from 'fumadocs-ui/components/type-table';
import type { MDXComponents } from 'mdx/types';
import type { ComponentPropsWithoutRef } from 'react';

type StaticImageDataLike = {
  src: string;
  height?: number;
  width?: number;
  blurDataURL?: string;
};

type ImgSrc = ComponentPropsWithoutRef<'img'>['src'];

type DocsImageProps = Omit<ComponentPropsWithoutRef<'img'>, 'src'> & {
  src?: ImgSrc | StaticImageDataLike;
};

function isStaticImageDataLike(src: DocsImageProps['src']): src is StaticImageDataLike {
  return typeof src === 'object' && src !== null && 'src' in src && typeof src.src === 'string';
}

function resolveImageSrc(src: DocsImageProps['src']) {
  if (!src) return src;
  if (typeof src === 'string') return withBasePath(src);
  if (isStaticImageDataLike(src)) {
    return withBasePath(src.src);
  }
  return src;
}

function DocsImage({ src, alt, ...props }: DocsImageProps) {
  const resolved = resolveImageSrc(src);
  const dimensions =
    isStaticImageDataLike(src)
      ? {
          width: props.width ?? src.width,
          height: props.height ?? src.height,
        }
      : {};

  if (typeof resolved !== 'string') {
    return <img {...props} {...dimensions} alt={alt} src={resolved} />;
  }

  return <DocsZoomImage {...props} {...dimensions} alt={alt} src={resolved} />;
}

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...CalloutComponents,
    ...CardComponents,
    ...defaultMdxComponents,
    ...AccordionComponents,
    ...FilesComponents,
    ...StepsComponents,
    ...TabsComponents,
    Banner,
    DynamicCodeBlock,
    InlineTOC,
    Tab,
    Tabs,
    ArchDiagram,
    BenchmarkPageSections,
    BenchmarkPaperFigures,
    BenchmarkRecipeCatalog,
    DedicatedEnvCatalog,
    CallChainDiagram,
    DocsMoved,
    DocsImage,
    DocsWelcomeAcknowledgements,
    MathBlock,
    MathInline,
    KvCatalog,
    MetricQuickNav,
    ModelArchDiagram,
    ModelPaperFigures,
    ModelCommandBuilder,
    ModelRecipeCatalog,
    ModelRecipeHeader,
    ModelVariantCards,
    RecipeRecords,
    ModelRelatedRecipes,
    PythonApiCatalog,
    PythonApiGroupReference,
    PythonApiReference,
    img: DocsImage,
    StudioRealtimeEnvVars,
    StudioVisualizerGallery,
    TeaserImage,
    TypeTable,
    Video: DocsVideo,
    WorldFoundryArchitecture,
    WorldFoundryWorkflow,
    WorldModelProgression,
    video: DocsVideo,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
