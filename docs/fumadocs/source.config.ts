import { defineCollections, defineConfig, defineDocs, frontmatterSchema } from 'fumadocs-mdx/config';
import { metaSchema, pageSchema } from 'fumadocs-core/source/schema';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import { z } from 'zod';
import { remarkBenchmarkSections } from './lib/remark-benchmark-sections';

// You can customize Zod schemas for frontmatter and `meta.json` here
// see https://fumadocs.dev/docs/mdx/collections
export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: pageSchema.extend({
      pageSource: z.enum(['generated', 'authored']).optional(),
      modelId: z.string().optional(),
    }),
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
});

export const blog = defineCollections({
  type: 'doc',
  dir: 'content/blog',
  schema: frontmatterSchema.extend({
    author: z.string().optional(),
    date: z.union([z.string(), z.date()]).optional(),
  }),
});

export default defineConfig({
  mdxOptions: {
    remarkPlugins: [remarkMath, remarkBenchmarkSections],
    rehypePlugins: (plugins) => [rehypeKatex, ...plugins],
  },
});
