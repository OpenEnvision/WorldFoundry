import { getPageMarkdownSlugs, getPageSourceMarkdown, resolveMarkdownSlug, source } from '@/lib/source';
import { notFound } from 'next/navigation';

export const revalidate = false;

export async function GET(_req: Request, { params }: RouteContext<'/llms.mdx/[...slug]'>) {
  const { slug } = await params;
  const target = resolveMarkdownSlug(slug);
  const page = target ? source.getPage(target.pageSlugs, target.locale) : undefined;
  if (!page) notFound();

  return new Response(await getPageSourceMarkdown(page), {
    headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
    },
  });
}

export function generateStaticParams() {
  return source.getPages().map((page) => ({
    slug: getPageMarkdownSlugs(page),
  }));
}
