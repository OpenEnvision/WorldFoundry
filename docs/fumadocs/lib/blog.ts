import { blog } from '@/lib/source';

function toDateValue(value: unknown): Date | null {
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value;
  if (typeof value === 'string' && value.trim()) {
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) return parsed;
  }
  return null;
}

export function formatBlogDate(value: unknown): string | null {
  const date = toDateValue(value);
  if (!date) return null;
  return new Intl.DateTimeFormat('en', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  }).format(date);
}

export function getBlogPosts() {
  return [...blog.getPages()].sort((a, b) => {
    const aTime = toDateValue(a.data.date)?.getTime() ?? 0;
    const bTime = toDateValue(b.data.date)?.getTime() ?? 0;
    return bTime - aTime;
  });
}

export function getBlogPost(slug: string) {
  return blog.getPage([slug]);
}
