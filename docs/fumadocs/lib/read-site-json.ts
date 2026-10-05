import 'server-only';

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const cache = new Map<string, unknown>();

/** Read generated JSON at runtime so webpack does not parse multi-MB catalogs. */
export function readSiteJson<T>(relativePath: string): T {
  const cached = cache.get(relativePath);
  if (cached) {
    return cached as T;
  }

  const value = JSON.parse(readFileSync(join(process.cwd(), relativePath), 'utf8')) as T;
  cache.set(relativePath, value);
  return value;
}
