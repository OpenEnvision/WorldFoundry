import { cp, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// The /llms.mdx route exports each page's markdown under the page's own
// `${page.url}.md` path. Static hosting cannot rewrite, so move that tree onto
// the public paths the docs link to (for example `out/zh/docs/quickstart.md`).
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const outDir = path.join(root, 'out');
const markdownDir = path.join(outDir, 'llms.mdx');

await cp(markdownDir, outDir, { recursive: true });
await rm(markdownDir, { recursive: true });
