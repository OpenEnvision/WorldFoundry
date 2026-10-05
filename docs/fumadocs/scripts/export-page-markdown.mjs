import { mkdir, readdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const contentDir = path.join(root, 'content', 'docs');
const outDir = path.join(root, 'out');

async function walkMarkdown(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walkMarkdown(fullPath)));
      continue;
    }
    if (/\.mdx?$/i.test(entry.name)) files.push(fullPath);
  }

  return files;
}

function toPublicPath(relativePath) {
  const localeMatch = relativePath.match(/^(.*)\.zh\.(mdx?)$/i);
  const withoutLocale = localeMatch ? `${localeMatch[1]}.${localeMatch[2]}` : relativePath;
  let slug = withoutLocale.replace(/\.mdx?$/i, '');

  if (slug === 'index') slug = '';
  else if (slug.endsWith(`${path.posix.sep}index`) || slug.endsWith('/index')) {
    slug = slug.slice(0, -'/index'.length);
  }

  const docsPath = slug ? `/docs/${slug.replaceAll(path.sep, '/')}.md` : '/docs.md';
  return localeMatch ? `/zh${docsPath}` : docsPath;
}

async function main() {
  try {
    await stat(outDir);
  } catch {
    return;
  }

  const files = await walkMarkdown(contentDir);

  await Promise.all(
    files.map(async (filePath) => {
      const relativePath = path.relative(contentDir, filePath);
      const publicPath = toPublicPath(relativePath);
      const destination = path.join(outDir, publicPath.slice(1));
      await mkdir(path.dirname(destination), { recursive: true });
      await writeFile(destination, await readFile(filePath, 'utf8'));
    }),
  );
}

await main();
