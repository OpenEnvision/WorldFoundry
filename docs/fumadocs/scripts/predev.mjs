import { spawnSync } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const docsRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const sourceServer = join(docsRoot, '.source', 'server.ts');
const sourceConfig = join(docsRoot, 'source.config.ts');
const mdxCli = join(docsRoot, 'node_modules', 'fumadocs-mdx', 'bin.js');

function run(script) {
  const result = spawnSync('npm', ['run', script], {
    cwd: docsRoot,
    stdio: 'inherit',
  });
  if (result.error || result.status !== 0) {
    console.error(result.error ?? `npm run ${script} failed`);
    process.exit(result.status ?? 1);
  }
}

function runMdxGenerate() {
  if (!existsSync(mdxCli)) {
    console.log('docs/fumadocs: fumadocs-mdx not installed; falling back to npm run mdx:generate');
    run('mdx:generate');
    return;
  }

  const result = spawnSync(process.execPath, [mdxCli], {
    cwd: docsRoot,
    stdio: 'inherit',
  });
  if (result.error || result.status !== 0) {
    console.error(result.error ?? 'MDX generation failed');
    process.exit(result.status ?? 1);
  }
}

function contentIsNewerThanGeneratedSource() {
  if (!existsSync(sourceServer)) {
    return true;
  }

  const sourceMtime = statSync(sourceServer).mtimeMs;
  if (existsSync(sourceConfig) && statSync(sourceConfig).mtimeMs > sourceMtime) {
    return true;
  }

  // Directory timestamps cover additions/deletions; file timestamps cover
  // edited frontmatter and navigation before the dev server starts.
  const result = spawnSync(
    'find',
    [
      join(docsRoot, 'content'),
      '(',
      '-type',
      'd',
      '-o',
      '-name',
      'meta.json',
      '-o',
      '-name',
      'meta.zh.json',
      '-o',
      '-name',
      '*.mdx',
      ')',
      '-newer',
      sourceServer,
      '-print',
      '-quit',
    ],
    { encoding: 'utf8' },
  );

  if (result.error || result.status !== 0) {
    throw result.error ?? new Error('Unable to check MDX source changes');
  }
  return Boolean(result.stdout.trim());
}

function shouldRunMdxGenerate() {
  // Fast startup may reuse existing output, but a clean checkout needs it once.
  if (!existsSync(sourceServer)) {
    return true;
  }
  if (process.env.WF_DOCS_REFRESH_MDX === '1') {
    return true;
  }

  if (process.env.WF_DOCS_SKIP_MDX === '1') {
    return false;
  }

  return contentIsNewerThanGeneratedSource();
}

if (shouldRunMdxGenerate()) {
  runMdxGenerate();
} else {
  console.log(
    'docs/fumadocs: reusing .source/ MDX output (set WF_DOCS_REFRESH_MDX=1 to regenerate, WF_DOCS_SKIP_MDX=1 to always skip)',
  );
}

const apiOut = join(docsRoot, 'generated', 'python-api.json');
if (process.env.WF_DOCS_REFRESH_API === '1' || !existsSync(apiOut)) {
  run('api:generate');
} else {
  console.log(
    'docs/fumadocs: reusing generated/python-api.json (set WF_DOCS_REFRESH_API=1 to regenerate)',
  );
}
