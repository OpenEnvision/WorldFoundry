import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createMDX } from 'fumadocs-mdx/next';
import { docsMovedRedirects } from './lib/docs-moved-redirects.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const configPath = fileURLToPath(import.meta.url);
const withMDX = createMDX();
const cacheRoot =
  process.env.WF_DOCS_CACHE_ROOT ?? path.join(__dirname, 'tmp');
// Keep generated type paths portable. dev:ssd redirects tmp/ to a cache that
// belongs to this checkout, without writing machine paths into tsconfig.json.
// With output: export, Next.js also treats a custom distDir as the export
// destination. Keep production on its default so start/CI serve out/.
const distDir = process.env.NODE_ENV === 'production' ? '.next' : 'tmp/next';
const webpackCacheDir =
  process.env.WF_DOCS_WEBPACK_CACHE_DIR ?? path.join(cacheRoot, 'webpack');
const configuredBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const basePath =
  configuredBasePath && configuredBasePath !== '/'
    ? `/${configuredBasePath.replace(/^\/+|\/+$/g, '')}`
    : '';

fs.mkdirSync(cacheRoot, { recursive: true });

const uiOnlyDev = process.env.WF_DOCS_UI_ONLY === '1';
const devWatchIgnored = [
  '**/.next/**',
  '**/.next*/**',
  '**/out/**',
  '**/out*/**',
  '**/node_modules/**',
  '**/generated/**',
  path.join(__dirname, 'tmp', '**'),
  '**/public/org-logos/**',
  path.join(__dirname, 'lib', 'model-recipes-data.json'),
  path.join(__dirname, 'lib', 'model-recipes-index.json'),
  path.join(__dirname, 'lib', 'catalog-coverage-data.json'),
  path.join(__dirname, 'lib', 'benchmark-catalog-status.json'),
  path.join(__dirname, 'lib', 'model-paper-media.json'),
];

if (uiOnlyDev) {
  devWatchIgnored.push(
    path.join(__dirname, 'content', 'docs', 'guides', 'supported-models'),
    path.join(__dirname, 'content', 'docs', 'evaluation', 'benchmark-hub'),
  );
}

/** @type {import('next').NextConfig} */
const config = {
  allowedDevOrigins: ['127.0.0.1', 'localhost'],
  distDir,
  ...(basePath
    ? {
        assetPrefix: basePath,
        basePath,
      }
    : {}),
  ...(process.env.NODE_ENV === 'production'
    ? { output: 'export' }
    : {
        async redirects() {
          return docsMovedRedirects.map(([source, destination]) => ({
            source,
            destination,
            permanent: true,
          }));
        },
        async rewrites() {
          return [
            { source: '/docs.md', destination: '/llms.mdx/docs/index.md' },
            { source: '/docs.mdx', destination: '/llms.mdx/docs/index.md' },
            { source: '/zh/docs.md', destination: '/llms.mdx/docs/zh/index.md' },
            { source: '/zh/docs.mdx', destination: '/llms.mdx/docs/zh/index.md' },
            { source: '/docs/:path*.md', destination: '/llms.mdx/docs/:path*.md' },
            { source: '/docs/:path*.mdx', destination: '/llms.mdx/docs/:path*.md' },
            { source: '/zh/docs/:path*.md', destination: '/llms.mdx/docs/zh/:path*.md' },
            { source: '/zh/docs/:path*.mdx', destination: '/llms.mdx/docs/zh/:path*.md' },
          ];
        },
      }),
  outputFileTracingRoot: __dirname,
  reactStrictMode: true,
  images: {
    unoptimized: true,
  },
  experimental: {
    optimizePackageImports: ['lucide-react', 'fumadocs-ui', 'fumadocs-core'],
  },
  webpack(config, { dev }) {
    if (dev) {
      config.cache = {
        type: 'filesystem',
        cacheDirectory: webpackCacheDir,
        buildDependencies: {
          config: [configPath],
        },
      };
      config.watchOptions = {
        ...config.watchOptions,
        ignored: devWatchIgnored,
      };
    }
    return config;
  },
};

export default withMDX(config);
