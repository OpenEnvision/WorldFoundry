import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createMDX } from 'fumadocs-mdx/next';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const configPath = fileURLToPath(import.meta.url);
const withMDX = createMDX();
const useFastDevCache = process.env.WF_DOCS_FAST_CACHE === '1';
const distDir = useFastDevCache
  ? path.join(__dirname, 'tmp/worldfoundry-docs-next')
  : '.next';
const webpackCacheDir =
  process.env.WF_DOCS_WEBPACK_CACHE_DIR ??
  path.join(__dirname, 'tmp/worldfoundry-webpack-cache');
const configuredBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const basePath =
  configuredBasePath && configuredBasePath !== '/'
    ? `/${configuredBasePath.replace(/^\/+|\/+$/g, '')}`
    : '';

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
  ...(process.env.NODE_ENV === 'production' ? { output: 'export' } : {}),
  outputFileTracingRoot: path.resolve(__dirname, '..', '..'),
  reactStrictMode: true,
  images: {
    unoptimized: true,
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
        ignored: [
          '**/.next/**',
          '**/.next*/**',
          '**/out/**',
          '**/out*/**',
          '**/node_modules/**',
          '**/generated/**',
          '**/public/model-logos/**',
          path.join(__dirname, 'lib', 'model-recipes-data.json'),
          path.join(__dirname, 'lib', 'model-recipes-index.json'),
        ],
      };
    }
    return config;
  },
};

export default withMDX(config);
