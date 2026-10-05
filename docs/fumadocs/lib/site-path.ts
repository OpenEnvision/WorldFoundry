function normalizeBasePath(value: string | undefined) {
  if (!value || value === '/') return '';
  const withLeadingSlash = value.startsWith('/') ? value : `/${value}`;
  return withLeadingSlash.endsWith('/') ? withLeadingSlash.slice(0, -1) : withLeadingSlash;
}

export const basePath = normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH);

// Official docs public/ on GitHub main. One repo/ref; two hosts because GitHub
// serves LFS objects and regular blobs differently:
// - /demos/*.mp4 are LFS → media.githubusercontent.com/media/
//   (raw.githubusercontent.com would return the ~130B pointer text)
// - /cover_4x4_hero.mp4 is a regular blob → raw.githubusercontent.com
//   (the LFS media host 404s for non-LFS files)
const OFFICIAL_MEDIA_REPO = 'OpenEnvision/WorldFoundry';
const OFFICIAL_MEDIA_REF = 'main';
const OFFICIAL_PUBLIC_ROOT = 'docs/fumadocs/public';

const OFFICIAL_LFS_ASSET_BASE_URL =
  `https://media.githubusercontent.com/media/${OFFICIAL_MEDIA_REPO}/${OFFICIAL_MEDIA_REF}/${OFFICIAL_PUBLIC_ROOT}`;
const OFFICIAL_RAW_ASSET_BASE_URL =
  `https://raw.githubusercontent.com/${OFFICIAL_MEDIA_REPO}/${OFFICIAL_MEDIA_REF}/${OFFICIAL_PUBLIC_ROOT}`;

const lfsAssetBaseUrl = (
  process.env.NEXT_PUBLIC_DEMO_ASSET_BASE_URL || OFFICIAL_LFS_ASSET_BASE_URL
).replace(/\/+$/g, '');
const rawAssetBaseUrl = OFFICIAL_RAW_ASSET_BASE_URL;

function isLfsMediaPath(path: string) {
  return path.startsWith('/demos/');
}

function isRawMediaPath(path: string) {
  return path === '/cover_4x4_hero.mp4';
}

export function withBasePath(path: string | undefined) {
  if (!path || !basePath) return path;
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(path)) return path;

  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  if (normalizedPath === basePath || normalizedPath.startsWith(`${basePath}/`)) {
    return normalizedPath;
  }

  return `${basePath}${normalizedPath}`;
}

export function withMediaPath(path: string | undefined) {
  if (!path) return path;
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(path)) return path;

  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  if (isLfsMediaPath(normalizedPath) && lfsAssetBaseUrl) {
    return `${lfsAssetBaseUrl}${normalizedPath}`;
  }
  if (isRawMediaPath(normalizedPath)) {
    return `${rawAssetBaseUrl}${normalizedPath}`;
  }

  return withBasePath(normalizedPath);
}

export function stripBasePath(pathname: string) {
  if (!basePath) return pathname;
  if (pathname === basePath) return '/';
  if (pathname.startsWith(`${basePath}/`)) return pathname.slice(basePath.length);
  return pathname;
}
