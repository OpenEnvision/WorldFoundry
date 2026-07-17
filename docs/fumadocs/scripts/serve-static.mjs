import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, resolve, sep } from 'node:path';

const root = resolve(process.cwd(), 'out');
const args = process.argv.slice(2);
const listenIndex = args.indexOf('--listen');
const listenValue = listenIndex >= 0 ? args[listenIndex + 1] : process.env.PORT ?? '3000';

let hostname = '0.0.0.0';
let port = 3000;

if (listenValue?.startsWith('tcp://')) {
  const address = new URL(listenValue);
  hostname = address.hostname;
  port = Number(address.port || 3000);
} else {
  port = Number(listenValue || 3000);
}

const mimeTypes = {
  '.avif': 'image/avif',
  '.css': 'text/css; charset=utf-8',
  '.gif': 'image/gif',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.mp4': 'video/mp4',
  '.ogg': 'audio/ogg',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
  '.webm': 'video/webm',
  '.webp': 'image/webp',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.xml': 'application/xml; charset=utf-8',
};

function candidates(pathname, preferRsc = false) {
  if (pathname === '/') return preferRsc ? ['/index.txt', '/index.html'] : ['/index.html'];
  if (mimeTypes[extname(pathname).toLowerCase()]) return [pathname];

  const cleanPath = pathname.endsWith('/') ? pathname.slice(0, -1) : pathname;
  if (preferRsc) {
    return [`${cleanPath}.txt`, `${cleanPath}.html`, `${cleanPath}/index.txt`, `${cleanPath}/index.html`];
  }
  return [`${cleanPath}.html`, `${cleanPath}/index.html`];
}

function safePath(pathname) {
  const absolute = resolve(root, `.${pathname}`);
  return absolute === root || absolute.startsWith(`${root}${sep}`) ? absolute : null;
}

async function findFile(pathname, preferRsc = false) {
  for (const candidate of candidates(pathname, preferRsc)) {
    const absolute = safePath(candidate);
    if (!absolute) continue;

    try {
      const details = await stat(absolute);
      if (details.isFile()) return { absolute, details, pathname: candidate };
    } catch (error) {
      if (error.code !== 'ENOENT' && error.code !== 'ENOTDIR') throw error;
    }
  }

  return null;
}

function sendFile(request, response, file, statusCode = 200) {
  const { absolute, details, pathname } = file;
  const headers = {
    'Accept-Ranges': 'bytes',
    'Cache-Control': pathname.startsWith('/_next/static/')
      ? 'public, max-age=31536000, immutable'
      : 'public, max-age=0, must-revalidate',
    'Content-Type': mimeTypes[extname(pathname).toLowerCase()] ?? 'application/octet-stream',
    'Last-Modified': details.mtime.toUTCString(),
  };

  const match = request.headers.range?.match(/^bytes=(\d*)-(\d*)$/);
  if (match && statusCode === 200) {
    const suffixLength = match[1] === '' ? Number(match[2]) : null;
    const start = suffixLength === null ? Number(match[1]) : Math.max(0, details.size - suffixLength);
    const requestedEnd = suffixLength === null && match[2] !== '' ? Number(match[2]) : details.size - 1;
    const end = Math.min(requestedEnd, details.size - 1);

    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || start > end) {
      response.writeHead(416, { 'Content-Range': `bytes */${details.size}` });
      response.end();
      return;
    }

    response.writeHead(206, {
      ...headers,
      'Content-Length': end - start + 1,
      'Content-Range': `bytes ${start}-${end}/${details.size}`,
    });

    if (request.method === 'HEAD') response.end();
    else createReadStream(absolute, { start, end }).pipe(response);
    return;
  }

  response.writeHead(statusCode, { ...headers, 'Content-Length': details.size });
  if (request.method === 'HEAD') response.end();
  else createReadStream(absolute).pipe(response);
}

const server = createServer(async (request, response) => {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    response.writeHead(405, { Allow: 'GET, HEAD' });
    response.end('Method Not Allowed');
    return;
  }

  let pathname;
  let preferRsc = false;
  try {
    const requestUrl = new URL(request.url, 'http://localhost');
    pathname = decodeURIComponent(requestUrl.pathname);
    preferRsc = requestUrl.searchParams.has('_rsc') || request.headers.rsc === '1';
  } catch {
    response.writeHead(400);
    response.end('Bad Request');
    return;
  }

  try {
    const file = await findFile(pathname, preferRsc);
    if (file) {
      sendFile(request, response, file);
      return;
    }

    const notFound = await findFile('/404.html');
    if (notFound) {
      sendFile(request, response, notFound, 404);
      return;
    }

    response.writeHead(404);
    response.end('Not Found');
  } catch (error) {
    console.error(error);
    response.writeHead(500);
    response.end('Internal Server Error');
  }
});

server.listen(port, hostname, () => {
  console.log(`WorldFoundry docs: http://${hostname}:${port}`);
});
