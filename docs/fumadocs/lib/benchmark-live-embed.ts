import type { BenchmarkLeaderboardBoard } from '@/lib/benchmark-leaderboards';

export type HuggingFaceSpaceRef = {
  org: string;
  name: string;
};

export function huggingfaceSpaceRef(url: string | undefined): HuggingFaceSpaceRef | undefined {
  if (!url) return undefined;
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.replace(/^www\./, '');
    if (host !== 'huggingface.co') return undefined;
    const parts = parsed.pathname.split('/').filter(Boolean);
    if (parts[0] !== 'spaces' || !parts[1] || !parts[2]) return undefined;
    return { org: parts[1], name: parts[2] };
  } catch {
    return undefined;
  }
}

export function huggingfaceEmbedUrl(url: string | undefined): string | undefined {
  const space = huggingfaceSpaceRef(url);
  if (!space) return undefined;
  return `https://huggingface.co/spaces/${space.org}/${space.name}/embed`;
}

function huggingfaceDirectHosts(space: HuggingFaceSpaceRef): string[] {
  const slug = `${space.org}-${space.name}`.toLowerCase().replace(/_/g, '-');
  return [`https://${slug}.hf.space`, `https://${slug}.static.hf.space`];
}

/** Drop `?embed=true` (HF XFO DENY). Keep http for HTTP docs hosts; HTTPS pages fall back if mixed-content blocks. */
export function normalizeLiveEmbedUrl(url: string | undefined): string | undefined {
  if (!url) return undefined;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') return undefined;
    if (huggingfaceSpaceRef(parsed.toString()) && parsed.searchParams.get('embed') === 'true') {
      return huggingfaceEmbedUrl(parsed.toString());
    }
    parsed.searchParams.delete('embed');
    return parsed.toString();
  } catch {
    return undefined;
  }
}

export function liveEmbedCandidates(board: BenchmarkLeaderboardBoard): string[] {
  const candidates: string[] = [];
  const push = (value?: string) => {
    const normalized = normalizeLiveEmbedUrl(value);
    if (normalized && !candidates.includes(normalized)) candidates.push(normalized);
  };

  const space = huggingfaceSpaceRef(board.liveUrl);
  if (space) {
    push(board.liveEmbedUrl);
    push(huggingfaceEmbedUrl(board.liveUrl));
    for (const host of huggingfaceDirectHosts(space)) push(host);
    return candidates;
  }

  push(board.liveEmbedUrl);
  if (board.kind === 'live') push(board.liveUrl);
  return candidates;
}

export function liveEmbedHost(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}
