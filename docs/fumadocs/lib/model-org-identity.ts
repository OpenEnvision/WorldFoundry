import logoMap from '@/lib/model-logo-map.json';

export type OrgIdentity = {
  key: string;
  name: string;
  abbr: string;
  src?: string;
  /** Square symbol cropped from a wide logo, used below title size. */
  markSrc?: string;
  /** Wide text-only logo; below title size the mark shows `abbr`. */
  wordmark?: boolean;
};

export type IdentityMarkSize = 'small' | 'medium' | 'large';

const orgs = logoMap.orgs as Record<string, OrgIdentity>;
const modelLogos = logoMap.modelLogos as Record<string, string>;
const benchmarkLogos = (logoMap.benchmarkLogos ?? {}) as Record<string, string>;

const HUNYUAN_ORG_KEY = 'tencent-hunyuan';
const WAN_ORG_KEY = 'wan';

const HUNYUAN_MODEL_IDS = new Set([
  'hunyuan-game-craft',
  'hunyuanvideo',
  'hunyuanvideo-1.5',
  'hunyuanworld-1',
  'hunyuanworld-mirror',
  'hy-embodied',
  'hy-embodied-vla',
  'hy-world-2.0',
  'hy-worldplay',
]);

const WAN_MODEL_IDS = new Set([
  'wan-2p5',
  'wan-2p6',
  'wan-2p7',
  'wan2.1',
  'wan2.1-vace',
  'wan2.2',
  'wan21-fun-14b-cam',
  'wan21-fun-1p3b-cam',
  'wan22-fun-5b-cam',
  'wan22-fun-a14b-cam',
]);

function isHunyuanIdentity(id: string, provider?: string) {
  if (HUNYUAN_MODEL_IDS.has(id)) return true;
  if (/^(hunyuan|hy-world|hy-embodied|hyworld)\b/i.test(id)) return true;
  return /hunyuan|混元/i.test(provider ?? '');
}

function isWanIdentity(id: string, provider?: string) {
  if (WAN_MODEL_IDS.has(id)) return true;
  // Official Wan / Wan2.x / Wan-Fun / VACE ids only — not ati-wan*, minwm-wan*, fastvideo-causal-wan*.
  if (/^(wan2[._]?\d|wan-2p\d|wan21-|wan22-)/i.test(id)) return true;
  return /通义万相|tongyi\s*wanxiang|\bwan-ai\b|\bwan-video\b/i.test(provider ?? '');
}

function lookupModelOrgKey(id: string, provider?: string) {
  if (isHunyuanIdentity(id, provider)) return HUNYUAN_ORG_KEY;
  if (isWanIdentity(id, provider)) return WAN_ORG_KEY;
  return modelLogos[id];
}

function orgFromKey(key: string | undefined): OrgIdentity | undefined {
  return key ? orgs[key] : undefined;
}

export function resolveModelOrg(id: string, provider?: string): OrgIdentity | undefined {
  return orgFromKey(lookupModelOrgKey(id, provider));
}

export function resolveModelOrgKey(id: string, provider?: string) {
  return resolveModelOrg(id, provider)?.key;
}

export function resolveBenchmarkOrg(id: string, logoKey?: string): OrgIdentity | undefined {
  return orgFromKey(logoKey || benchmarkLogos[id]);
}

/** Image for an identity mark; wide logos only appear in full at title size. */
export function orgMarkSrc(org: OrgIdentity | undefined, size: IdentityMarkSize): string | undefined {
  if (!org || size === 'large') return org?.src;
  return org.markSrc ?? (org.wordmark ? undefined : org.src);
}
