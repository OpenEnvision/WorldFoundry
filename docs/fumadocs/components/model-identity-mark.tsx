import { withBasePath } from '@/lib/site-path';

import logoMap from '@/lib/model-logo-map.json';

type ModelIdentityMarkProps = {
  id: string;
  name: string;
  provider: string;
  category: string;
  size?: 'small' | 'medium' | 'large';
};

type LogoAsset = {
  key: string;
  src: string;
  label: string;
};

const logos = logoMap.logos as Record<string, LogoAsset>;
const modelLogos = logoMap.modelLogos as Record<string, string>;

const providerMarks: Array<[RegExp, string]> = [
  [/worldfoundry/i, 'WF'],
  [/tencent|hunyuan/i, 'TH'],
  [/nvidia|cosmos/i, 'NV'],
  [/physical[- ]intelligence/i, 'PI'],
  [/wan[- ]?ai/i, 'WA'],
  [/bytedance/i, 'BD'],
  [/hugging\s*face/i, 'HF'],
  [/lerobot/i, 'LR'],
  [/facebook|meta/i, 'M'],
  [/alibaba|qwen/i, 'QW'],
  [/google|deepmind/i, 'G'],
  [/openai/i, 'OA'],
  [/api/i, 'API'],
];

function initials(value: string) {
  const normalized = value
    .replace(/[^a-zA-Z0-9]+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  if (normalized.length > 1) {
    return normalized
      .slice(0, 2)
      .map((part) => part[0])
      .join('')
      .toUpperCase();
  }

  return (normalized[0] ?? 'M').slice(0, 2).toUpperCase();
}

function markFor(provider: string, name: string) {
  const known = providerMarks.find(([pattern]) => pattern.test(provider));
  return known?.[1] ?? initials(provider || name);
}

function logoFor(id: string) {
  const key = modelLogos[id];
  return key ? logos[key] : undefined;
}

export function ModelIdentityMark({
  id,
  name,
  provider,
  category,
  size = 'medium',
}: ModelIdentityMarkProps) {
  const asset = logoFor(id);

  return (
    <span
      className={`wf-model-mark wf-model-mark-${size}${asset ? ' has-logo' : ''}`}
      data-category={category}
      data-logo={asset?.key}
      title={asset?.label ?? (provider || name)}
      aria-hidden="true"
    >
      {asset ? (
        <img
          className="wf-model-mark-image"
          src={withBasePath(asset.src)}
          alt=""
          width={200}
          height={200}
          draggable={false}
        />
      ) : (
        <span>{markFor(provider, name)}</span>
      )}
    </span>
  );
}
