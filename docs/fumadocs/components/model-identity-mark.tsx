import { withBasePath } from '@/lib/site-path';

import { resolveModelOrg } from '@/lib/model-org-identity';

export { resolveModelOrgKey } from '@/lib/model-org-identity';

type ModelIdentityMarkProps = {
  id: string;
  name: string;
  provider: string;
  category: string;
  size?: 'small' | 'medium' | 'large';
};

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

export function ModelIdentityMark({
  id,
  name,
  provider,
  category,
  size = 'medium',
}: ModelIdentityMarkProps) {
  const org = resolveModelOrg(id, provider);
  const hasLogo = Boolean(org?.src);

  return (
    <span
      className={`wf-model-mark wf-model-mark-${size}${hasLogo ? ' has-logo' : ''}`}
      data-category={category}
      data-logo={org?.key}
      title={org?.name ?? name}
      aria-hidden="true"
    >
      {org?.src ? (
        <img
          className="wf-model-mark-image"
          src={withBasePath(org.src)}
          alt=""
          width={200}
          height={200}
          draggable={false}
        />
      ) : (
        <span>{org?.abbr ?? initials(name)}</span>
      )}
    </span>
  );
}
