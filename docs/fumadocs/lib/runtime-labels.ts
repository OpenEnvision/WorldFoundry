/** Catalog CUDA cell: keep exact versions compact (`12.8`) and make range tags readable (`CUDA 11.8+`). */

function cudaVersionFromDigits(digits: string) {
  if (digits.length < 2) return digits;
  return `${digits.slice(0, -1)}.${digits.slice(-1)}`;
}

export function formatCudaLabel(label: string | null | undefined) {
  if (!label) return '—';

  const trimmed = label.trim();
  const compact = trimmed.replace(/^CUDA\s+/i, '').replace(/[_-]+/g, ' ').trim();

  const plus = compact.match(/^(\d+\.\d+)\+$/);
  if (plus) return `CUDA ${plus[1]}+`;

  const exact = compact.match(/^(\d+\.\d+)$/);
  if (exact) return exact[1];

  const humanized = compact.match(/^Cu(\d{2,3})\s+(Or Newer|Recommended)$/i);
  if (humanized) {
    const version = cudaVersionFromDigits(humanized[1]);
    return /newer/i.test(humanized[2]) ? `CUDA ${version}+` : version;
  }

  const raw = compact.match(/^cu(\d{2,3})(?:\s+(or newer|recommended))?$/i);
  if (raw) {
    const version = cudaVersionFromDigits(raw[1]);
    if (raw[2] && /newer/i.test(raw[2])) return `CUDA ${version}+`;
    return version;
  }

  if (/^prepare(?:\s+only)?$/i.test(compact)) return 'Prepare-only';
  if (/^cpu$/i.test(compact)) return 'CPU';

  return trimmed;
}
