'use client';

import { useState } from 'react';

export function ApiCopyButton({
  value,
  label = 'Copy',
  doneLabel = 'Copied',
}: {
  value: string;
  label?: string;
  doneLabel?: string;
}) {
  const [copied, setCopied] = useState(false);

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  return (
    <button
      type="button"
      className="wf-api-copy"
      onClick={onCopy}
      aria-label={copied ? doneLabel : label}
    >
      {copied ? doneLabel : label}
    </button>
  );
}
