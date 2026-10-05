'use client';

import { useState } from 'react';

export function DocsCopyMarkdownButton({
  label,
  copiedLabel,
  markdownUrl,
}: {
  label: string;
  copiedLabel: string;
  markdownUrl: string;
}) {
  const [copied, setCopied] = useState(false);

  async function onCopy() {
    try {
      const response = await fetch(markdownUrl);
      const text = await response.text();
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      window.open(markdownUrl, '_blank', 'noopener,noreferrer');
    }
  }

  return (
    <button type="button" className="pi-doc-action-link" onClick={onCopy}>
      {copied ? copiedLabel : label}
    </button>
  );
}
