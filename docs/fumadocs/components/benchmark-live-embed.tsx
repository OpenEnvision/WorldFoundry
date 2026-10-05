'use client';

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';

type BenchmarkLiveEmbedProps = {
  srcs: string[];
  title: string;
  caption: string;
  openHref: string;
  openLabel: string;
  failedNote: string;
  fallback: ReactNode;
};

const LOAD_TIMEOUT_MS = 12000;

function frameLooksBlocked(frame: HTMLIFrameElement): boolean {
  try {
    const href = frame.contentWindow?.location.href ?? '';
    // Initial about:blank load is normal; XFO-blocked frames stay there until timeout.
    if (href === 'about:blank') return false;
    const doc = frame.contentDocument;
    if (!doc?.body) return true;
    const text = (doc.body.textContent || '').trim();
    return doc.body.childElementCount === 0 && text.length === 0;
  } catch {
    return false;
  }
}

export function BenchmarkLiveEmbed({
  srcs,
  title,
  caption,
  openHref,
  openLabel,
  failedNote,
  fallback,
}: BenchmarkLiveEmbedProps) {
  const frameRef = useRef<HTMLIFrameElement | null>(null);
  const [index, setIndex] = useState(0);
  const [failed, setFailed] = useState(srcs.length === 0);
  const [settled, setSettled] = useState(false);
  const src = srcs[index];

  const advance = useCallback(() => {
    setSettled(false);
    setIndex((current) => {
      if (current + 1 < srcs.length) return current + 1;
      setFailed(true);
      return current;
    });
  }, [srcs.length]);

  useEffect(() => {
    setSettled(false);
  }, [src]);

  useEffect(() => {
    if (failed || !src || settled) return;
    const timer = window.setTimeout(() => {
      advance();
    }, LOAD_TIMEOUT_MS);
    return () => window.clearTimeout(timer);
  }, [advance, failed, settled, src]);

  const handleLoad = useCallback(() => {
    const frame = frameRef.current;
    if (frame && frameLooksBlocked(frame)) {
      advance();
      return;
    }
    setSettled(true);
  }, [advance]);

  if (failed || !src) {
    return (
      <div className="wf-benchmark-live-fallback">
        <p className="wf-benchmark-board-note">{failedNote}</p>
        {fallback}
      </div>
    );
  }

  return (
    <figure className="wf-benchmark-live-embed-shell">
      <div className="wf-benchmark-live-embed">
        <iframe
          ref={frameRef}
          src={src}
          title={title}
          referrerPolicy="no-referrer-when-downgrade"
          sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox"
          onLoad={handleLoad}
          onError={advance}
        />
      </div>
      <figcaption>
        <span>{caption}</span>
        <a href={openHref} rel="noreferrer">
          {openLabel}
        </a>
      </figcaption>
    </figure>
  );
}
