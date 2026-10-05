'use client';

import {
  useCallback,
  useEffect,
  useState,
  type ComponentPropsWithoutRef,
  type MouseEvent,
  type SyntheticEvent,
} from 'react';
import { createPortal } from 'react-dom';

import { withBasePath } from '@/lib/site-path';

type StaticImageDataLike = {
  src: string;
  height?: number;
  width?: number;
};

type DocsZoomImageProps = Omit<ComponentPropsWithoutRef<'img'>, 'src'> & {
  src?: string | StaticImageDataLike;
};

function resolveSrc(src: DocsZoomImageProps['src']) {
  if (!src) return undefined;
  if (typeof src === 'string') return withBasePath(src) ?? src;
  return withBasePath(src.src) ?? src.src;
}

function mergeClassName(className?: string) {
  return className ? `wf-docs-zoom-image ${className}` : 'wf-docs-zoom-image';
}

function hideBrokenFrame(node: HTMLElement | null) {
  const frame = node?.closest('figure, .wf-model-paper-thumb');
  if (frame instanceof HTMLElement) {
    frame.hidden = true;
  }
}

export function DocsZoomImage({
  src,
  alt,
  className,
  width,
  height,
  onClick,
  onError,
  ...props
}: DocsZoomImageProps) {
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  const [portalRoot, setPortalRoot] = useState<HTMLElement | null>(null);

  useEffect(() => {
    setPortalRoot(document.body);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  const resolved = resolveSrc(src);
  const handleError = useCallback(
    (event: SyntheticEvent<HTMLImageElement>) => {
      hideBrokenFrame(event.currentTarget);
      setFailed(true);
      onError?.(event);
    },
    [onError],
  );

  if (!resolved || failed) return null;

  const resolvedWidth =
    width ?? (src && typeof src === 'object' ? src.width : undefined);
  const resolvedHeight =
    height ?? (src && typeof src === 'object' ? src.height : undefined);
  const mergedClassName = mergeClassName(className);

  function openZoom(event: MouseEvent<HTMLImageElement>) {
    onClick?.(event);
    if (event.defaultPrevented) return;
    setOpen(true);
  }

  return (
    <span data-rmiz="">
      <span data-rmiz-content="found">
        <img
          {...props}
          alt={alt ?? ''}
          className={mergedClassName}
          height={resolvedHeight}
          src={resolved}
          width={resolvedWidth}
          onClick={openZoom}
          onError={handleError}
        />
      </span>
      {open && portalRoot
        ? createPortal(
            <div
              className="wf-docs-zoom-overlay"
              role="dialog"
              aria-modal="true"
              aria-label={alt || 'Zoomed image'}
              onClick={() => setOpen(false)}
            >
              <img src={resolved} alt={alt ?? ''} />
            </div>,
            portalRoot,
          )
        : null}
    </span>
  );
}
