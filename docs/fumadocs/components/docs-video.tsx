'use client';

import { withBasePath, withMediaPath } from '@/lib/site-path';
import {
  Children,
  isValidElement,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ComponentPropsWithoutRef,
  type ReactNode,
  type SyntheticEvent,
} from 'react';

function hideRecipeFigure(node: HTMLElement | null) {
  const figure = node?.closest('figure');
  if (figure instanceof HTMLElement) {
    figure.hidden = true;
  }
}

function isDemoVideoChild(node: ReactNode): boolean {
  if (!isValidElement<{ src?: unknown; children?: ReactNode }>(node)) return false;
  if (node.type === ModelPageNoDemoVideo || node.type === DocsVideo) return true;
  const typeName = typeof node.type === 'string' ? node.type : undefined;
  if (typeName === 'video' || typeName === 'Video') return true;
  const src = typeof node.props?.src === 'string' ? node.props.src : '';
  if (/\.(mp4|webm|mov)(\?|#|$)/i.test(src) || src.includes('/demos/')) return true;
  return Children.toArray(node.props?.children).some(isDemoVideoChild);
}

/** Model-detail pages never render catalog / Studio demo clips. */
export function ModelPageNoDemoVideo(_props: ComponentPropsWithoutRef<'video'>) {
  const ref = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    hideRecipeFigure(ref.current);
  }, []);
  return <span ref={ref} className="wf-model-page-no-demo" hidden />;
}

export function ModelPageFigure(props: ComponentPropsWithoutRef<'figure'>) {
  if (Children.toArray(props.children).some(isDemoVideoChild)) {
    return null;
  }
  return <figure {...props} />;
}

export function DocsVideo({
  src,
  poster,
  autoPlay,
  muted,
  playsInline,
  preload,
  onError,
  ...props
}: ComponentPropsWithoutRef<'video'>) {
  const resolvedSrc = typeof src === 'string' ? withMediaPath(src) : src;
  const resolvedPoster = typeof poster === 'string' ? withBasePath(poster) : poster;
  const [failed, setFailed] = useState(false);
  const mountedRef = useRef(false);
  const failedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    if (failedRef.current) setFailed(true);
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const handleError = useCallback(
    (event: SyntheticEvent<HTMLVideoElement>) => {
      hideRecipeFigure(event.currentTarget);
      failedRef.current = true;
      if (mountedRef.current) setFailed(true);
      onError?.(event);
    },
    [onError],
  );

  if (failed || !resolvedSrc) {
    return null;
  }

  return (
    <video
      {...props}
      src={resolvedSrc}
      poster={resolvedPoster}
      autoPlay={autoPlay}
      muted={Boolean(autoPlay) || Boolean(muted)}
      playsInline={playsInline ?? true}
      preload={preload ?? (autoPlay ? 'auto' : 'metadata')}
      onError={handleError}
    />
  );
}
