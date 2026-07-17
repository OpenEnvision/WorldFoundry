'use client';

import { Pause, Play } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import { withBasePath } from '@/lib/site-path';

type ConnectionLike = EventTarget & {
  effectiveType?: string;
  saveData?: boolean;
};

const motionPreferenceKey = 'worldfoundry-home-motion-paused';

export function HomeHeroMedia() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [motionAllowed, setMotionAllowed] = useState(false);
  const [videoAllowed, setVideoAllowed] = useState(false);
  const [videoVisible, setVideoVisible] = useState(true);
  const [pageVisible, setPageVisible] = useState(true);
  const [videoReady, setVideoReady] = useState(false);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const compactViewport = window.matchMedia('(max-width: 640px)');
    const connection = (navigator as Navigator & { connection?: ConnectionLike }).connection;
    let videoTimer = 0;

    try {
      setPaused(window.localStorage.getItem(motionPreferenceKey) === 'true');
    } catch {
      // Storage can be unavailable in hardened browser contexts.
    }

    const mountVideoAfterLoad = () => {
      videoTimer = window.setTimeout(() => setVideoAllowed(true), 1200);
    };

    const updatePreference = () => {
      window.clearTimeout(videoTimer);
      window.removeEventListener('load', mountVideoAfterLoad);
      const allowsMotion = !preference.matches;
      const conservesData = Boolean(
        connection?.saveData ||
          ['slow-2g', '2g', '3g'].includes(connection?.effectiveType ?? '') ||
          compactViewport.matches,
      );

      setMotionAllowed(allowsMotion);
      setVideoAllowed(false);
      setVideoReady(false);

      if (allowsMotion && !conservesData) {
        if (document.readyState === 'complete') mountVideoAfterLoad();
        else window.addEventListener('load', mountVideoAfterLoad, { once: true });
      }
    };

    updatePreference();
    preference.addEventListener('change', updatePreference);
    compactViewport.addEventListener('change', updatePreference);
    connection?.addEventListener('change', updatePreference);

    return () => {
      window.clearTimeout(videoTimer);
      window.removeEventListener('load', mountVideoAfterLoad);
      preference.removeEventListener('change', updatePreference);
      compactViewport.removeEventListener('change', updatePreference);
      connection?.removeEventListener('change', updatePreference);
    };
  }, []);

  useEffect(() => {
    const updateVisibility = () => setPageVisible(document.visibilityState !== 'hidden');
    updateVisibility();
    document.addEventListener('visibilitychange', updateVisibility);
    return () => document.removeEventListener('visibilitychange', updateVisibility);
  }, []);

  useEffect(() => {
    const root = document.querySelector('.wf-home-shell');
    root?.classList.toggle('wf-home-motion-paused', paused);
    return () => root?.classList.remove('wf-home-motion-paused');
  }, [paused]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || typeof IntersectionObserver === 'undefined') return;

    const observer = new IntersectionObserver(
      ([entry]) => setVideoVisible(entry?.isIntersecting ?? true),
      { threshold: 0.08 },
    );
    observer.observe(video);
    return () => observer.disconnect();
  }, [videoAllowed]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (paused || !motionAllowed || !pageVisible || !videoVisible) {
      video.pause();
      return;
    }

    void video.play().catch(() => {
      // The poster remains visible if a browser blocks background autoplay.
    });
  }, [motionAllowed, pageVisible, paused, videoAllowed, videoVisible]);

  function toggleMotion() {
    setPaused((current) => {
      const next = !current;
      try {
        window.localStorage.setItem(motionPreferenceKey, String(next));
      } catch {
        // The control still works for the current page when storage is unavailable.
      }
      return next;
    });
  }

  const poster = withBasePath('/cover_4x4_hero-poster.webp');

  return (
    <>
      <img
        className="wf-home-hero-poster"
        src={poster}
        alt=""
        aria-hidden="true"
      />
      {videoAllowed ? (
        <video
          ref={videoRef}
          className={`wf-home-hero-video${videoReady ? ' is-ready' : ''}`}
          src={withBasePath('/cover_4x4_hero.mp4')}
          poster={poster}
          autoPlay={!paused}
          loop
          muted
          playsInline
          preload={paused ? 'none' : 'metadata'}
          aria-hidden="true"
          tabIndex={-1}
          onCanPlay={() => setVideoReady(true)}
        />
      ) : null}
      {motionAllowed ? (
        <button
          className="wf-home-motion-toggle"
          type="button"
          aria-label={paused ? 'Play homepage motion' : 'Pause homepage motion'}
          aria-pressed={paused}
          title={paused ? 'Play homepage motion' : 'Pause homepage motion'}
          onClick={toggleMotion}
        >
          {paused ? (
            <Play aria-hidden="true" size={14} strokeWidth={1.8} />
          ) : (
            <Pause aria-hidden="true" size={14} strokeWidth={1.8} />
          )}
        </button>
      ) : null}
    </>
  );
}
