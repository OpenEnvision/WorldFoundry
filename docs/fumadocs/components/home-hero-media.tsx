'use client';

import { ChevronLeft, ChevronRight, Play, Volume2, VolumeX } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { homeHeroSlides } from '@/lib/home-hero-slides';

type ConnectionLike = EventTarget & {
  saveData?: boolean;
};

function slideAt(index: number, offset: number) {
  const length = homeHeroSlides.length;
  return homeHeroSlides[(index + offset + length) % length];
}

export function HomeHeroMedia() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [motionAllowed, setMotionAllowed] = useState(false);
  const [videoAllowed, setVideoAllowed] = useState(false);
  const [pageVisible, setPageVisible] = useState(true);
  const [carouselVisible, setCarouselVisible] = useState(true);
  const [videoReady, setVideoReady] = useState(false);
  const [playbackRequested, setPlaybackRequested] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const mutedRef = useRef(true);
  const mountedRef = useRef(false);

  const visibleSlides = useMemo(
    () =>
      ([-1, 0, 1] as const).map((offset) => ({
        offset,
        slide: slideAt(activeIndex, offset),
      })),
    [activeIndex],
  );

  const showVideo = (videoAllowed && motionAllowed) || playbackRequested;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const connection = (navigator as Navigator & { connection?: ConnectionLike }).connection;
    let videoTimer = 0;
    let saveData = Boolean(connection?.saveData);

    const updatePreference = () => {
      window.clearTimeout(videoTimer);
      const allowsMotion = !preference.matches;
      setMotionAllowed(allowsMotion);
      if (allowsMotion && !saveData) {
        videoTimer = window.setTimeout(() => setVideoAllowed(true), 900);
      } else {
        setVideoAllowed(false);
        setVideoReady(false);
      }
    };

    const updateConnection = () => {
      const nextSaveData = Boolean(connection?.saveData);
      if (nextSaveData === saveData) return;
      saveData = nextSaveData;
      updatePreference();
    };

    updatePreference();
    preference.addEventListener('change', updatePreference);
    connection?.addEventListener('change', updateConnection);

    return () => {
      window.clearTimeout(videoTimer);
      preference.removeEventListener('change', updatePreference);
      connection?.removeEventListener('change', updateConnection);
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
    root?.classList.toggle('wf-home-motion-paused', !motionAllowed);
    return () => root?.classList.remove('wf-home-motion-paused');
  }, [motionAllowed]);

  useEffect(() => {
    const root = document.querySelector('.wf-home-hero-carousel-shell');
    if (!root || typeof IntersectionObserver === 'undefined') return;

    const observer = new IntersectionObserver(
      ([entry]) => setCarouselVisible(entry?.isIntersecting ?? true),
      { threshold: 0.12 },
    );
    observer.observe(root);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (video.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
      setVideoReady(true);
    }

    if (!pageVisible || !carouselVisible || !showVideo) {
      video.pause();
      return;
    }

    video.defaultMuted = true;
    video.muted = mutedRef.current;
    void video.play().catch(() => {
      if (mountedRef.current) setPlaying(false);
    });
  }, [carouselVisible, motionAllowed, pageVisible, showVideo, activeIndex]);

  useEffect(() => {
    mutedRef.current = muted;
    const video = videoRef.current;
    if (video) video.muted = muted;
  }, [muted, activeIndex, videoReady]);

  const resetPlayback = useCallback(() => {
    setVideoReady(false);
    setPlaybackRequested(false);
    setPlaying(false);
    setMuted(true);
    mutedRef.current = true;
  }, []);

  const goPrev = useCallback(() => {
    setActiveIndex((current) => (current - 1 + homeHeroSlides.length) % homeHeroSlides.length);
    resetPlayback();
  }, [resetPlayback]);

  const goNext = useCallback(() => {
    setActiveIndex((current) => (current + 1) % homeHeroSlides.length);
    resetPlayback();
  }, [resetPlayback]);

  return (
    <div className="wf-home-hero-carousel-shell" aria-label="World model showcase">
      <div className="wf-home-hero-carousel">
        <div className="wf-home-hero-carousel-viewport">
          <div className="wf-home-hero-carousel-track">
            {visibleSlides.map(({ slide, offset }) => {
              const isActive = offset === 0;
              return (
                <article
                  key={slide.id}
                  className={`wf-home-hero-card${isActive ? ' is-active' : ''}${offset === -1 ? ' is-prev' : ''}${offset === 1 ? ' is-next' : ''}`}
                  aria-hidden={!isActive}
                >
                  <div className="wf-home-hero-card-frame">
                    <div className="wf-home-hero-card-media">
                      <img className="wf-home-hero-card-poster" src={slide.poster} alt="" aria-hidden="true" />
                      {isActive ? (
                        <video
                          ref={videoRef}
                          className={`wf-home-hero-card-video${videoReady ? ' is-ready' : ''}`}
                          src={showVideo ? slide.src : undefined}
                          poster={slide.poster}
                          autoPlay={showVideo}
                          loop
                          muted
                          playsInline
                          preload={showVideo ? 'auto' : 'none'}
                          aria-hidden="true"
                          tabIndex={-1}
                          onCanPlay={() => {
                            if (mountedRef.current) setVideoReady(true);
                          }}
                          onPlaying={() => setPlaying(true)}
                          onPause={() => setPlaying(false)}
                          onError={() => {
                            setPlaying(false);
                            setVideoReady(false);
                          }}
                        />
                      ) : null}
                    </div>
                    {isActive && slide.hasAudio && playing ? (
                      <button
                        type="button"
                        className="wf-home-hero-sound"
                        aria-label={muted ? 'Unmute video' : 'Mute video'}
                        aria-pressed={!muted}
                        onClick={() => {
                          const next = !muted;
                          setMuted(next);
                          mutedRef.current = next;
                          const video = videoRef.current;
                          if (video) video.muted = next;
                        }}
                      >
                        {muted ? (
                          <VolumeX aria-hidden="true" size={16} />
                        ) : (
                          <Volume2 aria-hidden="true" size={16} />
                        )}
                      </button>
                    ) : null}
                    {isActive && !playing ? (
                      <button
                        type="button"
                        className="wf-home-hero-play"
                        aria-label={`Play ${slide.label} video`}
                        onClick={() => {
                          const video = videoRef.current;
                          if (!video) return;
                          video.muted = true;
                          video.defaultMuted = true;
                          if (!video.getAttribute('src')) video.src = slide.src;
                          setPlaybackRequested(true);
                          void video.play().catch(() => setPlaying(false));
                        }}
                      >
                        <Play aria-hidden="true" size={16} />
                        Play video
                      </button>
                    ) : null}
                  </div>
                </article>
              );
            })}
          </div>
        </div>

        <div className="wf-home-hero-carousel-controls">
          <button type="button" className="wf-home-hero-carousel-nav" aria-label="Previous demo" onClick={goPrev}>
            <ChevronLeft aria-hidden="true" size={18} strokeWidth={1.8} />
          </button>
          <div className="wf-home-hero-carousel-dots" role="tablist" aria-label="Demo selection">
            {homeHeroSlides.map((slide, index) => (
              <button
                key={slide.id}
                type="button"
                role="tab"
                aria-selected={index === activeIndex}
                aria-label={slide.label}
                className={index === activeIndex ? 'is-active' : undefined}
                onClick={() => {
                  setActiveIndex(index);
                  resetPlayback();
                }}
              />
            ))}
          </div>
          <button type="button" className="wf-home-hero-carousel-nav" aria-label="Next demo" onClick={goNext}>
            <ChevronRight aria-hidden="true" size={18} strokeWidth={1.8} />
          </button>
        </div>
      </div>
    </div>
  );
}
