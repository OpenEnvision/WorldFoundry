'use client';

import { ChevronLeft, ChevronRight } from 'lucide-react';
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { homeHeroSlides, type HomeHeroSlide } from '@/lib/home-hero-slides';

type ConnectionLike = EventTarget & {
  effectiveType?: string;
  saveData?: boolean;
};

function slideAt(index: number, offset: number) {
  const length = homeHeroSlides.length;
  return homeHeroSlides[(index + offset + length) % length];
}

export function HomeHeroMedia({ children }: { children?: ReactNode }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [motionAllowed, setMotionAllowed] = useState(false);
  const [videoAllowed, setVideoAllowed] = useState(false);
  const [pageVisible, setPageVisible] = useState(true);
  const [carouselVisible, setCarouselVisible] = useState(true);
  const [videoReady, setVideoReady] = useState(false);
  const mountedRef = useRef(false);

  const isCarousel = homeHeroSlides.length > 1;
  const visibleSlides = useMemo(() => {
    if (!isCarousel) {
      const slide = homeHeroSlides[0];
      return [{ offset: 0 as const, slide, key: slide.id }];
    }

    return ([-1, 0, 1] as const).map((offset) => ({
      offset,
      slide: slideAt(activeIndex, offset),
      key: `${slideAt(activeIndex, offset).id}-${offset}`,
    }));
  }, [activeIndex, isCarousel]);

  const activeSlide = homeHeroSlides[activeIndex];
  const showVideo = activeSlide.kind === 'video' && videoAllowed && motionAllowed;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const compactViewport = window.matchMedia('(max-width: 640px)');
    const connection = (navigator as Navigator & { connection?: ConnectionLike }).connection;
    let videoTimer = 0;

    const mountVideoAfterLoad = () => {
      videoTimer = window.setTimeout(() => setVideoAllowed(true), 900);
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

    if (!motionAllowed || !pageVisible || !carouselVisible || !showVideo) {
      video.pause();
      return;
    }

    void video.play().catch(() => {
      // Poster remains visible if autoplay is blocked.
    });
  }, [carouselVisible, motionAllowed, pageVisible, showVideo, activeIndex]);

  const goPrev = useCallback(() => {
    setActiveIndex((current) => (current - 1 + homeHeroSlides.length) % homeHeroSlides.length);
    setVideoReady(false);
  }, []);

  const goNext = useCallback(() => {
    setActiveIndex((current) => (current + 1) % homeHeroSlides.length);
    setVideoReady(false);
  }, []);

  function renderSlideMedia(slide: HomeHeroSlide, isActive: boolean) {
    if (slide.kind === 'video') {
      return (
        <>
          <img className="wf-home-hero-card-poster" src={slide.poster} alt="" aria-hidden="true" />
          {isActive && showVideo ? (
            <video
              ref={videoRef}
              className={`wf-home-hero-card-video${videoReady ? ' is-ready' : ''}`}
              src={slide.src}
              poster={slide.poster}
              autoPlay
              loop
              muted
              playsInline
              preload="auto"
              aria-hidden="true"
              tabIndex={-1}
              onCanPlay={() => {
                if (mountedRef.current) setVideoReady(true);
              }}
            />
          ) : null}
        </>
      );
    }

    return (
      <img
        className="wf-home-hero-card-image"
        src={slide.src}
        alt=""
        aria-hidden="true"
        loading={isActive ? 'eager' : 'lazy'}
      />
    );
  }

  return (
    <div className="wf-home-hero-carousel-shell" aria-label="World model showcase">
      <div className="wf-home-hero-carousel">
        <div className="wf-home-hero-carousel-viewport">
          <div className="wf-home-hero-carousel-track">
            {visibleSlides.map(({ slide, offset, key }) => {
              const isActive = offset === 0;
              const showOverlay = isActive && children && slide.id === homeHeroSlides[0].id;
              return (
                <article
                  key={key}
                  className={`wf-home-hero-card${isActive ? ' is-active' : ''}${offset === -1 ? ' is-prev' : ''}${offset === 1 ? ' is-next' : ''}`}
                  aria-hidden={!isActive}
                >
                  <div className="wf-home-hero-card-frame">
                    <div className="wf-home-hero-card-media">
                      {renderSlideMedia(slide, isActive)}
                    </div>
                    {showOverlay ? (
                      <div className="wf-home-hero-card-overlay">{children}</div>
                    ) : null}
                  </div>
                </article>
              );
            })}
          </div>
        </div>

        <div className="wf-home-hero-carousel-caption" aria-live="polite">
          <strong>{activeSlide.label}</strong>
          <span>{activeSlide.caption}</span>
        </div>

        {isCarousel ? (
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
                    setVideoReady(false);
                  }}
                />
              ))}
            </div>
            <button type="button" className="wf-home-hero-carousel-nav" aria-label="Next demo" onClick={goNext}>
              <ChevronRight aria-hidden="true" size={18} strokeWidth={1.8} />
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
