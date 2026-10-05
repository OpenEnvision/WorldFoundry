import { withBasePath, withMediaPath } from '@/lib/site-path';

export type HomeHeroSlide =
  | {
      id: string;
      kind: 'video';
      label: string;
      caption: string;
      src: string;
      poster: string;
    }
  | {
      id: string;
      kind: 'image';
      label: string;
      caption: string;
      src: string;
    };

export const homeHeroSlides: HomeHeroSlide[] = [
  {
    id: 'showcase',
    kind: 'video',
    label: 'World models',
    caption: 'Unified inference and evaluation across video, 3D/4D, embodied, and interactive worlds.',
    src: withMediaPath('/cover_4x4_hero.mp4') ?? '/cover_4x4_hero.mp4',
    poster: withBasePath('/cover_4x4_hero-poster.webp') ?? '/cover_4x4_hero-poster.webp',
  },
];
