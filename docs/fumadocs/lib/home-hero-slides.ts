import { withBasePath } from '@/lib/site-path';

// Keep the original showcase video on its published blob instead of adding it
// to the Pages artifact.
const WORLDFOUNDRY_SHOWCASE_URL =
  'https://raw.githubusercontent.com/OpenEnvision/WorldFoundry/ef8001ec617d9256ee439f2f3452b27b87e3b166/docs/fumadocs/public/cover_4x4_hero.mp4';

export type HomeHeroSlide = {
  id: string;
  label: string;
  src: string;
  poster: string;
  hasAudio?: boolean;
};

function mediaPath(path: string) {
  return withBasePath(path) ?? path;
}

export const homeHeroSlides: HomeHeroSlide[] = [
  {
    id: 'teaser-2',
    label: 'World models · 02',
    src: mediaPath('/teaser2.mp4'),
    poster: mediaPath('/teaser2-poster.jpg'),
  },
  {
    id: 'worldfoundry',
    label: 'WorldFoundry',
    src: WORLDFOUNDRY_SHOWCASE_URL,
    poster: mediaPath('/cover_4x4_hero-poster.webp'),
  },
  {
    id: 'teaser-1',
    label: 'World models · 01',
    src: mediaPath('/teaser1.mp4'),
    poster: mediaPath('/teaser1-poster.jpg'),
    hasAudio: true,
  },
];
