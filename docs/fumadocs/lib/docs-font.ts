import { GeistMono } from 'geist/font/mono';
import { GeistSans } from 'geist/font/sans';

/** Docs UI sans — matches Fumadocs (Geist), with Noto Sans SC as CJK fallback in CSS. */
export const docsSansFont = GeistSans;

/** Code and other monospace text; bundled so it renders the same on every OS. */
export const docsMonoFont = GeistMono;
