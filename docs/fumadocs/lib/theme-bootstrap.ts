/**
 * Blocking theme bootstrap injected into <head> via useServerInsertedHTML.
 * next-themes' in-tree <script> is marked as a data block under React 19, so
 * FOUC prevention must live here instead.
 * Matches fumadocs RootProvider defaults: attribute=class, defaultTheme=system.
 */
export const THEME_BOOTSTRAP_SCRIPT = `(function(){try{var s=localStorage.getItem('theme'),m=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light',t=!s||s==='system'?m:s;document.documentElement.classList.remove('light','dark');document.documentElement.classList.add(t);document.documentElement.style.colorScheme=t}catch(e){}})();`;
