import { Provider } from '@/components/provider';
import { ThemeBootstrap } from '@/components/theme-bootstrap';
import { brandDisplayFont } from '@/lib/brand-font';
import { docsSansFont } from '@/lib/docs-font';
import { withBasePath } from '@/lib/site-path';
import 'fumadocs-ui/components/image-zoom2.css';
import './styles/tokens.css';
import './global.css';
import './fd-skin.css';
import './arch-diagram.css';
import type { Metadata } from 'next';

const faviconPath = withBasePath('/favicon.svg') ?? '/favicon.svg';

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? 'http://localhost:3000'),
  title: {
    default: 'WorldFoundry Docs',
    template: '%s | WorldFoundry',
  },
  description:
    'Open-source infrastructure for world intelligence: discover, run, inspect, and evaluate video, 3D/4D, interactive-world, and embodied models under shared contracts.',
  icons: {
    icon: [{ url: faviconPath, type: 'image/svg+xml' }],
    shortcut: faviconPath,
  },
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${brandDisplayFont.variable} ${docsSansFont.variable}`}
    >
      <body className="flex flex-col min-h-screen" suppressHydrationWarning>
        <ThemeBootstrap />
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
