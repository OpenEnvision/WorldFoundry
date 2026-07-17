import { EcosystemPage } from '@/components/ecosystem-page';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Blog',
  description: 'WorldFoundry updates, technical notes, release notes, and integration writeups.',
};

export default function BlogPage() {
  return (
    <EcosystemPage
      active="blog"
      comingSoon="Updates, technical notes, and release highlights will be posted here soon."
      description="Updates, technical notes, and release highlights from the WorldFoundry team."
      footerLabel="Blog"
      label="Project notes"
      title="Blog"
    />
  );
}
