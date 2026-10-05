import { WORLDFOUNDRY_GITHUB_REPO } from '@/lib/site-links';

/** Inline mark — lucide-react v1 no longer ships a GitHub icon. */
function GitHubMark({ size = 20, strokeWidth = 1.8 }: { size?: number; strokeWidth?: number }) {
  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65S8.93 17.38 9 18v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}

export function SiteGitHubLink() {
  return (
    <a
      href={WORLDFOUNDRY_GITHUB_REPO}
      className="pi-github-link"
      target="_blank"
      rel="noreferrer noopener"
      aria-label="GitHub"
    >
      <GitHubMark />
    </a>
  );
}
