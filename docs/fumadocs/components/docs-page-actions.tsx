import { DocsCopyMarkdownButton } from '@/components/docs-page-action-buttons';

type DocsPageActionsProps = {
  copyMarkdownCopiedLabel: string;
  copyMarkdownLabel: string;
  editLabel: string;
  githubUrl: string;
  lastUpdated?: {
    formatted: string;
    iso: string;
  } | null;
  lastUpdatedLabel: string;
  markdownLabel: string;
  markdownUrl: string;
  pageTitle: string;
};

export function DocsPageActions({
  copyMarkdownCopiedLabel,
  copyMarkdownLabel,
  editLabel,
  githubUrl,
  lastUpdated,
  lastUpdatedLabel,
  markdownLabel,
  markdownUrl,
}: DocsPageActionsProps) {
  const editUrl = githubUrl.replace('/blob/', '/edit/');

  return (
    <div className="pi-doc-actions" aria-label="Document actions">
      <div className="pi-doc-actions-links">
        <DocsCopyMarkdownButton
          copiedLabel={copyMarkdownCopiedLabel}
          label={copyMarkdownLabel}
          markdownUrl={markdownUrl}
        />
        <a className="pi-doc-action-link" href={editUrl} rel="noreferrer" target="_blank">
          {editLabel}
        </a>
        <a className="pi-doc-action-link" href={githubUrl} rel="noreferrer" target="_blank">
          GitHub
        </a>
        <a className="pi-doc-action-link" href={markdownUrl} rel="noreferrer" target="_blank">
          {markdownLabel}
        </a>
      </div>
      {lastUpdated ? (
        <p className="pi-doc-last-updated">
          <time dateTime={lastUpdated.iso}>
            {lastUpdatedLabel}: {lastUpdated.formatted}
          </time>
        </p>
      ) : null}
    </div>
  );
}
