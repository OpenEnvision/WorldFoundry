import { DocsCopyMarkdownButton } from '@/components/docs-page-action-buttons';

type DocsPageActionsProps = {
  copyMarkdownCopiedLabel: string;
  copyMarkdownLabel: string;
  editLabel: string;
  editUrl: string;
  lastUpdated?: {
    formatted: string;
    iso: string;
  } | null;
  lastUpdatedLabel: string;
  markdownLabel: string;
  markdownUrl: string;
};

export function DocsPageActions({
  copyMarkdownCopiedLabel,
  copyMarkdownLabel,
  editLabel,
  editUrl,
  lastUpdated,
  lastUpdatedLabel,
  markdownLabel,
  markdownUrl,
}: DocsPageActionsProps) {
  return (
    <div className="pi-doc-actions" aria-label="Document actions">
      <div className="pi-doc-actions-links">
        <DocsCopyMarkdownButton
          copiedLabel={copyMarkdownCopiedLabel}
          label={copyMarkdownLabel}
          markdownUrl={markdownUrl}
        />
        <a className="pi-doc-action-link" href={markdownUrl} rel="noreferrer" target="_blank">
          {markdownLabel}
        </a>
        <a className="pi-doc-action-link" href={editUrl} rel="noreferrer" target="_blank">
          {editLabel}
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
