import { ExternalLink, FileCode2, Pencil } from 'lucide-react';

type DocsPageActionsProps = {
  editLabel: string;
  githubUrl: string;
  lastUpdated?: {
    formatted: string;
    iso: string;
  } | null;
  lastUpdatedLabel: string;
  markdownLabel: string;
  markdownUrl: string;
};

export function DocsPageActions({
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
        <a className="pi-doc-action-chip" href={editUrl} rel="noreferrer" target="_blank">
          <Pencil aria-hidden="true" size={14} strokeWidth={2} />
          <span>{editLabel}</span>
        </a>
        <a className="pi-doc-action-chip" href={githubUrl} rel="noreferrer" target="_blank">
          <ExternalLink aria-hidden="true" size={14} strokeWidth={2} />
          <span>GitHub</span>
        </a>
        <a className="pi-doc-action-chip" href={markdownUrl}>
          <FileCode2 aria-hidden="true" size={14} strokeWidth={2} />
          <span>{markdownLabel}</span>
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
