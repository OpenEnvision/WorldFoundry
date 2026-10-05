import { BenchmarkLiveEmbed } from '@/components/benchmark-live-embed';
import { BenchmarkPaperFigures } from '@/components/benchmark-paper-figures';
import {
  getBenchmarkLeaderboard,
  leaderboardColumns,
  localizedEmptyReason,
  localizedLiveLabel,
  localizedPaperExtra,
  rowScore,
} from '@/lib/benchmark-leaderboards';
import {
  getBenchmarkEmbedFallbackFigures,
  getBenchmarkLeaderboardFigures,
  type BenchmarkPaperFigure,
} from '@/lib/benchmark-paper-figures';
import { liveEmbedCandidates, liveEmbedHost } from '@/lib/benchmark-live-embed';
import type { Locale } from '@/lib/i18n';

const copy = {
  en: {
    live: 'Live',
    liveNote: 'Official live board, embedded on this page. Not WorldFoundry-run scores.',
    openLive: 'Open live board',
    embedFailed:
      'This live board refused to embed, or the official Space is sleeping. Official paper figure and table below; open the live source for updates.',
    paperNote: 'Results from the paper (not a live board; not WorldFoundry-run scores).',
    emptyBoard: 'No official leaderboard ingested.',
    upstream: 'Upstream page',
    model: 'Model',
  },
  zh: {
    live: '实时',
    liveNote: '官方实时榜，嵌在本页。不是 WorldFoundry 跑分。',
    openLive: '打开实时榜',
    embedFailed: '实时榜拒绝嵌入，或官方 Space 处于休眠。下面是官方论文图与表；更新请看实时源。',
    paperNote: '论文报告结果（非实时榜；不是 WorldFoundry 跑分）。',
    emptyBoard: '尚未收录官方 leaderboard。',
    upstream: '上游页面',
    model: '模型',
  },
} as const;

function PaperCitation({
  board,
  extra,
}: {
  board: NonNullable<ReturnType<typeof getBenchmarkLeaderboard>>;
  extra: string;
}) {
  if (!board.paperUrl && !extra) return null;
  return (
    <>
      {extra ? <> {extra}</> : null}
      {board.paperUrl ? (
        <>
          {' '}
          <a href={board.paperUrl} rel="noreferrer">
            {board.paperTitle}
            {board.paperYear ? ` (${board.paperYear})` : ''}
          </a>
          .
        </>
      ) : null}
    </>
  );
}

function DetailTable({
  board,
  locale,
}: {
  board: NonNullable<ReturnType<typeof getBenchmarkLeaderboard>>;
  locale: Locale;
}) {
  const t = copy[locale];
  const entries = board.entries ?? [];
  if (entries.length === 0) return null;
  const columns = leaderboardColumns(board, locale);

  return (
    <div className="wf-benchmark-board-table-wrap">
      <table className="wf-benchmark-board-table">
        <thead>
          <tr>
            <th scope="col">{t.model}</th>
            {columns.map((column) => (
              <th key={column.id} scope="col">
                {locale === 'zh' ? column.labelZh || column.label : column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {entries.map((row, index) => (
            <tr key={`${row.model}:${index}`}>
              <th scope="row">
                <strong>{row.model}</strong>
                {row.note ? <small>{row.note}</small> : null}
              </th>
              {columns.map((column) => (
                <td key={column.id}>{rowScore(row, column.id)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BoardFigures({
  id,
  locale,
  extras,
  fallback = false,
}: {
  id: string;
  locale: Locale;
  extras?: BenchmarkPaperFigure[];
  fallback?: boolean;
}) {
  const catalog = fallback
    ? getBenchmarkEmbedFallbackFigures(id, locale)
    : getBenchmarkLeaderboardFigures(id, locale);
  const seen = new Set(catalog.map((figure) => figure.src));
  const extraFigures = (extras ?? []).filter((figure) => !seen.has(figure.src));
  if (catalog.length === 0 && extraFigures.length === 0) return null;
  return (
    <>
      {catalog.length > 0 ? (
        <BenchmarkPaperFigures benchmarkId={id} locale={locale} figures={catalog} />
      ) : null}
      {extraFigures.length > 0 ? (
        <BenchmarkPaperFigures benchmarkId={id} locale={locale} figures={extraFigures} />
      ) : null}
    </>
  );
}

function LiveFallback({
  id,
  locale,
  board,
  extraFigures,
  liveUrl,
  host,
  openLive,
}: {
  id: string;
  locale: Locale;
  board: NonNullable<ReturnType<typeof getBenchmarkLeaderboard>>;
  extraFigures: BenchmarkPaperFigure[];
  liveUrl?: string;
  host: string;
  openLive: string;
}) {
  return (
    <>
      <BoardFigures id={id} locale={locale} extras={extraFigures} fallback />
      <DetailTable board={board} locale={locale} />
      {liveUrl ? (
        <p className="wf-benchmark-board-live">
          <a href={liveUrl} rel="noreferrer">
            {openLive}
          </a>
          {host ? <span> · {host}</span> : null}
        </p>
      ) : null}
    </>
  );
}

export function BenchmarkLeaderboard({
  id,
  locale = 'en',
  fallbackUrl,
}: {
  id: string;
  locale?: Locale;
  fallbackUrl?: string;
}) {
  const board = getBenchmarkLeaderboard(id);
  const t = copy[locale];
  const extraFigures = (board?.paperFigures ?? []).map((figure) => ({
    role: 'results' as const,
    src: figure.src,
    alt: locale === 'zh' ? figure.captionZh || figure.caption || '' : figure.caption || '',
    caption: locale === 'zh' ? figure.captionZh || figure.caption || '' : figure.caption || '',
  }));

  if (!board) {
    return (
      <p className="wf-benchmark-section-empty">
        {t.emptyBoard}
        {fallbackUrl ? (
          <>
            {' '}
            <a href={fallbackUrl} rel="noreferrer">
              {t.upstream}
            </a>
          </>
        ) : null}
      </p>
    );
  }

  const liveUrl = board.liveUrl;
  const embedSrcs = board.kind === 'live' ? liveEmbedCandidates(board) : [];
  const liveLabel = localizedLiveLabel(board, locale);
  const host = liveUrl ? liveEmbedHost(liveUrl) : '';
  const paperExtra = localizedPaperExtra(board, locale);
  const emptyReason = localizedEmptyReason(board, locale);

  if (board.kind === 'live') {
    const fallback = (
      <LiveFallback
        id={id}
        locale={locale}
        board={board}
        extraFigures={extraFigures}
        liveUrl={liveUrl}
        host={host}
        openLive={t.openLive}
      />
    );
    if (embedSrcs.length === 0) {
      return (
        <>
          <p className="wf-benchmark-board-note">
            {t.embedFailed}
            {liveLabel ? <> {liveLabel}.</> : null}
          </p>
          {fallback}
        </>
      );
    }
    return (
      <>
        <p className="wf-benchmark-board-note">
          {t.liveNote}
          {liveLabel ? <> {liveLabel}.</> : null}
        </p>
        <BenchmarkLiveEmbed
          srcs={embedSrcs}
          title={liveLabel || t.live}
          caption={`${host} · ${t.live}`}
          openHref={liveUrl || embedSrcs[0]}
          openLabel={t.openLive}
          failedNote={t.embedFailed}
          fallback={fallback}
        />
      </>
    );
  }

  if (board.kind === 'empty') {
    return (
      <>
        <p className="wf-benchmark-section-empty">
          {emptyReason || t.emptyBoard}
          <PaperCitation board={board} extra="" />
        </p>
        <BoardFigures id={id} locale={locale} extras={extraFigures} />
      </>
    );
  }

  return (
    <>
      <p className="wf-benchmark-board-note">
        {t.paperNote}
        <PaperCitation board={board} extra={paperExtra} />
      </p>
      <BoardFigures id={id} locale={locale} extras={extraFigures} />
      <DetailTable board={board} locale={locale} />
    </>
  );
}
