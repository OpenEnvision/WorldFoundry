import catalog from '@/lib/benchmark-leaderboards.json';
import type { Locale } from '@/lib/i18n';

export type BenchmarkLeaderboardKind = 'live' | 'paper' | 'empty';

export type BenchmarkLeaderboardColumn = {
  id: string;
  label: string;
  labelZh?: string;
};

export type BenchmarkLeaderboardRow = {
  rank?: number;
  model: string;
  score?: string;
  scores?: Record<string, string>;
  note?: string;
};

export type BenchmarkLeaderboardFigure = {
  src: string;
  caption?: string;
  captionZh?: string;
};

export type BenchmarkLeaderboardBoard = {
  kind: BenchmarkLeaderboardKind;
  liveUrl?: string;
  liveEmbedUrl?: string;
  liveLabel?: string;
  liveLabelZh?: string;
  paperTitle?: string;
  paperYear?: string;
  paperUrl?: string;
  paperNote?: string;
  paperNoteZh?: string;
  paperFigures?: BenchmarkLeaderboardFigure[];
  metric?: string;
  metricZh?: string;
  columns?: BenchmarkLeaderboardColumn[];
  entries?: BenchmarkLeaderboardRow[];
  emptyReason?: string;
  emptyReasonZh?: string;
};

type LeaderboardFile = {
  version: number;
  note: string;
  boards: Record<string, BenchmarkLeaderboardBoard>;
};

const data = catalog as LeaderboardFile;

export function getBenchmarkLeaderboard(id: string): BenchmarkLeaderboardBoard | undefined {
  return data.boards[id];
}

export function localizedLiveLabel(board: BenchmarkLeaderboardBoard, locale: Locale) {
  return locale === 'zh' ? board.liveLabelZh || board.liveLabel || '' : board.liveLabel || '';
}

export function localizedMetric(board: BenchmarkLeaderboardBoard, locale: Locale) {
  return locale === 'zh' ? board.metricZh || board.metric || '' : board.metric || '';
}

export function localizedEmptyReason(board: BenchmarkLeaderboardBoard, locale: Locale) {
  return locale === 'zh' ? board.emptyReasonZh || board.emptyReason || '' : board.emptyReason || '';
}

export function localizedPaperExtra(board: BenchmarkLeaderboardBoard, locale: Locale) {
  return locale === 'zh' ? board.paperNoteZh || board.paperNote || '' : board.paperNote || '';
}

export function localizedColumnLabel(column: BenchmarkLeaderboardColumn, locale: Locale) {
  return locale === 'zh' ? column.labelZh || column.label : column.label;
}

export function leaderboardColumns(board: BenchmarkLeaderboardBoard, locale: Locale): BenchmarkLeaderboardColumn[] {
  if (board.columns && board.columns.length > 0) return board.columns;
  const metric = localizedMetric(board, locale);
  if (metric) return [{ id: 'score', label: metric }];
  const hasScores = (board.entries ?? []).some((row) => row.scores && Object.keys(row.scores).length > 0);
  if (hasScores) {
    const ids = new Set<string>();
    for (const row of board.entries ?? []) {
      for (const key of Object.keys(row.scores ?? {})) ids.add(key);
    }
    return [...ids].map((id) => ({ id, label: id }));
  }
  return [{ id: 'score', label: locale === 'zh' ? '分数' : 'Score' }];
}

export function rowScore(row: BenchmarkLeaderboardRow, columnId: string): string {
  if (row.scores?.[columnId]) return row.scores[columnId];
  if (columnId === 'score' && row.score) return row.score;
  return '';
}
