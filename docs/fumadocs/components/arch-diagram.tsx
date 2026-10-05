import {
  ARCH_DIAGRAMS,
  archText,
  type ArchBoundarySpec,
  type ArchCallSpec,
  type ArchDiagramId,
  type ArchDiagramSpec,
  type ArchLocale,
  type ArchNode,
  type ArchSequenceSpec,
  type ArchSplitSpec,
} from '@/lib/arch-diagrams';
import type { CSSProperties, ReactNode } from 'react';

function t(text: Parameters<typeof archText>[0], locale: ArchLocale): string {
  return archText(text, locale);
}

function NodeCard({
  node,
  locale,
  index,
}: {
  node: ArchNode;
  locale: ArchLocale;
  index?: number;
}) {
  return (
    <article className={`wf-arch-card${node.optional ? ' wf-arch-card--optional' : ''}`}>
      <div className="wf-arch-card-head">
        {typeof index === 'number' ? (
          <span className="wf-arch-idx" aria-hidden="true">
            {String(index).padStart(2, '0')}
          </span>
        ) : null}
        {node.role ? <span className="wf-arch-role">{t(node.role, locale)}</span> : null}
        {node.optional ? (
          <span className="wf-arch-optional">{locale === 'zh' ? '可选' : 'optional'}</span>
        ) : null}
      </div>
      <div className="wf-arch-card-title">{t(node.title, locale)}</div>
      {node.detail ? <p className="wf-arch-card-detail">{t(node.detail, locale)}</p> : null}
      {node.file ? <p className="wf-arch-file">{t(node.file, locale)}</p> : null}
      {node.steps && node.steps.length > 0 ? (
        <ol className="wf-arch-micro">
          {node.steps.map((step, stepIndex) => (
            <li key={`${t(step, locale)}-${stepIndex}`}>
              <span>{t(step, locale)}</span>
            </li>
          ))}
        </ol>
      ) : null}
    </article>
  );
}

function CallLabel({
  label,
  file,
  locale,
}: {
  label?: ArchNode['title'];
  file?: ArchNode['file'];
  locale: ArchLocale;
}) {
  const text = label ? t(label, locale) : '';
  return (
    <div className={`wf-arch-call${text ? '' : ' wf-arch-call--bare'}`}>
      <span className="wf-arch-call-shaft" aria-hidden="true" />
      {text ? (
        <div className="wf-arch-call-body">
          <code>{text}</code>
          {file ? <span>{t(file, locale)}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

type GraphEdge = {
  file?: ArchNode['file'];
  label: ArchNode['title'];
};

type GraphVertex = {
  edge?: GraphEdge;
  id: string;
  index: number;
  node: ArchNode;
};

type GraphRank = {
  edgeDown?: GraphEdge;
  fork?: boolean;
  forkNote?: ArchNode['detail'];
  key: string;
  label?: string;
  nodes: GraphVertex[];
};

function buildCallRanks(spec: ArchCallSpec, locale: ArchLocale): GraphRank[] {
  const hasLanes = spec.steps.some((step) => step.type === 'lane');
  const ranks: GraphRank[] = [];
  let current: GraphRank = { key: 'rank-0', nodes: [] };
  let pending: GraphEdge | undefined;
  let index = 1;
  let rankCount = 0;

  const flush = () => {
    if (current.nodes.length > 0 || current.label) {
      ranks.push(current);
      rankCount += 1;
    }
    current = { key: `rank-${rankCount}`, nodes: [] };
  };

  for (const [stepIndex, step] of spec.steps.entries()) {
    if (step.type === 'lane') {
      if (pending && current.nodes.length > 0) {
        current.edgeDown = pending;
        pending = undefined;
      }
      flush();
      current.label = t(step.label, locale);
      continue;
    }

    if (step.type === 'call') {
      pending = { file: step.file, label: step.label };
      continue;
    }

    if (step.type === 'fork') {
      if (pending && current.nodes.length > 0) {
        current.edgeDown = pending;
        pending = undefined;
      }
      if (current.nodes.length > 0) {
        flush();
      }
      current.nodes = step.ids
        .map((id) => spec.nodes[id])
        .filter((node): node is ArchNode => Boolean(node))
        .map((node, forkIndex) => {
          const vertexIndex = index;
          index += 1;
          return { id: `${step.ids[forkIndex]}-${stepIndex}`, index: vertexIndex, node };
        });
      current.fork = true;
      current.forkNote = step.note;
      current.key = `fork-${stepIndex}`;
      continue;
    }

    const node = spec.nodes[step.id];
    if (!node) {
      continue;
    }

    if (current.fork && current.nodes.length > 0) {
      if (pending) {
        current.edgeDown = pending;
        pending = undefined;
      }
      flush();
    }

    if (!hasLanes && current.nodes.length > 0) {
      if (pending) {
        current.edgeDown = pending;
        pending = undefined;
      }
      flush();
    }

    current.nodes.push({ id: `${step.id}-${stepIndex}`, index, node });
    index += 1;

    if (pending && current.nodes.length > 1) {
      current.nodes[current.nodes.length - 2] = {
        ...current.nodes[current.nodes.length - 2],
        edge: pending,
      };
      pending = undefined;
    }
  }

  if (pending && current.nodes.length > 0) {
    current.edgeDown = pending;
  }
  flush();
  return ranks;
}

function shortFile(path: string) {
  const [file, line] = path.split(/(?=:)/);
  const parts = file.split('/');
  const base = parts.length <= 2 ? file : parts.slice(-2).join('/');
  return line ? `${base}${line}` : base;
}

function sameLabel(left: string, right: string) {
  return left.replace(/\s+/g, '').toLowerCase() === right.replace(/\s+/g, '').toLowerCase();
}

function incomingEdge(
  edge: GraphEdge | undefined,
  next: GraphVertex | undefined,
  locale: ArchLocale,
): GraphEdge | undefined {
  if (!edge || !next) {
    return edge;
  }
  if (sameLabel(t(edge.label, locale), t(next.node.title, locale))) {
    return { ...edge, label: '' };
  }
  return edge;
}

function chunkRanks(ranks: GraphRank[]): GraphRank[] {
  const out: GraphRank[] = [];

  for (const rank of ranks) {
    if (rank.fork || rank.nodes.length <= 3) {
      out.push(rank);
      continue;
    }

    const size = rank.nodes.length % 2 === 0 ? 2 : 3;
    for (let offset = 0; offset < rank.nodes.length; offset += size) {
      const nodes = rank.nodes.slice(offset, offset + size);
      const last = nodes[nodes.length - 1];
      const lastChunk = offset + size >= rank.nodes.length;
      out.push({
        edgeDown: lastChunk ? rank.edgeDown : last.edge,
        key: `${rank.key}-${offset}`,
        label: offset === 0 ? rank.label : undefined,
        nodes: lastChunk
          ? nodes
          : nodes.map((node, nodeIndex) =>
              nodeIndex === nodes.length - 1 ? { ...node, edge: undefined } : node,
            ),
      });
    }
  }

  return out;
}

function GraphBox({ locale, vertex }: { locale: ArchLocale; vertex: GraphVertex }) {
  const { node } = vertex;
  const detail = node.detail ? t(node.detail, locale) : undefined;
  const file = node.file ? t(node.file, locale) : undefined;
  const tip = [file, detail].filter(Boolean).join('\n');

  return (
    <div
      className={`wf-cg-node${node.optional ? ' wf-cg-node--optional' : ''}`}
      title={tip || undefined}
    >
      {node.role ? <span className="wf-cg-kicker">{t(node.role, locale)}</span> : null}
      <strong className="wf-cg-name">{t(node.title, locale)}</strong>
      {file ? <span className="wf-cg-file">{shortFile(file)}</span> : null}
    </div>
  );
}

function GraphEdgeLabel({
  edge,
  locale,
  vertical,
}: {
  edge?: GraphEdge;
  locale: ArchLocale;
  vertical?: boolean;
}) {
  const label = edge ? t(edge.label, locale) : '';
  const tip = edge?.file ? t(edge.file, locale) : undefined;

  return (
    <div
      className={`wf-cg-edge${vertical ? ' wf-cg-edge--down' : ''}${label ? '' : ' wf-cg-edge--bare'}`}
      title={tip}
    >
      {vertical ? (
        <>
          {label ? <code>{label}</code> : null}
          <span className="wf-cg-vline" aria-hidden="true">
            <span className="wf-cg-shaft" />
            <span className="wf-cg-arrow" />
          </span>
        </>
      ) : (
        <>
          <span className="wf-cg-shaft" aria-hidden="true" />
          {label ? <code>{label}</code> : null}
          <span className="wf-cg-shaft wf-cg-shaft--tail" aria-hidden="true" />
          <span className="wf-cg-arrow" aria-hidden="true" />
        </>
      )}
    </div>
  );
}

function CallGraph({ spec, locale }: { spec: ArchCallSpec; locale: ArchLocale }) {
  const ranks = chunkRanks(buildCallRanks(spec, locale));

  return (
    <div className="wf-cg">
      {ranks.map((rank, rankIndex) => (
        <div className="wf-cg-rank" key={rank.key}>
          {rank.label ? <div className="wf-cg-lane">{rank.label}</div> : null}
          <div className="wf-cg-spine">
            <div
              className={`wf-cg-row${rank.fork ? ' wf-cg-row--fork' : ''}${rank.nodes.length > 1 ? ' wf-cg-row--multi' : ''}`}
              data-count={rank.nodes.length}
            >
              {rank.nodes.flatMap((vertex, vertexIndex) => {
                const items = [
                  <GraphBox key={vertex.id} locale={locale} vertex={vertex} />,
                ];
                if (vertexIndex > 0 && !rank.fork) {
                  items.unshift(
                    <GraphEdgeLabel
                      edge={incomingEdge(rank.nodes[vertexIndex - 1].edge, vertex, locale)}
                      key={`${vertex.id}-in`}
                      locale={locale}
                    />,
                  );
                }
                return items;
              })}
            </div>
            {rank.forkNote ? <p className="wf-cg-note">{t(rank.forkNote, locale)}</p> : null}
            {rankIndex < ranks.length - 1 ? (
              <GraphEdgeLabel
                edge={incomingEdge(rank.edgeDown, ranks[rankIndex + 1].nodes[0], locale)}
                locale={locale}
                vertical
              />
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}

function SeqCaption({
  file,
  label,
  locale,
}: {
  file?: ArchNode['file'];
  label: ArchNode['title'];
  locale: ArchLocale;
}) {
  return (
    <div className="wf-arch-seq-cap">
      <span className="wf-arch-seq-msg-label">{t(label, locale)}</span>
      {file ? <span className="wf-arch-seq-msg-file">{t(file, locale)}</span> : null}
    </div>
  );
}

function SequenceDiagram({ spec, locale }: { spec: ArchSequenceSpec; locale: ArchLocale }) {
  const n = spec.actors.length;
  return (
    <div className="wf-arch-seq" style={{ '--wf-arch-actors': n } as CSSProperties}>
      <div className="wf-arch-seq-actors">
        {spec.actors.map((actor, actorIndex) => (
          <div className="wf-arch-seq-actor" key={`${t(actor.label, locale)}-${actorIndex}`}>
            <strong>{t(actor.label, locale)}</strong>
            {actor.detail ? <span>{t(actor.detail, locale)}</span> : null}
          </div>
        ))}
      </div>
      <div className="wf-arch-seq-stage">
        {spec.actors.map((actor, actorIndex) => (
          <span
            aria-hidden="true"
            className="wf-arch-seq-life"
            key={`life-${t(actor.label, locale)}-${actorIndex}`}
            style={{ left: `${((actorIndex + 0.5) / n) * 100}%` }}
          />
        ))}
        {spec.messages.map((message, messageIndex) => {
          const isSelf = message.kind === 'self' || message.from === message.to;
          if (isSelf) {
            return (
              <div
                className="wf-arch-seq-self"
                key={`msg-${messageIndex}`}
                style={{ marginLeft: `${((message.from + 0.5) / n) * 100}%` }}
              >
                <SeqCaption file={message.file} label={message.label} locale={locale} />
                <span className="wf-arch-seq-self-loop" aria-hidden="true" />
              </div>
            );
          }
          const leftIndex = Math.min(message.from, message.to);
          const span = Math.abs(message.to - message.from);
          const backward = message.to < message.from || message.kind === 'return';
          return (
            <div
              className={`wf-arch-seq-msg${backward ? ' wf-arch-seq-msg--return' : ''}`}
              key={`msg-${messageIndex}`}
              style={{
                marginLeft: `${((leftIndex + 0.5) / n) * 100}%`,
                width: `${(span / n) * 100}%`,
              }}
            >
              <SeqCaption file={message.file} label={message.label} locale={locale} />
              <span className="wf-arch-seq-arrow" aria-hidden="true" />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function BoundaryStack({ locale, nodes }: { locale: ArchLocale; nodes: ArchNode[] }) {
  return (
    <div className="wf-cg wf-bd-stack">
      {nodes.map((node, index) => {
        const vertex: GraphVertex = {
          id: `${index}-${t(node.title, locale)}`,
          index: index + 1,
          node,
        };
        return (
          <div className="wf-cg-rank" key={vertex.id}>
            <div className="wf-cg-spine">
              <div className="wf-cg-row" data-count="1">
                <GraphBox locale={locale} vertex={vertex} />
              </div>
              {index < nodes.length - 1 ? (
                <GraphEdgeLabel locale={locale} vertical />
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function BoundaryDiagram({ spec, locale }: { spec: ArchBoundarySpec; locale: ArchLocale }) {
  return (
    <div className="wf-bd">
      <div className="wf-bd-col">
        <div className="wf-cg-lane">{t(spec.leftTitle, locale)}</div>
        <BoundaryStack locale={locale} nodes={spec.left} />
        <p className="wf-bd-cap">{t(spec.leftCaption, locale)}</p>
      </div>
      <div className="wf-bd-rail">
        <span className="wf-bd-rail-label">{t(spec.boundaryLabel, locale)}</span>
        <span className="wf-bd-rail-line" aria-hidden="true" />
        <div className="wf-bd-cross">
          <span className="wf-bd-arm" aria-hidden="true" />
          <div className="wf-cg-node wf-bd-port">
            <strong className="wf-cg-name">{t(spec.crossMain, locale)}</strong>
            <span className="wf-cg-file">{t(spec.crossSub, locale)}</span>
          </div>
          <span className="wf-bd-arm wf-bd-arm--out" aria-hidden="true" />
        </div>
      </div>
      <div className="wf-bd-col">
        <div className="wf-cg-lane">{t(spec.rightTitle, locale)}</div>
        <BoundaryStack locale={locale} nodes={spec.right} />
        <p className="wf-bd-cap">{t(spec.rightCaption, locale)}</p>
      </div>
    </div>
  );
}

function SplitDiagram({ spec, locale }: { spec: ArchSplitSpec; locale: ArchLocale }) {
  return (
    <div className="wf-arch-split">
      <div className="wf-arch-split-cols">
        <div className="wf-arch-split-col">
          <div className="wf-arch-col-title">{t(spec.leftTitle, locale)}</div>
          <CallGraph locale={locale} spec={spec.left} />
        </div>
        <div className="wf-arch-split-col">
          <div className="wf-arch-col-title">{t(spec.rightTitle, locale)}</div>
          <CallGraph locale={locale} spec={spec.right} />
        </div>
      </div>
      <div className="wf-arch-split-join" aria-hidden="true" />
      {spec.mergeCall ? (
        <CallLabel file={spec.mergeCall.file} label={spec.mergeCall.label} locale={locale} />
      ) : (
        <div className="wf-arch-call">
          <span className="wf-arch-call-shaft" aria-hidden="true" />
        </div>
      )}
      <div className="wf-arch-step">
        <NodeCard locale={locale} node={spec.merge} />
      </div>
    </div>
  );
}

function DiagramBody({ spec, locale }: { spec: ArchDiagramSpec; locale: ArchLocale }) {
  if (spec.kind === 'call') {
    return <CallGraph locale={locale} spec={spec} />;
  }
  if (spec.kind === 'sequence') {
    return <SequenceDiagram locale={locale} spec={spec} />;
  }
  if (spec.kind === 'boundary') {
    return <BoundaryDiagram locale={locale} spec={spec} />;
  }
  return <SplitDiagram locale={locale} spec={spec} />;
}

export function ArchDiagram({
  id,
  locale = 'en',
}: {
  id: ArchDiagramId;
  locale?: ArchLocale;
}) {
  const spec = ARCH_DIAGRAMS[id];
  if (!spec) {
    return null;
  }

  return (
    <figure className={`wf-arch wf-arch--${spec.kind}`} aria-label={t(spec.aria, locale)}>
      <header className="wf-arch-head">
        <span className="wf-arch-kind">{t(spec.kindLabel, locale)}</span>
        <div className="wf-arch-title">{t(spec.title, locale)}</div>
      </header>
      <div className="wf-arch-body">
        <DiagramBody locale={locale} spec={spec} />
      </div>
      {spec.caption ? <figcaption>{t(spec.caption, locale)}</figcaption> : null}
    </figure>
  );
}
