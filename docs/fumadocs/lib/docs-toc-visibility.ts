import type { TOCItemType } from 'fumadocs-core/toc';

const TOP_LEVEL_DEPTH = 2;
const BRANCH_DEPTH = 3;

function getHeadingId(url: string) {
  return url.startsWith('#') ? url.slice(1) : null;
}

export type DocsTocVisibility = {
  visible: ReadonlySet<number>;
  branchActive: ReadonlySet<number>;
};

/**
 * vLLM-style TOC disclosure: keep top-level sections visible, and only expand
 * the branch that contains the currently scrolled heading (class + methods, etc.).
 */
export function computeDocsTocVisibility(
  items: TOCItemType[],
  activeId: string | null,
): DocsTocVisibility {
  const visible = new Set<number>();
  const branchActive = new Set<number>();

  const meta = items.map((item, index) => ({
    index,
    depth: item.depth,
    id: getHeadingId(item.url),
  }));

  for (const { index, depth } of meta) {
    if (depth <= TOP_LEVEL_DEPTH) {
      visible.add(index);
    }
  }

  if (!activeId) {
    return { visible, branchActive };
  }

  const activeIndex = meta.findIndex((entry) => entry.id === activeId);
  if (activeIndex < 0) {
    return { visible, branchActive };
  }

  const activeDepth = meta[activeIndex].depth;
  if (activeDepth <= TOP_LEVEL_DEPTH) {
    return { visible, branchActive };
  }

  const ancestors: number[] = [activeIndex];
  let cursorDepth = activeDepth;
  for (let index = activeIndex - 1; index >= 0; index -= 1) {
    if (meta[index].depth < cursorDepth) {
      ancestors.unshift(index);
      cursorDepth = meta[index].depth;
    }
  }

  for (const index of ancestors) {
    if (meta[index].depth >= BRANCH_DEPTH) {
      branchActive.add(index);
    }
  }

  const branchRoot =
    ancestors.find((index) => meta[index].depth === BRANCH_DEPTH) ??
    ancestors.find((index) => meta[index].depth > TOP_LEVEL_DEPTH) ??
    activeIndex;

  const rootDepth = meta[branchRoot].depth;
  for (let index = branchRoot; index < meta.length; index += 1) {
    if (index > branchRoot && meta[index].depth <= rootDepth) {
      break;
    }
    visible.add(index);
  }

  return { visible, branchActive };
}
