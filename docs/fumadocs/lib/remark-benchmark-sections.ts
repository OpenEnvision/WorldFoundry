type MdastNode = {
  type: string;
  depth?: number;
  name?: string;
  value?: string;
  children?: MdastNode[];
  attributes?: Array<{ type?: string; name?: string; value?: unknown }>;
};

type MdastRoot = {
  type: 'root';
  children: MdastNode[];
};

const SKIP_FILES = new Set(['index', 'runtime-environments']);
const ABOUT_HEADINGS = new Set(['about', '简介']);

function filePath(file: { path?: string; history?: string[] }) {
  return file.path || file.history?.[0] || '';
}

function benchmarkIdFromPath(path: string) {
  const match = path.match(/evaluation\/benchmark-hub\/([^/]+?)(?:\.zh)?\.mdx$/);
  if (!match) return undefined;
  const id = match[1];
  if (SKIP_FILES.has(id)) return undefined;
  return id;
}

function headingText(node: MdastNode) {
  if (node.type !== 'heading') return '';
  const parts: string[] = [];
  const walk = (item: MdastNode) => {
    if (item.value) parts.push(item.value);
    item.children?.forEach(walk);
  };
  walk(node);
  return parts.join('').trim();
}

function isAboutHeading(node: MdastNode) {
  return node.type === 'heading' && node.depth === 2 && ABOUT_HEADINGS.has(headingText(node).toLowerCase());
}

function isSectionHeading(node: MdastNode) {
  return node.type === 'heading' && (node.depth ?? 0) <= 2;
}

function classNameOf(node: MdastNode) {
  const attr = node.attributes?.find((item) => item.name === 'className');
  return typeof attr?.value === 'string' ? attr.value : '';
}

function isLegacyMetricCatalog(node: MdastNode) {
  return node.type === 'mdxJsxFlowElement' && node.name === 'div' && classNameOf(node).includes('pi-kv-catalog');
}

function alreadyInserted(children: MdastNode[]) {
  return children.some(
    (node) => node.type === 'mdxJsxFlowElement' && node.name === 'BenchmarkPageSections',
  );
}

function sectionsNode(id: string, locale: 'en' | 'zh'): MdastNode {
  return {
    type: 'mdxJsxFlowElement',
    name: 'BenchmarkPageSections',
    attributes: [
      { type: 'mdxJsxAttribute', name: 'id', value: id },
      { type: 'mdxJsxAttribute', name: 'locale', value: locale },
    ],
    children: [],
  };
}

export function remarkBenchmarkSections() {
  return (tree: MdastRoot, file: { path?: string; history?: string[] }) => {
    const path = filePath(file);
    const id = benchmarkIdFromPath(path);
    if (!id || !tree.children) return;

    tree.children = tree.children.filter((node) => !isLegacyMetricCatalog(node));
    if (alreadyInserted(tree.children)) return;

    const locale = path.endsWith('.zh.mdx') ? 'zh' : 'en';
    const aboutIndex = tree.children.findIndex(isAboutHeading);
    if (aboutIndex < 0) {
      tree.children.unshift(sectionsNode(id, locale));
      return;
    }

    let insertAt = tree.children.length;
    for (let index = aboutIndex + 1; index < tree.children.length; index += 1) {
      if (isSectionHeading(tree.children[index])) {
        insertAt = index;
        break;
      }
    }
    tree.children.splice(insertAt, 0, sectionsNode(id, locale));
  };
}
