import apiReference from '@/generated/python-api.json';
import { methodAnchor, symbolAnchor } from '@/lib/docs-api-toc';
import { withBasePath } from '@/lib/site-path';
import type { ReactNode } from 'react';

type ApiParameter = {
  name: string;
  annotation: string | null;
  default: string | null;
  kind: string;
  description: string | null;
};

type ApiField = Omit<ApiParameter, 'kind'>;

type LocalizedText = { en: string; zh: string };

type ApiMethod = {
  name: string;
  kind: string;
  signature: string;
  parameters: ApiParameter[];
  return_annotation: string | null;
  returns_description: string;
  raises: Record<string, string>;
  notes: string;
  warnings: string;
  docstring: string;
  intro?: LocalizedText;
  source_path: string;
  line: number;
};

type ApiSymbol = {
  name: string;
  qualified_name: string;
  public_module: string;
  page: string | null;
  source_path: string;
  line: number;
  kind: string;
  signature: string;
  parameters: ApiParameter[];
  return_annotation: string | null;
  returns_description: string;
  raises: Record<string, string>;
  notes: string;
  warnings: string;
  docstring: string;
  intro?: LocalizedText;
  fields: ApiField[];
  methods: ApiMethod[];
};

type CatalogPage = {
  slug: string;
  kind: string;
  title: LocalizedText;
  symbols: string[];
  symbol_count?: number;
};

type CatalogSection = {
  id: string;
  title: LocalizedText;
  description: LocalizedText;
  pages: CatalogPage[];
};

type ApiReferenceData = {
  repository: string;
  branch: string;
  catalog: CatalogSection[];
  groups: Record<string, string[]>;
  group_titles: Record<string, LocalizedText>;
  name_index: Record<string, string>;
  symbols: Record<string, ApiSymbol>;
};

type Locale = 'en' | 'zh';

const data = apiReference as ApiReferenceData;

const labels = {
  en: {
    attributes: 'Attributes',
    count: 'public symbols',
    classmethod: 'class method',
    defaultValue: 'default',
    details: 'Source docstring',
    method: 'method',
    methods: 'Methods',
    onThisPage: 'On this page',
    overview: 'Overview',
    parameters: 'Parameters',
    raises: 'Raises',
    notes: 'Notes',
    property: 'property',
    returns: 'Returns',
    source: 'source',
    staticmethod: 'static method',
    symbols: 'Symbols',
    warnings: 'Warnings',
  },
  zh: {
    attributes: '属性',
    count: '个公开符号',
    classmethod: '类方法',
    defaultValue: '默认值',
    details: '源码 docstring',
    method: '方法',
    methods: '方法',
    overview: '简介',
    parameters: '参数',
    raises: '异常',
    notes: '说明',
    property: '属性',
    returns: '返回值',
    source: '源码',
    staticmethod: '静态方法',
    symbols: '符号',
    warnings: '警告',
  },
} as const;

function sourceUrl(sourcePath: string, line: number) {
  return `${data.repository}/blob/${data.branch}/${sourcePath}#L${line}`;
}

const KIND_ABBR: Record<string, string> = {
  class: 'cls',
  function: 'func',
  protocol: 'prot',
  method: 'meth',
  property: 'prop',
  classmethod: 'cmeth',
  staticmethod: 'smeth',
};

function docsPrefix(locale: Locale) {
  return locale === 'zh' ? '/zh/docs' : '/docs';
}

function symbolHref(qualified: string, locale: Locale) {
  const entry = data.symbols[qualified];
  if (!entry?.page) return null;
  const path = `${docsPrefix(locale)}/api-reference/${entry.page}#${symbolAnchor(qualified)}`;
  return withBasePath(path) ?? path;
}

function resolveCodeTarget(text: string): string | null {
  const bare = text.replace(/[()[\]]+$/g, '');
  if (data.symbols[bare]) return bare;
  if (data.name_index[bare]) return data.name_index[bare];
  const short = bare.includes('.') ? bare.split('.').pop()! : bare;
  if (data.name_index[short]) return data.name_index[short];
  return null;
}

function LinkedCode({
  children,
  locale,
  className,
}: {
  children: string;
  locale: Locale;
  className?: string;
}) {
  const target = resolveCodeTarget(children);
  const href = target ? symbolHref(target, locale) : null;
  if (href) {
    return (
      <a className={className ? `${className} wf-api-xref` : 'wf-api-xref'} href={href}>
        <code>{children}</code>
      </a>
    );
  }
  return <code className={className}>{children}</code>;
}

function renderInline(text: string, locale: Locale, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let part = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) {
      nodes.push(text.slice(last, match.index));
    }
    const token = match[0];
    if (token.startsWith('**')) {
      nodes.push(<strong key={`${keyPrefix}-b-${part}`}>{token.slice(2, -2)}</strong>);
    } else {
      nodes.push(
        <LinkedCode key={`${keyPrefix}-c-${part}`} locale={locale}>
          {token.slice(1, -1)}
        </LinkedCode>,
      );
    }
    part += 1;
    last = match.index + token.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function Description({ value, locale }: { value: string; locale: Locale }) {
  if (!value) return null;
  const blocks = value.split(/\n\s*\n/).filter(Boolean);
  return (
    <div className="wf-api-docstring">
      {blocks.map((block, blockIndex) => {
        const lines = block.split('\n').map((line) => line.trimEnd());
        const isList = lines.every((line) => /^[-*]\s+/.test(line.trim()) || line.trim() === '');
        if (isList) {
          return (
            <ul key={`list-${blockIndex}`}>
              {lines
                .map((line) => line.trim())
                .filter((line) => /^[-*]\s+/.test(line))
                .map((line, itemIndex) => (
                  <li key={`li-${blockIndex}-${itemIndex}`}>
                    {renderInline(line.replace(/^[-*]\s+/, ''), locale, `l${blockIndex}-${itemIndex}`)}
                  </li>
                ))}
            </ul>
          );
        }
        const paragraph = lines.map((line) => line.trim()).filter(Boolean).join(' ');
        return <p key={`p-${blockIndex}`}>{renderInline(paragraph, locale, `p${blockIndex}`)}</p>;
      })}
    </div>
  );
}

function pickIntro(intro: LocalizedText | undefined, locale: Locale) {
  if (!intro) return '';
  return intro[locale] || intro.en || '';
}

function IntroBlock({
  intro,
  docstring,
  locale,
}: {
  intro?: LocalizedText;
  docstring: string;
  locale: Locale;
}) {
  const t = labels[locale];
  const text = pickIntro(intro, locale);
  if (!text && !docstring) return null;

  const normalizedIntro = text.replace(/\s+/g, ' ').trim();
  const normalizedDoc = docstring.replace(/\s+/g, ' ').trim();
  const showDetails =
    Boolean(docstring) &&
    normalizedDoc.length > 0 &&
    normalizedDoc !== normalizedIntro &&
    !normalizedIntro.startsWith(normalizedDoc) &&
    docstring.includes('\n\n');

  return (
    <div className="wf-api-overview">
      {text ? (
        <div className="wf-api-intro">
          <h4>{t.overview}</h4>
          <Description locale={locale} value={text} />
        </div>
      ) : (
        <Description locale={locale} value={docstring} />
      )}
      {showDetails ? (
        <div className="wf-api-details">
          <h4>{t.details}</h4>
          <Description locale={locale} value={docstring} />
        </div>
      ) : null}
    </div>
  );
}

function KindBadge({ kind }: { kind: string }) {
  const label = KIND_ABBR[kind] ?? kind.slice(0, 4);
  return (
    <span className={`wf-api-kind wf-api-kind-${kind}`} title={kind}>
      {label}
    </span>
  );
}

function AnnotatedType({ value, locale }: { value: string; locale: Locale }) {
  const parts = value.split(/([A-Za-z_][\w.]*)/g);
  return (
    <span className="wf-api-type">
      {parts.map((part, index) => {
        if (!/^[A-Za-z_]/.test(part)) return part;
        const target = resolveCodeTarget(part);
        const href = target ? symbolHref(target, locale) : null;
        if (!href) return part;
        return (
          <a className="wf-api-xref" href={href} key={`${part}-${index}`}>
            {part}
          </a>
        );
      })}
    </span>
  );
}

const SIG_KEYWORDS = new Set(['def', 'class', 'async', 'True', 'False', 'None']);

function paramPrefix(kind: string) {
  if (kind === 'var_positional') return '*';
  if (kind === 'var_keyword') return '**';
  return '';
}

function formatParamText(param: ApiParameter) {
  let text = `${paramPrefix(param.kind)}${param.name}`;
  if (param.annotation) text += `: ${param.annotation}`;
  if (param.default !== null) text += ` = ${param.default}`;
  return text;
}

function parseSignatureLead(signature: string) {
  const match = signature.match(/^(async\s+)?(def|class)\s+([A-Za-z_]\w*)\s*\(/);
  if (match) {
    return {
      keyword: `${match[1] ?? ''}${match[2]}`.trim(),
      name: match[3],
      isCallable: true,
    };
  }
  const methodMatch = signature.match(/^([A-Za-z_]\w*)\s*\(/);
  if (methodMatch) {
    return { keyword: '', name: methodMatch[1], isCallable: true };
  }
  const propertyMatch = signature.match(/^([A-Za-z_]\w*)\s*->\s*(.+)$/);
  if (propertyMatch) {
    return { keyword: '', name: propertyMatch[1], isCallable: false, propertyReturn: propertyMatch[2] };
  }
  return { keyword: '', name: '', isCallable: false };
}

function shouldFormatMultiline(parameters: ApiParameter[], signature: string) {
  if (signature.length > 88) return true;
  if (parameters.length >= 3) return true;
  return parameters.some((param) => formatParamText(param).length > 44);
}

function SignatureType({ value, locale }: { value: string; locale: Locale }) {
  return (
    <span className="wf-api-sig-type">
      <AnnotatedType locale={locale} value={value} />
    </span>
  );
}

function SignatureParam({ param, locale }: { param: ApiParameter; locale: Locale }) {
  const prefix = paramPrefix(param.kind);
  return (
    <>
      {prefix ? <span className="wf-api-sig-op">{prefix}</span> : null}
      <span className="wf-api-sig-param">{param.name}</span>
      {param.annotation ? (
        <>
          <span className="wf-api-sig-op">: </span>
          <SignatureType locale={locale} value={param.annotation} />
        </>
      ) : null}
      {param.default !== null ? (
        <>
          <span className="wf-api-sig-op"> = </span>
          <span className="wf-api-sig-default">{param.default}</span>
        </>
      ) : null}
    </>
  );
}

function HighlightedSignature({ text, locale }: { text: string; locale: Locale }) {
  const pattern =
    /(->|\*\*|::|[(),:[\]|=]|\s+)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\d+(?:\.\d+)?)|([A-Za-z_]\w*(?:\.\w+)*)/g;
  const nodes: ReactNode[] = [];
  let match: RegExpExecArray | null;
  let index = 0;
  while ((match = pattern.exec(text)) !== null) {
    const token = match[0];
    if (!token) continue;
    if (match[1]) {
      nodes.push(
        <span className="wf-api-sig-op" key={`op-${index}`}>
          {token}
        </span>,
      );
    } else if (match[2]) {
      nodes.push(
        <span className="wf-api-sig-default" key={`str-${index}`}>
          {token}
        </span>,
      );
    } else if (match[3]) {
      nodes.push(
        <span className="wf-api-sig-default" key={`num-${index}`}>
          {token}
        </span>,
      );
    } else if (match[4]) {
      const target = resolveCodeTarget(token);
      const href = target ? symbolHref(target, locale) : null;
      let className = 'wf-api-sig-ident';
      if (SIG_KEYWORDS.has(token)) className = 'wf-api-sig-keyword';
      else if (token[0] === token[0].toUpperCase()) className = 'wf-api-sig-type';
      nodes.push(
        href ? (
          <a className={`${className} wf-api-xref`} href={href} key={`id-${index}`}>
            {token}
          </a>
        ) : (
          <span className={className} key={`id-${index}`}>
            {token}
          </span>
        ),
      );
    }
    index += 1;
  }
  return <>{nodes}</>;
}

function SignatureDisplay({
  signature,
  name,
  parameters,
  returnAnnotation,
  locale,
  variant = 'block',
}: {
  signature: string;
  name: string;
  parameters: ApiParameter[];
  returnAnnotation: string | null;
  locale: Locale;
  variant?: 'block' | 'compact';
}) {
  const lead = parseSignatureLead(signature);
  const symbolName = name || lead.name;
  const multiline = shouldFormatMultiline(parameters, signature);
  const propertyReturn =
    'propertyReturn' in lead && typeof lead.propertyReturn === 'string'
      ? lead.propertyReturn
      : null;

  if (propertyReturn && symbolName) {
    return (
      <code className={`wf-api-sig-code wf-api-sig-${variant}`}>
        <span className="wf-api-sig-name">{symbolName}</span>
        <span className="wf-api-sig-op"> -&gt; </span>
        <SignatureType locale={locale} value={propertyReturn} />
      </code>
    );
  }

  if (parameters.length > 0 && multiline && lead.isCallable && symbolName) {
    return (
      <code className={`wf-api-sig-code wf-api-sig-${variant}`}>
        <span className="wf-api-sig-line">
          {lead.keyword ? (
            <>
              <span className="wf-api-sig-keyword">{lead.keyword}</span>{' '}
            </>
          ) : null}
          <span className="wf-api-sig-name">{symbolName}</span>
          <span className="wf-api-sig-op">(</span>
        </span>
        {parameters.map((param, paramIndex) => (
          <span className="wf-api-sig-line wf-api-sig-indent" key={`${param.name}-${paramIndex}`}>
            <SignatureParam locale={locale} param={param} />
            {paramIndex < parameters.length - 1 ? <span className="wf-api-sig-op">,</span> : null}
          </span>
        ))}
        <span className="wf-api-sig-line">
          <span className="wf-api-sig-op">)</span>
          {returnAnnotation ? (
            <>
              <span className="wf-api-sig-op"> -&gt; </span>
              <SignatureType locale={locale} value={returnAnnotation} />
            </>
          ) : null}
        </span>
      </code>
    );
  }

  return (
    <code className={`wf-api-sig-code wf-api-sig-${variant}`}>
      <HighlightedSignature locale={locale} text={signature} />
    </code>
  );
}

function ParameterList({
  fields,
  locale,
  title,
}: {
  fields: ApiField[];
  locale: Locale;
  title: string;
}) {
  if (fields.length === 0) return null;
  const t = labels[locale];
  return (
    <div className="wf-api-parameters">
      <h4>{title}</h4>
      <dl>
        {fields.map((field) => (
          <div className="wf-api-parameter" key={field.name} id={`param-${field.name}`}>
            <dt>
              <code>{field.name}</code>
              {field.annotation ? <AnnotatedType locale={locale} value={field.annotation} /> : null}
            </dt>
            <dd>
              {field.description ? (
                <span>{renderInline(field.description, locale, `param-${field.name}`)}</span>
              ) : null}
              {field.default !== null ? (
                <span className="wf-api-default">
                  {t.defaultValue}: <code>{field.default}</code>
                </span>
              ) : null}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function SupplementaryDocs({
  locale,
  notes,
  raises,
  warnings,
}: {
  locale: Locale;
  notes: string;
  raises: Record<string, string>;
  warnings: string;
}) {
  const t = labels[locale];
  const raised = Object.entries(raises);
  if (!notes && !warnings && raised.length === 0) return null;
  return (
    <div className="wf-api-supplementary">
      {notes ? (
        <div>
          <h4>{t.notes}</h4>
          <Description locale={locale} value={notes} />
        </div>
      ) : null}
      {warnings ? (
        <div className="wf-api-warning">
          <h4>{t.warnings}</h4>
          <Description locale={locale} value={warnings} />
        </div>
      ) : null}
      {raised.length > 0 ? (
        <div>
          <h4>{t.raises}</h4>
          <dl>
            {raised.map(([name, description]) => (
              <div key={name}>
                <dt>
                  <code>{name}</code>
                </dt>
                <dd>{renderInline(description, locale, `raise-${name}`)}</dd>
              </div>
            ))}
          </dl>
        </div>
      ) : null}
    </div>
  );
}

function MethodReference({
  method,
  locale,
  symbol,
}: {
  method: ApiMethod;
  locale: Locale;
  symbol: string;
}) {
  const t = labels[locale];
  const methodKind = method.kind || 'method';
  return (
    <div className="wf-api-method" id={methodAnchor(symbol, method.name)}>
      <div className="wf-api-method-heading">
        <KindBadge kind={methodKind} />
        <SignatureDisplay
          locale={locale}
          name={method.name}
          parameters={method.parameters}
          returnAnnotation={method.return_annotation}
          signature={method.signature}
          variant="compact"
        />
        <a href={sourceUrl(method.source_path, method.line)} rel="noreferrer" target="_blank">
          {t.source}
        </a>
      </div>
      <IntroBlock intro={method.intro} docstring={method.docstring} locale={locale} />
      <ParameterList fields={method.parameters} locale={locale} title={t.parameters} />
      <SupplementaryDocs
        locale={locale}
        notes={method.notes}
        raises={method.raises}
        warnings={method.warnings}
      />
      {method.return_annotation ? (
        <p className="wf-api-return">
          <strong>{t.returns}:</strong>{' '}
          <AnnotatedType locale={locale} value={method.return_annotation} />
          {method.returns_description ? (
            <> — {renderInline(method.returns_description, locale, `ret-${method.name}`)}</>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}

export function PythonApiReference({
  symbol,
  locale = 'en',
  heading = false,
}: {
  symbol: string;
  locale?: Locale;
  heading?: boolean;
}) {
  const entry = data.symbols[symbol];
  if (!entry) {
    throw new Error(`Unknown generated Python API symbol: ${symbol}`);
  }
  const t = labels[locale];
  const anchor = symbolAnchor(symbol);
  const signatureParams =
    entry.parameters.length > 0
      ? entry.parameters
      : entry.fields.map((field) => ({ ...field, kind: 'positional_or_keyword' }));

  return (
    <div className="wf-api-reference not-prose" id={heading ? undefined : anchor}>
      {heading ? (
        <div className="wf-api-symbol-heading">
          <h3>
            <a href={`#${anchor}`}>
              <code>{entry.name}</code>
            </a>
          </h3>
          <KindBadge kind={entry.kind} />
        </div>
      ) : null}

      <div className="wf-api-signature">
        <pre>
          <SignatureDisplay
            locale={locale}
            name={entry.name}
            parameters={signatureParams}
            returnAnnotation={entry.return_annotation}
            signature={entry.signature}
          />
        </pre>
        <div className="wf-api-source-row">
          <div>
            {!heading ? <KindBadge kind={entry.kind} /> : null}
            <code>{entry.qualified_name}</code>
            <code>
              from {entry.public_module} import {entry.name}
            </code>
          </div>
          <a href={sourceUrl(entry.source_path, entry.line)} rel="noreferrer" target="_blank">
            {t.source}
          </a>
        </div>
      </div>

      <IntroBlock intro={entry.intro} docstring={entry.docstring} locale={locale} />
      <ParameterList
        fields={entry.kind === 'function' || entry.fields.length === 0 ? entry.parameters : entry.fields}
        locale={locale}
        title={entry.kind === 'function' || entry.fields.length === 0 ? t.parameters : t.attributes}
      />
      <SupplementaryDocs
        locale={locale}
        notes={entry.notes}
        raises={entry.raises}
        warnings={entry.warnings}
      />
      {entry.kind === 'function' && entry.return_annotation ? (
        <p className="wf-api-return">
          <strong>{t.returns}:</strong>{' '}
          <AnnotatedType locale={locale} value={entry.return_annotation} />
          {entry.returns_description ? (
            <> — {renderInline(entry.returns_description, locale, `ret-${entry.name}`)}</>
          ) : null}
        </p>
      ) : null}

      {entry.methods.length > 0 ? (
        <div className="wf-api-methods">
          <h4>{t.methods}</h4>
          {entry.methods.map((method) => (
            <MethodReference locale={locale} method={method} symbol={symbol} key={method.name} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function PythonApiGroupReference({
  group,
  locale = 'en',
}: {
  group: string;
  locale?: Locale;
}) {
  const symbols = data.groups[group];
  if (!symbols) {
    throw new Error(`Unknown generated Python API group: ${group}`);
  }
  const t = labels[locale];

  return (
    <div className="wf-api-group not-prose">
      <p className="wf-api-group-count">
        {symbols.length} {t.count}
      </p>
      {symbols.map((symbol) => {
        const anchor = symbolAnchor(symbol);
        return (
          <section className="wf-api-symbol" id={anchor} key={symbol}>
            <PythonApiReference heading locale={locale} symbol={symbol} />
          </section>
        );
      })}
    </div>
  );
}

export function PythonApiCatalog({ locale = 'en' }: { locale?: Locale }) {
  const t = labels[locale];
  return (
    <div className="wf-api-catalog not-prose">
      {data.catalog.map((section) => (
        <section className="wf-api-catalog-section" key={section.id}>
          <h2>{section.title[locale]}</h2>
          <p>{section.description[locale]}</p>
          {section.pages.map((page) => {
            const href = withBasePath(`${docsPrefix(locale)}/api-reference/${page.slug}`) ??
              `${docsPrefix(locale)}/api-reference/${page.slug}`;
            return (
              <div className="wf-api-catalog-page" key={page.slug}>
                <h3>
                  <a href={href}>{page.title[locale]}</a>
                  {page.kind === 'reference' ? (
                    <span className="wf-api-catalog-count">
                      {page.symbol_count ?? page.symbols.length} {t.symbols}
                    </span>
                  ) : null}
                </h3>
                {page.symbols.length > 0 ? (
                  <ul className="wf-api-catalog-symbols">
                    {page.symbols.map((symbol) => {
                      const entry = data.symbols[symbol];
                      const symbolLink =
                        withBasePath(
                          `${docsPrefix(locale)}/api-reference/${page.slug}#${symbolAnchor(symbol)}`,
                        ) ?? `${docsPrefix(locale)}/api-reference/${page.slug}#${symbolAnchor(symbol)}`;
                      return (
                        <li key={symbol}>
                          <a href={symbolLink}>
                            <code>
                              {entry.public_module}.{entry.name}
                            </code>
                          </a>
                        </li>
                      );
                    })}
                  </ul>
                ) : null}
              </div>
            );
          })}
        </section>
      ))}
    </div>
  );
}
