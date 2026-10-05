import type { ReactNode } from 'react';

export type KvCatalogRead = {
  label: ReactNode;
  value: ReactNode;
};

export type KvCatalogRow = {
  name: ReactNode;
  names?: ReactNode[];
  hintLabel?: ReactNode;
  hint?: ReactNode;
  description?: ReactNode;
  reads?: KvCatalogRead[];
};

function looksLikeIdentifier(value: string) {
  return /^[a-z][a-z0-9_./:[\]-]*$/.test(value);
}

function CatalogName({ value }: { value: ReactNode }) {
  if (typeof value === 'string') {
    return looksLikeIdentifier(value) ? <code>{value}</code> : <strong>{value}</strong>;
  }
  return value;
}

export function KvCatalog({
  heading,
  vars = false,
  rows,
}: {
  heading?: ReactNode;
  vars?: boolean;
  rows: KvCatalogRow[];
}) {
  return (
    <div className={['pi-kv-catalog', vars ? 'is-vars' : '', 'not-prose'].filter(Boolean).join(' ')}>
      <section className="pi-kv-group">
        {heading ? <h3>{heading}</h3> : null}
        <ul>
          {rows.map((row, index) => {
            const names = [row.name, ...(row.names ?? [])];
            const key = typeof row.name === 'string' ? row.name : index;
            return (
              <li className="pi-kv-row" key={key}>
                <div className="pi-kv-head">
                  <div className="pi-kv-names">
                    {names.map((name, nameIndex) => (
                      <CatalogName key={`${key}:${nameIndex}`} value={name} />
                    ))}
                  </div>
                  {row.hint != null && row.hint !== '' ? (
                    <span className="pi-kv-default">
                      {row.hintLabel != null && row.hintLabel !== '' ? <span>{row.hintLabel}</span> : null}
                      {row.hint}
                    </span>
                  ) : null}
                </div>
                {row.description != null && row.description !== '' ? <p>{row.description}</p> : null}
                {row.reads?.map((read, readIndex) => (
                  <p className="pi-kv-read" key={`${key}:read:${readIndex}`}>
                    <span>{read.label}</span>
                    {read.value}
                  </p>
                ))}
              </li>
            );
          })}
        </ul>
      </section>
    </div>
  );
}
