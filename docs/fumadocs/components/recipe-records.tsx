import type { ReactNode } from 'react';

export type RecipeRecordItem = {
  term: ReactNode;
  detail: ReactNode;
};

export function RecipeRecords({ items }: { items: RecipeRecordItem[] }) {
  return (
    <div className="wf-recipe-records">
      <dl>
        {items.map((item, index) => (
          <div key={typeof item.term === 'string' ? item.term : index}>
            <dt>{item.term}</dt>
            <dd>{item.detail}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
