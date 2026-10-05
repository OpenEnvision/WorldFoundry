'use client';

import { twMerge as cn } from 'tailwind-merge';
import {
  Children,
  isValidElement,
  useId,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

function escapeValue(v: string) {
  return v.toLowerCase().replace(/\s+/g, '-');
}

function collectTabPanels(children: ReactNode) {
  return Children.toArray(children).filter(isValidElement);
}

function resolveActiveIndex(
  items: string[] | undefined,
  defaultIndex: number,
  defaultValue?: string,
) {
  if (!items?.length) return 0;
  if (defaultValue) {
    const matched = items.findIndex((item) => escapeValue(item) === defaultValue);
    if (matched >= 0) return matched;
  }
  return Math.min(Math.max(defaultIndex, 0), items.length - 1);
}

export type TabsProps = {
  items?: string[];
  label?: ReactNode;
  defaultIndex?: number;
  defaultValue?: string;
  className?: string;
  children?: ReactNode;
};

export function Tabs({
  className,
  items,
  label,
  defaultIndex = 0,
  defaultValue,
  children,
}: TabsProps) {
  const groupId = useId().replace(/:/g, '');
  const panels = useMemo(() => collectTabPanels(children), [children]);
  const initialIndex = useMemo(
    () => resolveActiveIndex(items, defaultIndex, defaultValue),
    [defaultIndex, defaultValue, items],
  );
  const [selected, setSelected] = useState(initialIndex);

  if (!items?.length) {
    return <div className={className}>{children}</div>;
  }

  return (
    <div
      data-wf-docs-tabs="react"
      data-group={groupId}
      className={cn(
        'wf-docs-tabs flex flex-col overflow-hidden rounded-xl border bg-fd-secondary my-4',
        className,
      )}
    >
      {items.map((item, index) => (
        <input
          key={`input-${index}`}
          type="radio"
          name={`wf-docs-tabs-${groupId}`}
          id={`wf-docs-tabs-${groupId}-${index}`}
          checked={index === selected}
          onChange={() => setSelected(index)}
          className="wf-docs-tabs-input sr-only"
        />
      ))}

      <div
        className="flex gap-3.5 text-fd-secondary-foreground overflow-x-auto px-4 not-prose"
        role="tablist"
      >
        {label ? (
          <span className="text-sm font-medium my-auto me-auto">{label}</span>
        ) : null}
        {items.map((item, index) => (
          <label
            key={`label-${index}`}
            htmlFor={`wf-docs-tabs-${groupId}-${index}`}
            role="tab"
            data-tab-index={index}
            data-state={index === selected ? 'active' : 'inactive'}
            aria-selected={index === selected}
            onClick={() => setSelected(index)}
            className="wf-docs-tabs-trigger inline-flex items-center gap-2 whitespace-nowrap text-fd-muted-foreground border-b border-transparent py-2 text-sm font-medium transition-colors [&_svg]:size-4 hover:text-fd-accent-foreground cursor-pointer"
          >
            {item}
          </label>
        ))}
      </div>

      <div className="wf-docs-tabs-panels">
        {panels.map((panel, index) => (
          <div
            key={items[index] ?? index}
            role="tabpanel"
            data-tab-index={index}
            data-state={index === selected ? 'active' : 'inactive'}
            hidden={index !== selected}
            className="wf-docs-tabs-panel p-4 text-[0.9375rem] bg-fd-background rounded-xl outline-none prose-no-margin [&>figure:only-child]:-m-4 [&>figure:only-child]:border-none"
          >
            {panel}
          </div>
        ))}
      </div>
    </div>
  );
}

export type TabProps = {
  value?: string;
  className?: string;
  children?: ReactNode;
};

export function Tab({ children }: TabProps) {
  return <>{children}</>;
}
