'use client';

import { twMerge as cn } from 'tailwind-merge';
import {
  Children,
  isValidElement,
  useId,
  useMemo,
  type ReactNode,
} from 'react';

function escapeValue(v: string) {
  return v.toLowerCase().replace(/\s+/g, '-');
}

function collectTabPanels(children: ReactNode) {
  return Children.toArray(children).filter(isValidElement);
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

  const activeIndex = useMemo(() => {
    if (!items?.length) return 0;
    if (defaultValue) {
      const matched = items.findIndex((item) => escapeValue(item) === defaultValue);
      if (matched >= 0) return matched;
    }
    return Math.min(Math.max(defaultIndex, 0), items.length - 1);
  }, [defaultIndex, defaultValue, items]);

  if (!items?.length) {
    return <div className={className}>{children}</div>;
  }

  return (
    <div
      data-wf-docs-tabs="css"
      data-group={groupId}
      className={cn(
        'wf-docs-tabs flex flex-col overflow-hidden rounded-xl border bg-fd-secondary my-4',
        className,
      )}
    >
      {items.map((item, index) => (
        <input
          key={`input-${item}`}
          type="radio"
          name={`wf-docs-tabs-${groupId}`}
          id={`wf-docs-tabs-${groupId}-${escapeValue(item)}`}
          defaultChecked={index === activeIndex}
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
            key={`label-${item}`}
            htmlFor={`wf-docs-tabs-${groupId}-${escapeValue(item)}`}
            role="tab"
            data-tab-index={index}
            data-state={index === activeIndex ? 'active' : 'inactive'}
            aria-selected={index === activeIndex}
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
            data-state={index === activeIndex ? 'active' : 'inactive'}
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
