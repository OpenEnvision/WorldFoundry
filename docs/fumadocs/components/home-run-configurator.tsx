'use client';

import Link from 'next/link';
import { Check, Copy, ExternalLink } from 'lucide-react';
import { useMemo, useState } from 'react';

import { ModelIdentityMark } from '@/components/model-identity-mark';

export type HomeRecipeOption = {
  id: string;
  name: string;
  provider: string;
  category: string;
  tasks: string[];
  status: string;
  environment: string | null;
  python: string | null;
  cuda: string | null;
};

type CommandMode = 'prepare' | 'inspect' | 'run';

function commandFor(mode: CommandMode, modelId: string) {
  if (mode === 'prepare') {
    return `bash scripts/inference/prepare_model_infer.sh ${modelId}`;
  }
  if (mode === 'inspect') {
    return `worldfoundry-eval zoo model-show --model-id ${modelId} --include-manifest --json`;
  }
  return [
    'worldfoundry-eval evaluate \\',
    '  --mode model \\',
    `  --model-id ${modelId} \\`,
    '  --model-runner worldfoundry:pipeline \\',
    '  --requests-path tmp/requests.jsonl \\',
    `  --output-dir tmp/model_eval/${modelId} \\`,
    '  --metric artifact_count \\',
    '  --json',
  ].join('\n');
}

export function HomeRunConfigurator({ models }: { models: HomeRecipeOption[] }) {
  const [modelId, setModelId] = useState(models[0]?.id ?? '');
  const [task, setTask] = useState(models[0]?.tasks[0] ?? '');
  const [mode, setMode] = useState<CommandMode>('run');
  const [copied, setCopied] = useState(false);
  const model = models.find((item) => item.id === modelId) ?? models[0];
  const command = useMemo(() => commandFor(mode, model?.id ?? modelId), [mode, model, modelId]);

  async function copyCommand() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  if (!model) return null;

  return (
    <div className="wf-home-configurator">
      <div className="wf-home-configurator-controls">
        <div className="wf-home-configurator-selection" key={model.id}>
          <ModelIdentityMark
            id={model.id}
            name={model.name}
            provider={model.provider}
            category={model.category}
            size="medium"
          />
          <div>
            <span>Selected model</span>
            <strong>{model.name}</strong>
            <small>{model.provider}</small>
          </div>
        </div>
        <label className="wf-home-configurator-field">
          <span>Model</span>
          <select
            value={model.id}
            onChange={(event) => {
              const next = models.find((item) => item.id === event.target.value);
              setModelId(event.target.value);
              setTask(next?.tasks[0] ?? '');
            }}
          >
            {models.map((item) => (
              <option value={item.id} key={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <label className="wf-home-configurator-field">
          <span>Task</span>
          <select value={task} onChange={(event) => setTask(event.target.value)}>
            {(model.tasks.length > 0 ? model.tasks : ['Not recorded']).map((item) => (
              <option value={item} key={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <div className="wf-home-configurator-fact">
          <span>Runtime</span>
          <strong>{model.environment ?? 'Not recorded'}</strong>
          <small>
            {[model.python ? `Python ${model.python}` : null, model.cuda].filter(Boolean).join(' · ') || 'Version not recorded'}
          </small>
        </div>
        <div className="wf-home-configurator-fact">
          <span>Readiness</span>
          <strong>{model.status}</strong>
          <small>Separate from runner evidence</small>
        </div>
      </div>

      <div className="wf-home-configurator-terminal">
        <div className="wf-home-configurator-terminal-head">
          <span>Command preview</span>
          <button
            type="button"
            onClick={copyCommand}
            aria-live="polite"
            data-copied={copied ? 'true' : undefined}
          >
            {copied ? <Check aria-hidden="true" size={14} /> : <Copy aria-hidden="true" size={14} />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>
        <div className="wf-home-configurator-modes" role="tablist" aria-label="Command type">
            {(['prepare', 'inspect', 'run'] as const).map((item) => (
              <button
                type="button"
                role="tab"
                aria-selected={mode === item}
                className={mode === item ? 'is-active' : undefined}
                onClick={() => setMode(item)}
                key={item}
              >
                {item[0].toUpperCase() + item.slice(1)}
              </button>
            ))}
        </div>
        <pre key={`${model.id}:${mode}`}>
          <code>{command}</code>
        </pre>
        <footer>
          <span>{task || 'Task not recorded'}</span>
          <Link href={`/docs/guides/supported-models/${model.id}`}>
            Open full recipe
            <ExternalLink aria-hidden="true" size={12} />
          </Link>
        </footer>
      </div>
    </div>
  );
}
