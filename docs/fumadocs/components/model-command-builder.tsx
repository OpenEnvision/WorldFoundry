'use client';

import { Check, Copy } from 'lucide-react';
import { useMemo, useState } from 'react';

import type { ModelRecipe } from '@/lib/model-recipe-types';

type Locale = 'en' | 'zh';
type CommandTab = 'prepare' | 'install' | 'inspect' | 'check' | 'run';

const labels = {
  en: {
    title: 'Build your run',
    description: 'Choose a recorded variant, then copy the exact setup or inspection command.',
    variant: 'Variant',
    task: 'Task',
    environment: 'Environment',
    device: 'Device',
    prepare: 'Prepare',
    install: 'Install',
    inspect: 'Inspect',
    check: 'Check assets',
    run: 'Run',
    copy: 'Copy',
    copied: 'Copied',
    defaultVariant: 'Default model ID',
    notRecorded: 'Not recorded',
  },
  zh: {
    title: '构建运行命令',
    description: '选择仓库中已记录的 variant，然后复制对应的准备、检查或运行命令。',
    variant: 'Variant',
    task: '任务',
    environment: '环境',
    device: '设备',
    prepare: '准备',
    install: '安装',
    inspect: '查看 Manifest',
    check: '检查资产',
    run: '运行',
    copy: '复制',
    copied: '已复制',
    defaultVariant: '默认 Model ID',
    notRecorded: '未记录',
  },
} as const;

function runCommand(modelId: string) {
  return [
    'worldfoundry-eval evaluate \\',
    '  --mode model \\',
    `  --model-id ${modelId} \\`,
    '  --model-runner worldfoundry:pipeline \\',
    '  --model-manifest-dir worldfoundry/data/models/catalog \\',
    '  --requests-path tmp/requests.jsonl \\',
    `  --output-dir tmp/model_eval/${modelId} \\`,
    '  --metric artifact_count \\',
    '  --json',
  ].join('\n');
}

function commandFor(tab: CommandTab, recipe: ModelRecipe, runtimeModelId: string) {
  switch (tab) {
    case 'prepare':
      return `bash scripts/inference/prepare_model_infer.sh ${runtimeModelId}`;
    case 'install':
      return `bash scripts/setup/model_env_install.sh --model ${runtimeModelId}`;
    case 'inspect':
      return recipe.commands.inspect;
    case 'check':
      return recipe.commands.check;
    case 'run':
      return runCommand(runtimeModelId);
  }
}

export function ModelCommandBuilder({ recipe, locale = 'en' }: { recipe: ModelRecipe; locale?: Locale }) {
  const t = labels[locale];
  const choices = useMemo(
    () =>
      recipe.variants.length > 0
        ? recipe.variants
        : [
            {
              id: recipe.id,
              label: recipe.name,
              task: recipe.tasks[0] ?? '',
              runtimeProfile: recipe.runtime.profileId ?? '',
              pipelineBinding: recipe.runtime.bindingId ?? '',
              status: recipe.status.integration,
            },
          ],
    [recipe],
  );
  const [selectedId, setSelectedId] = useState(choices[0]?.id ?? recipe.id);
  const [tab, setTab] = useState<CommandTab>('run');
  const [copied, setCopied] = useState(false);
  const selected = choices.find((choice) => choice.id === selectedId) ?? choices[0];
  const command = commandFor(tab, recipe, selected?.id ?? recipe.id);

  async function copyCommand() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  const tabs: CommandTab[] = ['prepare', 'install', 'inspect', 'check', 'run'];

  return (
    <section className="wf-command-builder" aria-labelledby="wf-command-builder-title">
      <div className="wf-command-builder-heading">
        <div>
          <h2 id="wf-command-builder-title">{t.title}</h2>
          <p>{t.description}</p>
        </div>
        <span>{recipe.runtime.runner ?? recipe.runtime.runnerTarget ?? t.notRecorded}</span>
      </div>

      <div className="wf-command-builder-controls">
        <label>
          <span>{t.variant}</span>
          <select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
            {choices.map((choice) => (
              <option value={choice.id} key={choice.id}>
                {choice.id}
              </option>
            ))}
          </select>
        </label>
        <div>
          <span>{t.task}</span>
          <strong>{selected?.task || recipe.tasks[0] || t.notRecorded}</strong>
        </div>
        <div>
          <span>{t.environment}</span>
          <strong>{recipe.runtime.environmentName ?? t.notRecorded}</strong>
        </div>
        <div>
          <span>{t.device}</span>
          <strong>{recipe.runtime.cudaLabel ?? t.notRecorded}</strong>
        </div>
      </div>

      <div className="wf-command-builder-tabs" role="tablist" aria-label={t.title}>
        {tabs.map((item) => (
          <button
            type="button"
            role="tab"
            aria-selected={tab === item}
            className={tab === item ? 'is-active' : undefined}
            key={item}
            onClick={() => {
              setTab(item);
              setCopied(false);
            }}
          >
            {t[item]}
          </button>
        ))}
      </div>

      <div className="wf-command-builder-code">
        <pre key={`${selectedId}:${tab}`}>
          <code>{command}</code>
        </pre>
        <button type="button" onClick={copyCommand} aria-live="polite">
          {copied ? <Check aria-hidden="true" size={14} /> : <Copy aria-hidden="true" size={14} />}
          {copied ? t.copied : t.copy}
        </button>
      </div>
    </section>
  );
}
