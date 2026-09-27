'use client';

import { Check, Copy } from 'lucide-react';
import { useMemo, useState } from 'react';

import type { ModelRecipe, ModelRecipeTask } from '@/lib/model-recipe-types';
import { readerStatusLabel } from '@/lib/model-status-label';

type Locale = 'en' | 'zh';
type CommandTab = 'prepare' | 'install' | 'check' | 'run';

const labels = {
  en: {
    title: 'Build your run',
    variant: 'Variant',
    task: 'Task',
    pipeline: 'Pipeline',
    loading: 'Loading',
    invocation: 'Invocation',
    runner: 'Runner',
    environment: 'Environment',
    device: 'Device',
    taskContract: 'Task contract',
    taskProfile: 'Profile',
    inputs: 'Inputs',
    artifacts: 'Artifacts',
    required: 'Required',
    optional: 'Optional',
    defaultValue: 'Default',
    more: 'Show more',
    less: 'Show less',
    inputSingular: 'input',
    inputPlural: 'inputs',
    artifactSingular: 'artifact',
    artifactPlural: 'artifacts',
    noPipeline: 'No runnable inference pipeline is recorded for this model.',
    noContract: 'No contract is recorded for this variant.',
    noArtifacts: 'No artifact contract is recorded for this task.',
    prepare: 'Prepare',
    install: 'Install',
    check: 'Check assets',
    run: 'Run',
    copy: 'Copy',
    copied: 'Copied',
    command: 'Command',
    defaultVariant: 'Default model ID',
    notRecorded: 'Not recorded',
  },
  zh: {
    title: '构建运行命令',
    variant: '变体',
    task: '任务',
    pipeline: '推理 Pipeline',
    loading: '加载方式',
    invocation: '调用方式',
    runner: 'Runner',
    environment: '环境',
    device: '设备',
    taskContract: '任务契约',
    taskProfile: '配置',
    inputs: '输入',
    artifacts: '输出',
    required: '必填',
    optional: '可选',
    defaultValue: '默认值',
    more: '展开',
    less: '收起',
    inputSingular: '项输入',
    inputPlural: '项输入',
    artifactSingular: '项输出',
    artifactPlural: '项输出',
    noPipeline: '该模型没有记录可运行的推理 Pipeline。',
    noContract: '该 variant 没有记录契约。',
    noArtifacts: '该 task 没有记录 Artifact 契约。',
    prepare: '准备',
    install: '安装',
    check: '检查资产',
    run: '运行',
    copy: '复制',
    copied: '已复制',
    command: '命令',
    defaultVariant: '默认 Model ID',
    notRecorded: '未记录',
  },
} as const;

type Copy = (typeof labels)[Locale];

const GENERIC_COPY = new Set([
  'Catalog task profile used by the model runtime.',
  'Recorded input field from the runtime profile.',
]);

type ContractInput = {
  field?: string;
  detail?: string;
  kind?: string;
  target?: string;
  required?: boolean;
  default?: unknown;
  choices?: string[];
  description?: string;
};

type FormattedDefault = {
  display: 'empty' | 'scalar' | 'text' | 'json';
  text: string;
};

function commandPlaceholder(field: { field: string; required?: boolean; default?: unknown; kind?: string }) {
  if (!field.required || (field.default !== undefined && field.default !== null && field.default !== '')) {
    return null;
  }
  if (field.field === 'language_embedding') {
    return '/path/to/embedding.json';
  }
  if (['prompt', 'instruction', 'text', 'caption'].includes(field.field)) {
    return '"Describe the desired output."';
  }
  if (['input_path', 'image', 'video', 'audio', 'images', 'input'].includes(field.field)) {
    return '/path/to/input';
  }
  if (field.kind === 'json' || field.kind === 'interaction_tokens') {
    return "'{}'";
  }
  return 'VALUE';
}

function shellWord(value: string) {
  return /^[A-Za-z0-9._/-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`;
}

function runCommand(modelId: string, task: ModelRecipeTask) {
  const lines = [
    'worldfoundry-eval run \\',
    `  ${shellWord(modelId)} \\`,
    `  --pipeline.task-profile ${shellWord(task.id)} \\`,
  ];
  for (const field of task.inputs) {
    const placeholder = commandPlaceholder(field);
    if (!placeholder) continue;
    lines.push(`  --pipeline.${field.field.replaceAll('_', '-')} ${placeholder} \\`);
  }
  lines.push('  --json');
  return lines.join('\n');
}

function commandFor(tab: CommandTab, recipe: ModelRecipe, runtimeModelId: string, task: ModelRecipeTask) {
  switch (tab) {
    case 'prepare':
      return `bash scripts/inference/prepare_model_infer.sh ${runtimeModelId}`;
    case 'install':
      return `bash scripts/setup/model_env_install.sh --model ${runtimeModelId}`;
    case 'check':
      return recipe.commands.check;
    case 'run':
      return runCommand(runtimeModelId, task);
  }
}

function isRunnableVariant(value: { pipelineTarget: string | null; status: string }) {
  const status = value.status.toLowerCase().replaceAll('-', '_');
  const unavailable = ['not_recorded', 'planned', 'profile', 'profile_only', 'blocked', 'unavailable', 'missing'];
  return Boolean(value.pipelineTarget) && !unavailable.some((marker) => status.includes(marker));
}

function variantLabel(
  value: { id: string; pipelineTarget: string | null; status: string },
  fallback: string,
  locale: Locale,
) {
  if (isRunnableVariant(value)) return value.id;
  const status = readerStatusLabel(value.status, locale, fallback);
  return `${value.id} — ${status}`;
}

type PipelineTargetParts = {
  modulePath: string | null;
  className: string;
  full: string;
};

function parsePipelineTarget(target: string): PipelineTargetParts {
  const colon = target.lastIndexOf(':');
  if (colon <= 0) {
    return { modulePath: null, className: target, full: target };
  }
  return {
    modulePath: target.slice(0, colon),
    className: target.slice(colon + 1),
    full: target,
  };
}

function PipelineTargetRoute({
  target,
  pipelineLabel,
  emptyLabel,
  copyLabel,
  copiedLabel,
}: {
  target: string | null;
  pipelineLabel: string;
  emptyLabel: string;
  copyLabel: string;
  copiedLabel: string;
}) {
  const [copied, setCopied] = useState(false);
  const parts = target ? parsePipelineTarget(target) : null;

  async function copyTarget() {
    if (!parts) return;
    try {
      await navigator.clipboard.writeText(parts.full);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="wf-command-builder-heading-route">
      <span className="wf-command-builder-heading-route-label">{pipelineLabel}</span>
      {parts ? (
        <div className="wf-command-builder-heading-route-body">
          <div className="wf-command-builder-heading-route-id">
            <code className="wf-command-builder-heading-route-class">{parts.className}</code>
            {parts.modulePath ? (
              <code className="wf-command-builder-heading-route-module" title={parts.modulePath}>
                {parts.modulePath}
              </code>
            ) : null}
          </div>
          <button
            type="button"
            className={`wf-command-builder-heading-route-copy${copied ? ' is-copied' : ''}`}
            aria-label={copied ? copiedLabel : copyLabel}
            title={parts.full}
            onClick={() => {
              void copyTarget();
            }}
          >
            {copied ? <Check aria-hidden="true" size={13} /> : <Copy aria-hidden="true" size={13} />}
          </button>
        </div>
      ) : (
        <span className="wf-command-builder-heading-route-empty">{emptyLabel}</span>
      )}
    </div>
  );
}

function parseDetail(detail?: string) {
  if (!detail) {
    return { required: undefined as boolean | undefined, defaultText: undefined as string | undefined, choices: [] as string[] };
  }
  const parts = detail.split(';').map((part) => part.trim()).filter(Boolean);
  let required: boolean | undefined;
  let defaultText: string | undefined;
  const choices: string[] = [];
  const leftovers: string[] = [];
  for (const part of parts) {
    if (part === 'Required') required = true;
    else if (part === 'Optional') required = false;
    else if (part.startsWith('default=')) defaultText = part.slice('default='.length);
    else if (part.startsWith('choices=')) {
      choices.push(...part.slice('choices='.length).split(',').map((item) => item.trim()).filter(Boolean));
    } else {
      leftovers.push(part);
    }
  }
  if (defaultText === undefined && leftovers.length > 0) {
    defaultText = leftovers.join('; ');
  }
  return { required, defaultText, choices };
}

function formatDefault(value: unknown): FormattedDefault {
  if (value === undefined || value === null || value === '') {
    return { display: 'empty', text: '' };
  }
  if (typeof value === 'boolean') {
    return { display: 'scalar', text: value ? 'true' : 'false' };
  }
  if (typeof value === 'object') {
    return { display: 'json', text: JSON.stringify(value, null, 2) };
  }
  const text = String(value);
  if ((text.startsWith('{') && text.endsWith('}')) || (text.startsWith('[') && text.endsWith(']'))) {
    try {
      const parsed = JSON.parse(text) as unknown;
      if (parsed && typeof parsed === 'object') {
        return { display: 'json', text: JSON.stringify(parsed, null, 2) };
      }
    } catch {
      // Keep the original string when it only looks like JSON.
    }
  }
  const looksLikeProse =
    text.includes(' ') &&
    /[A-Za-z\u4e00-\u9fff]/.test(text) &&
    !text.startsWith('/') &&
    !text.startsWith('${') &&
    !text.startsWith('http');
  if (text.includes('\n') || (text.length > 42 && looksLikeProse)) {
    return { display: 'text', text };
  }
  return { display: 'scalar', text };
}

function normalizeInput(item: ContractInput) {
  const parsed = parseDetail(item.detail);
  const defaultValue = item.default !== undefined ? item.default : parsed.defaultText;
  return {
    field: item.field ?? '',
    kind: item.kind,
    target: item.target,
    required: item.required ?? parsed.required ?? false,
    formatted: formatDefault(defaultValue),
    choices: item.choices?.length ? item.choices : parsed.choices,
    description: visibleCopy(item.description),
  };
}

function visibleCopy(value?: string) {
  const text = value?.trim() ?? '';
  return text && !GENERIC_COPY.has(text) ? text : '';
}

function countPhrase(count: number, singular: string, plural: string) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function ExpandableValue({
  text,
  more,
  less,
  variant,
  label,
}: {
  text: string;
  more: string;
  less: string;
  variant: 'text' | 'json';
  label?: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const long = text.length > 96 || text.split('\n').length > 3;
  return (
    <div className={`wf-command-builder-value is-${variant}${expanded ? ' is-expanded' : ''}`}>
      {label ? <span className="wf-command-builder-field-label">{label}</span> : null}
      {variant === 'json' ? <pre><code>{text}</code></pre> : <p>{text}</p>}
      {long ? (
        <button type="button" onClick={() => setExpanded((open) => !open)}>
          {expanded ? less : more}
        </button>
      ) : null}
    </div>
  );
}

function FieldDefault({
  t,
  formatted,
}: {
  t: Copy;
  formatted: FormattedDefault;
}) {
  if (formatted.display === 'empty') return null;
  if (formatted.display === 'scalar') {
    return (
      <div className="wf-command-builder-field-default">
        <span className="wf-command-builder-field-label">{t.defaultValue}</span>
        <code>{formatted.text}</code>
      </div>
    );
  }
  return (
    <ExpandableValue
      text={formatted.text}
      more={t.more}
      less={t.less}
      variant={formatted.display}
      label={t.defaultValue}
    />
  );
}

function ContractFieldRow({
  t,
  field,
}: {
  t: Copy;
  field: ReturnType<typeof normalizeInput>;
}) {
  return (
    <li className="wf-command-builder-contract-row">
      <div className="wf-command-builder-contract-row-top">
        <code className="wf-command-builder-contract-row-name">{field.field}</code>
        <div className="wf-command-builder-contract-row-badges">
          <span className={`wf-command-builder-pill${field.required ? ' is-required' : ''}`}>
            {field.required ? t.required : t.optional}
          </span>
          {field.kind ? <span className="wf-command-builder-pill">{field.kind}</span> : null}
        </div>
      </div>
      {field.description ? <p className="wf-command-builder-field-note">{field.description}</p> : null}
      <FieldDefault t={t} formatted={field.formatted} />
      {field.choices.length > 0 ? (
        <div className="wf-command-builder-choices" aria-label={t.optional}>
          {field.choices.map((choice) => (
            <span key={choice}>{choice}</span>
          ))}
        </div>
      ) : null}
    </li>
  );
}

function ContractPanel({
  t,
  taskId,
  taskLabel,
  description,
  inputs,
  artifacts,
}: {
  t: Copy;
  taskId: string;
  taskLabel: string;
  description: string;
  inputs: ContractInput[];
  artifacts: Array<{ kind?: string; filename?: string; description?: string }>;
}) {
  const fields = inputs.map(normalizeInput).filter((field) => field.field);
  const outputItems = artifacts.filter((item) => item.kind || item.filename);
  const splitColumns = fields.length > 0 && outputItems.length > 0;
  const denseInputGrid = fields.length >= 4;
  const lead = visibleCopy(description);
  const [open, setOpen] = useState(false);

  return (
    <div className={`wf-command-builder-contracts${open ? ' is-open' : ''}`}>
      <button
        type="button"
        className="wf-command-builder-disclosure"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="wf-command-builder-contracts-title">{t.taskContract}</span>
        <span className="wf-command-builder-contracts-meta">
          <span className="wf-command-builder-profile-chip">
            <span className="wf-command-builder-field-label">{t.taskProfile}</span>
            <span className="wf-command-builder-profile-text">
              {taskLabel && taskLabel !== taskId ? (
                <span className="wf-command-builder-profile-name">{taskLabel}</span>
              ) : null}
              <code>{taskId}</code>
            </span>
          </span>
          <span className="wf-command-builder-contracts-counts">
            {countPhrase(fields.length, t.inputSingular, t.inputPlural)}
            {' · '}
            {countPhrase(outputItems.length, t.artifactSingular, t.artifactPlural)}
          </span>
        </span>
      </button>
      {open ? (
      <div className="wf-command-builder-contracts-body">
        {lead ? <p className="wf-command-builder-contracts-lead">{lead}</p> : null}
        <div className={`wf-command-builder-contracts-grid${splitColumns ? '' : ' is-single'}`}>
          <section className="wf-command-builder-contract">
            <header>
              <h3>{t.inputs}</h3>
              <b>{fields.length}</b>
            </header>
            {fields.length ? (
              <ul className={`wf-command-builder-contract-rows${denseInputGrid ? ' is-dense-grid' : ''}`}>
                {fields.map((field) => (
                  <ContractFieldRow key={field.field} t={t} field={field} />
                ))}
              </ul>
            ) : (
              <p>{t.noContract}</p>
            )}
          </section>
          {outputItems.length > 0 || fields.length === 0 ? (
            <section className="wf-command-builder-contract">
              <header>
                <h3>{t.artifacts}</h3>
                <b>{outputItems.length}</b>
              </header>
              {outputItems.length ? (
                <ul className="wf-command-builder-contract-rows">
                  {outputItems.map((item, index) => (
                    <li
                      className="wf-command-builder-contract-row"
                      key={`${item.kind ?? 'artifact'}:${item.filename ?? ''}:${index}`}
                    >
                      <div className="wf-command-builder-contract-row-top">
                        <code className="wf-command-builder-contract-row-name">{item.filename || '—'}</code>
                        {item.kind ? (
                          <div className="wf-command-builder-contract-row-badges">
                            <span className="wf-command-builder-pill">{item.kind}</span>
                          </div>
                        ) : null}
                      </div>
                      {item.description ? <p className="wf-command-builder-field-note">{item.description}</p> : null}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>{t.noArtifacts}</p>
              )}
            </section>
          ) : null}
        </div>
      </div>
      ) : null}
    </div>
  );
}

export function ModelCommandBuilder({ recipe, locale = 'en' }: { recipe: ModelRecipe; locale?: Locale }) {
  const t = labels[locale];
  const choices = useMemo(
    () =>
      recipe.variants.length > 0
        ? recipe.variants.map((variant) => ({
            ...variant,
            pipelineTarget: variant.pipelineTarget ?? recipe.runtime.pipelineTarget,
          }))
        : [
            {
              id: recipe.id,
              label: recipe.name,
              // A model without explicit variants can still expose several
              // task modes. Keep the task field empty here so the renderer
              // below can show the complete model-level task list instead of
              // silently reducing it to tasks[0].
              task: '',
              runtimeProfile: recipe.runtime.profileId ?? '',
              pipelineBinding: recipe.runtime.bindingId ?? '',
              status: recipe.status.integration,
              pipelineTarget: recipe.runtime.pipelineTarget,
              runner: recipe.runtime.runner ?? recipe.runtime.runnerTarget,
              loadingMethod: recipe.runtime.loadingMethod,
              invocationMode: recipe.runtime.invocationMode,
              environmentName: recipe.runtime.environmentName,
              environmentKind: recipe.runtime.environmentKind,
              python: recipe.runtime.python,
              cudaLabel: recipe.runtime.cudaLabel,
              backendStage: recipe.runtime.backendStage,
              runtimeStatus: recipe.runtime.runtimeStatus,
              inputContract: recipe.inputContract,
              artifacts: recipe.artifacts,
            },
          ],
    [recipe],
  );
  const taskChoices: ModelRecipeTask[] =
    recipe.inferenceTasks.length > 0
      ? recipe.inferenceTasks
      : [
          {
            id: recipe.tasks[0] ?? 'default',
            label: recipe.tasks[0] ?? t.notRecorded,
            description: '',
            source: 'catalog',
            variantIds: [],
            inputs: [],
            artifacts: [],
          },
        ];
  const defaultChoice = choices.find(isRunnableVariant) ?? choices[0];
  const defaultTask = taskChoices.find(
    (task) => task.variantIds.length === 0 || task.variantIds.includes(defaultChoice?.id ?? ''),
  ) ?? taskChoices[0];
  const [selectedId, setSelectedId] = useState(defaultChoice?.id ?? recipe.id);
  const [selectedTaskId, setSelectedTaskId] = useState(defaultTask?.id ?? 'default');
  const [tab, setTab] = useState<CommandTab>(defaultChoice && isRunnableVariant(defaultChoice) ? 'run' : 'check');
  const [copied, setCopied] = useState(false);
  const selected = choices.find((choice) => choice.id === selectedId) ?? choices[0];
  const selectedTask = taskChoices.find((task) => task.id === selectedTaskId) ?? taskChoices[0];
  const selectedPipeline = recipe.variants.length > 0
    ? selected?.pipelineTarget
    : selected?.pipelineTarget ?? recipe.runtime.pipelineTarget;
  const selectedRunner = selected?.runner ?? recipe.runtime.runner ?? recipe.runtime.runnerTarget;
  const selectedEnvironment = selected?.environmentName ?? recipe.runtime.environmentName;
  const selectedDevice = selected?.cudaLabel ?? recipe.runtime.cudaLabel;
  const selectedProfile = selected?.runtimeProfile || recipe.runtime.profileId;
  const selectedInputContract = selectedTask.inputs.length
    ? selectedTask.inputs
    : selected?.inputContract?.length
      ? selected.inputContract
      : recipe.inputContract;
  const selectedArtifacts = selectedTask.artifacts.length
    ? selectedTask.artifacts
    : selected?.artifacts?.length
      ? selected.artifacts
      : recipe.artifacts;
  const hasPipeline = recipe.variants.length > 0
    ? Boolean(selected && isRunnableVariant(selected))
    : Boolean(selectedPipeline);
  const hasProfile = Boolean(selectedProfile);
  const command = commandFor(tab, recipe, selected?.id ?? recipe.id, selectedTask);

  async function copyCommand() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  const tabs: CommandTab[] = [
    ...(hasProfile ? (['prepare', 'install'] as CommandTab[]) : []),
    'check',
    ...(hasPipeline ? (['run'] as CommandTab[]) : []),
  ];
  const runUnavailable = tab === 'run' && !hasPipeline;

  return (
    <section className="wf-command-builder" id="use" aria-labelledby="wf-command-builder-title">
      <header className="wf-command-builder-heading">
        <h2 id="wf-command-builder-title">{t.title}</h2>
        <PipelineTargetRoute
          target={selectedPipeline}
          pipelineLabel={t.pipeline}
          emptyLabel={t.noPipeline}
          copyLabel={t.copy}
          copiedLabel={t.copied}
        />
      </header>

      <div className="wf-command-builder-panel">
        <div className="wf-command-builder-controls">
          <label>
            <span>{t.variant}</span>
            <select
              title={selected ? variantLabel(selected, t.notRecorded, locale) : t.notRecorded}
              value={selectedId}
              onChange={(event) => {
                const nextId = event.target.value;
                const nextChoice = choices.find((choice) => choice.id === nextId);
                setSelectedId(nextId);
                if (selectedTask.variantIds.length > 0 && !selectedTask.variantIds.includes(nextId)) {
                  const compatible = taskChoices.find((task) => task.variantIds.includes(nextId));
                  if (compatible) setSelectedTaskId(compatible.id);
                }
                if (tab === 'run' && (!nextChoice || !isRunnableVariant(nextChoice))) setTab('check');
                setCopied(false);
              }}
            >
              {choices.map((choice) => (
                <option value={choice.id} key={choice.id}>
                  {variantLabel(choice, t.notRecorded, locale)}
                </option>
              ))}
            </select>
          </label>
          <div>
            <span>{t.task}</span>
            {taskChoices.length > 1 ? (
              <select
                aria-label={t.task}
                title={`${selectedTask.label} (${selectedTask.id})`}
                value={selectedTask.id}
                onChange={(event) => {
                  const nextTask = taskChoices.find((task) => task.id === event.target.value) ?? taskChoices[0];
                  setSelectedTaskId(nextTask.id);
                  if (nextTask.variantIds.length > 0 && !nextTask.variantIds.includes(selected?.id ?? '')) {
                    const nextVariantId = nextTask.variantIds[0];
                    const nextChoice = choices.find((choice) => choice.id === nextVariantId);
                    setSelectedId(nextVariantId);
                    if (tab === 'run' && (!nextChoice || !isRunnableVariant(nextChoice))) setTab('prepare');
                  }
                  setCopied(false);
                }}
              >
                {taskChoices.map((task) => (
                  <option value={task.id} key={task.id}>
                    {task.label} ({task.id})
                  </option>
                ))}
              </select>
            ) : (
              <strong>
                {selectedTask.label} <code>{selectedTask.id}</code>
              </strong>
            )}
          </div>
        </div>
        {selectedEnvironment || selectedDevice || selectedRunner || selected?.loadingMethod || recipe.runtime.loadingMethod || selected?.invocationMode || recipe.runtime.invocationMode ? (
        <div className="wf-command-builder-runtime">
          {selectedEnvironment ? (
          <div>
            <span>{t.environment}</span>
            <strong title={selectedEnvironment}>{selectedEnvironment}</strong>
          </div>
          ) : null}
          {selectedDevice ? (
          <div>
            <span>{t.device}</span>
            <strong title={selectedDevice}>{selectedDevice}</strong>
          </div>
          ) : null}
          {selectedRunner ? (
          <div>
            <span>{t.runner}</span>
            <code title={selectedRunner}>{selectedRunner}</code>
          </div>
          ) : null}
          {selected?.loadingMethod || recipe.runtime.loadingMethod ? (
          <div>
            <span>{t.loading}</span>
            <strong>{selected?.loadingMethod ?? recipe.runtime.loadingMethod}</strong>
          </div>
          ) : null}
          {selected?.invocationMode || recipe.runtime.invocationMode ? (
          <div>
            <span>{t.invocation}</span>
            <strong>{selected?.invocationMode ?? recipe.runtime.invocationMode}</strong>
          </div>
          ) : null}
        </div>
        ) : null}
      </div>

      {taskChoices.length > 0 ? (
        <ContractPanel
          t={t}
          taskId={selectedTask.id}
          taskLabel={selectedTask.label}
          description={selectedTask.description}
          inputs={selectedInputContract}
          artifacts={selectedArtifacts}
        />
      ) : null}

      <div className="wf-command-builder-terminal">
        <div className="wf-command-builder-toolbar">
          <div className="wf-command-builder-tabs" role="tablist" aria-label={t.title}>
            {tabs.map((item) => (
              <button
                type="button"
                role="tab"
                aria-selected={tab === item}
                aria-controls="wf-command-builder-panel"
                className={tab === item ? 'is-active' : undefined}
                id={`wf-command-builder-tab-${item}`}
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
          {runUnavailable ? null : (
            <button
              type="button"
              className="wf-command-builder-copy"
              aria-live="polite"
              onClick={() => {
                void copyCommand();
              }}
            >
              {copied ? <Check aria-hidden="true" size={13} /> : <Copy aria-hidden="true" size={13} />}
              <span className="wf-command-builder-copy-label">{copied ? t.copied : t.copy}</span>
            </button>
          )}
        </div>
        <div
          className="wf-command-builder-code"
          id="wf-command-builder-panel"
          role="tabpanel"
          aria-labelledby={`wf-command-builder-tab-${tab}`}
        >
          {runUnavailable ? (
            <p className="wf-command-builder-unavailable">{t.noPipeline}</p>
          ) : (
            <pre key={`${selectedId}:${tab}`}>
              <code>{command}</code>
            </pre>
          )}
        </div>
      </div>
    </section>
  );
}
