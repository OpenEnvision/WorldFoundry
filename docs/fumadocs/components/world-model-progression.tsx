import {
  Eye,
  Layers,
  PencilRuler,
  RefreshCcw,
  Waypoints,
} from 'lucide-react';

import type { Locale } from '@/lib/i18n';

type ProgressionLocale = Locale;

const copy = {
  en: {
    aria: 'Capability progression from generative modeling toward world modeling',
    foundationKicker: 'Foundational',
    foundationTitle: 'Perceptual grounding',
    foundationBody:
      'Interpret, reconstruct, or synthesize observations. This is the perceptual basis for state inference — not yet a persistent model of the environment.',
    spineKicker: 'World-model spine',
    spineHint: 'persistent state → dynamics prediction → interactive feedback',
    spine: [
      {
        index: 'I',
        title: 'Persistent state',
        body: 'Infer and maintain structured environmental state across observations, identities, geometry, and partial views.',
      },
      {
        index: 'II',
        title: 'Dynamics',
        body: 'Predict coherent transitions under endogenous processes, actions, and interventions, so earlier events constrain later outcomes.',
      },
      {
        index: 'III',
        title: 'Closed-loop interaction',
        body: 'Act, observe, and update the same state. Feedback revises what the model believes before the next prediction.',
      },
    ],
    orthogonalKicker: 'Orthogonal',
    orthogonalTitle: 'Controllable transformation',
    orthogonalBody:
      'Selectively edit represented content while leaving the rest intact. Manipulation can strengthen any stage, but it is not by itself dynamics or interaction.',
  },
  zh: {
    aria: '从生成建模走向世界建模的能力分层',
    foundationKicker: '基础能力',
    foundationTitle: '感知锚定',
    foundationBody:
      '解释、重建或合成观察。它为状态推断提供感知基础，但本身还不等于对环境的持久表征。',
    spineKicker: '世界模型主轴',
    spineHint: '持久状态 → 动力学预测 → 交互反馈',
    spine: [
      {
        index: 'I',
        title: '持久状态',
        body: '从序列观察中推断并维护结构化环境状态，跟踪身份、几何、空间关系与部分可见条件下的属性。',
      },
      {
        index: 'II',
        title: '动力学',
        body: '按物理与因果规律预测状态转移，使内生过程、动作与干预的后果在后续轨迹中持续约束预测。',
      },
      {
        index: 'III',
        title: '闭环交互',
        body: '在同一环境中循环执行动作、观察与状态更新。反馈修正当前状态，再条件化下一步预测。',
      },
    ],
    orthogonalKicker: '正交能力',
    orthogonalTitle: '可控变换',
    orthogonalBody:
      '按指定控制改写表征中的内容，同时保持其余结构。它可以增强主轴上的任一阶段，但单独成立并不足以证明动力学或闭环交互。',
  },
} as const;

const spineIcons = [Layers, Waypoints, RefreshCcw] as const;

export function WorldModelProgression({
  locale = 'en',
}: {
  locale?: ProgressionLocale;
}) {
  const labels = copy[locale];

  return (
    <div className="wf-progression" aria-label={labels.aria}>
      <article className="wf-progression-foundation">
        <span className="wf-progression-kicker">{labels.foundationKicker}</span>
        <div className="wf-progression-foundation-copy">
          <span className="wf-progression-icon" aria-hidden="true">
            <Eye size={20} strokeWidth={1.7} />
          </span>
          <div>
            <h3>{labels.foundationTitle}</h3>
            <p>{labels.foundationBody}</p>
          </div>
        </div>
      </article>

      <div className="wf-progression-spine">
        <p className="wf-progression-kicker">{labels.spineKicker}</p>
        <p className="wf-progression-spine-hint">{labels.spineHint}</p>
        <ol className="wf-progression-spine-grid">
          {labels.spine.map((stage, index) => {
            const Icon = spineIcons[index];
            return (
              <li className="wf-progression-spine-card" key={stage.title}>
                <span className="wf-progression-spine-index" aria-hidden="true">
                  {stage.index}
                </span>
                <span className="wf-progression-icon" aria-hidden="true">
                  <Icon size={20} strokeWidth={1.7} />
                </span>
                <h3>{stage.title}</h3>
                <p>{stage.body}</p>
              </li>
            );
          })}
        </ol>
      </div>

      <article className="wf-progression-orthogonal">
        <span className="wf-progression-kicker">{labels.orthogonalKicker}</span>
        <div className="wf-progression-foundation-copy">
          <span className="wf-progression-icon" aria-hidden="true">
            <PencilRuler size={20} strokeWidth={1.7} />
          </span>
          <div>
            <h3>{labels.orthogonalTitle}</h3>
            <p>{labels.orthogonalBody}</p>
          </div>
        </div>
      </article>
    </div>
  );
}
