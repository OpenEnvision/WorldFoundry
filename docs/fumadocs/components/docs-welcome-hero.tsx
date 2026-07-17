import { DocsWelcomeWorkflow } from '@/components/docs-welcome-workflow';
import type { Locale } from '@/lib/i18n';
import { gitConfig } from '@/lib/shared';
import { WORLDFOUNDRY_GITHUB_REPO } from '@/lib/site-links';

type GithubStats = {
  stars: number;
  forks: number;
  watchers: number;
};

type BadgeKind = 'star' | 'watch' | 'fork';

const copy = {
  en: {
    welcome: 'Welcome to WorldFoundry',
    star: 'Star',
    watch: 'Watch',
    fork: 'Fork',
    highlights: [
      {
        title: 'One workflow',
        items: [
          'Discovery, inference, inspection, and evaluation in one reproducible path',
          'TUI, CLI, Studio, Python, and MCP share the same request and artifact contracts',
          'Inspect manifests, needs, and blockers before allocating GPUs or downloading weights',
          'Durable artifacts support later review, rescoring, and evidence reuse',
          'Python API reference generated from live source signatures and docstrings',
        ],
      },
      {
        title: 'Breadth with clear boundaries',
        items: [
          '240+ cataloged models and benchmarks with explicit readiness signals',
          'Native upstream runtimes across video, 3D/4D, interactive worlds, and embodied stacks',
          'Benchmarks score outputs without taking ownership of model loading',
          'Generation and scoring can run in separate environments when required',
          'Scorecards capture provenance, coverage, blockers, and what each result can support',
          'CLI stays the source of truth for integration state and next actions',
        ],
      },
    ],
  },
  zh: {
    welcome: '欢迎使用 WorldFoundry',
    star: 'Star',
    watch: 'Watch',
    fork: 'Fork',
    highlights: [
      {
        title: '一条工作流',
        items: [
          '模型发现、推理、检查与评测走同一条可复现路径',
          'TUI、CLI、Studio、Python 与 MCP 共享 request 与 artifact 契约',
          '申请 GPU 或下载权重前，可先查看 manifest、needs 与 blocker',
          '持久 artifact 支持后续 review、重新打分与证据复用',
          'Python API 参考由源码签名与 docstring 自动生成',
        ],
      },
      {
        title: '广覆盖、边界清晰',
        items: [
          '240+ 模型与 benchmark 目录，readiness 信号明确可见',
          '保留视频、3D/4D、交互世界与具身栈的原生 runtime',
          'Benchmark 只评测输出，不接管模型加载',
          '生成与打分可按需拆到不同环境执行',
          'Scorecard 记录 provenance、覆盖率、blocker 与结果能支撑何种声明',
          'CLI 是集成状态与 next_action 的权威来源',
        ],
      },
    ],
  },
} as const;

async function fetchGithubStats(): Promise<GithubStats | null> {
  try {
    const response = await fetch(
      `https://api.github.com/repos/${gitConfig.user}/${gitConfig.repo}`,
      { next: { revalidate: 3600 } },
    );
    if (!response.ok) return null;

    const data = (await response.json()) as {
      stargazers_count?: number;
      forks_count?: number;
      subscribers_count?: number;
    };

    return {
      stars: data.stargazers_count ?? 0,
      forks: data.forks_count ?? 0,
      watchers: data.subscribers_count ?? 0,
    };
  } catch {
    return null;
  }
}

function formatCount(value: number | undefined) {
  if (value === undefined) return '—';
  return new Intl.NumberFormat('en-US').format(value);
}

function BadgeIcon({ kind }: { kind: BadgeKind }) {
  if (kind === 'star') {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" className="pi-doc-welcome-badge-icon">
        <path
          fill="currentColor"
          d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.75.75 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.819 6.328a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Z"
        />
      </svg>
    );
  }

  if (kind === 'watch') {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" className="pi-doc-welcome-badge-icon">
        <path
          fill="currentColor"
          d="M1.679 7.932c.412-.831 1.027-1.865 1.86-2.81C5.14 3.57 6.86 2.25 8 2.25c1.14 0 2.86 1.32 4.461 2.872 1.833 1.945 2.448 2.978 2.86 3.81.123.246.123.504 0 .75-.412.832-1.027 1.865-2.86 2.81C10.86 12.43 9.14 13.75 8 13.75c-1.14 0-2.86-1.32-4.461-2.872-1.833-1.945-2.448-2.978-2.86-3.81a1.27 1.27 0 0 1 0-.75ZM8 10.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"
        />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="pi-doc-welcome-badge-icon">
      <path
        fill="currentColor"
        d="M5 3.25a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0Zm3.75 0a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM8 9.25a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0Zm-3.75 0a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM2.5 5.5a.75.75 0 0 0 0 1.5h10.5a.75.75 0 0 0 0-1.5H2.5Zm0 3.75a.75.75 0 0 0 0 1.5h10.5a.75.75 0 0 0 0-1.5H2.5Z"
      />
    </svg>
  );
}

function GithubBadge({
  href,
  label,
  count,
  kind,
}: {
  href: string;
  label: string;
  count: number | undefined;
  kind: BadgeKind;
}) {
  return (
    <a className="pi-doc-welcome-badge" href={href} rel="noreferrer" target="_blank">
      <span className="pi-doc-welcome-badge-label">
        <BadgeIcon kind={kind} />
        <span>{label}</span>
      </span>
      <span className="pi-doc-welcome-badge-count">{formatCount(count)}</span>
    </a>
  );
}

export async function DocsWelcomeHero({
  locale,
  description,
}: {
  locale: Locale;
  description: string;
}) {
  const labels = copy[locale];
  const stats = await fetchGithubStats();

  return (
    <section className="pi-doc-welcome-hero" aria-labelledby="docs-welcome-title">
      <p className="pi-doc-welcome-kicker">{labels.welcome}</p>

      <div className="pi-doc-welcome-brand">
        <h1 className="pi-brand-display pi-doc-welcome-title" id="docs-welcome-title">
          WorldFoundry
        </h1>
      </div>

      {description ? <p className="pi-doc-welcome-tagline">{description}</p> : null}

      <DocsWelcomeWorkflow locale={locale} />

      <div className="pi-doc-welcome-github">
        <GithubBadge
          count={stats?.stars}
          href={WORLDFOUNDRY_GITHUB_REPO}
          kind="star"
          label={labels.star}
        />
        <GithubBadge
          count={stats?.watchers}
          href={`${WORLDFOUNDRY_GITHUB_REPO}/subscription`}
          kind="watch"
          label={labels.watch}
        />
        <GithubBadge
          count={stats?.forks}
          href={`${WORLDFOUNDRY_GITHUB_REPO}/fork`}
          kind="fork"
          label={labels.fork}
        />
      </div>

      <div className="pi-doc-welcome-highlights">
        <div className="pi-doc-welcome-highlights-grid">
          {labels.highlights.map((group) => (
            <div className="pi-doc-welcome-highlights-group" key={group.title}>
              <p className="pi-doc-welcome-highlights-heading">{group.title}</p>
              <ul className="pi-doc-welcome-highlights-list">
                {group.items.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
