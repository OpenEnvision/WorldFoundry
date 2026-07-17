import Link from 'next/link';
import {
  ArrowRight,
  CircleDot,
  Layers3,
  MessageCircle,
  MessagesSquare,
  Puzzle,
  ShieldCheck,
  UserPlus,
} from 'lucide-react';
import { CatalogCoverage } from '@/components/catalog-coverage';
import { HomeHeroMedia } from '@/components/home-hero-media';
import { HomeRunConfigurator, type HomeRecipeOption } from '@/components/home-run-configurator';
import { SiteNav } from '@/components/site-nav';
import { SiteSearchTrigger } from '@/components/site-search-trigger';
import { WorldFoundryWorkflow } from '@/components/worldfoundry-system-map';
import { WorldFoundryWordmarkLink } from '@/components/worldfoundry-wordmark';
import {
  OPENENVISION_AWESOME_WORLD_MODELING,
  OPENENVISION_BLOGXIV_SITE,
  OPENENVISION_GAIA_REPO,
  WORLDFOUNDRY_GITHUB_ISSUES,
  WORLDFOUNDRY_GITHUB_REPO,
  WORLDFOUNDRY_SLACK_INVITE,
  WORLDFOUNDRY_WECHAT_QR,
} from '@/lib/site-links';
import { modelRecipeIndex } from '@/lib/model-recipe-index';
import { withBasePath } from '@/lib/site-path';

const pillars = [
  {
    title: 'Shared',
    description:
      'Environment setup, checkpoints, input shaping, previews, and reporting stay shared while model-specific behavior stays explicit.',
    Icon: Puzzle,
  },
  {
    title: 'Durable',
    description:
      'Expensive outputs survive the process that created them, so they can be inspected, rescored, compared, or audited without rerunning inference.',
    Icon: Layers3,
  },
  {
    title: 'Honest',
    description:
      'Catalog coverage, runnable integration, official validation, and leaderboard eligibility stay separate—not collapsed into one ambiguous “supported” label.',
    Icon: ShieldCheck,
  },
];

const capabilities = [
  {
    title: 'Know what exists',
    description:
      'Manifests expose stable IDs, sources, capabilities, assets, runtime bindings, readiness, and blockers before compute is allocated.',
    href: '/docs/overview/capabilities',
    link: 'See what is included',
  },
  {
    title: 'Run through shared boundaries',
    description:
      'Pipelines preserve model-native I/O while TUI, CLI, Studio, Python, and MCP reuse the same execution contracts.',
    href: '/docs/overview/design',
    link: 'Understand the design',
  },
  {
    title: 'Inspect before scoring',
    description:
      'Videos, geometry, actions, trajectories, and traces stay visible on disk and in Studio instead of disappearing inside scripts.',
    href: '/docs/guides/studio',
    link: 'Explore Studio',
  },
  {
    title: 'Evidence, not headlines',
    description:
      'Benchmark runners preserve per-sample outcomes, coverage, provenance, blockers, reports, and scorecards.',
    href: '/docs/evaluation',
    link: 'Read about evaluation',
  },
];

const featuredModelIds = [
  'hunyuanvideo-1.5',
  'bernini',
  'wan2.2',
  'matrix-game-2',
  'flashworld',
  'openvla',
];

const featuredModels: HomeRecipeOption[] = [
  ...featuredModelIds
    .map((id) => modelRecipeIndex.recipes.find((recipe) => recipe.id === id))
    .filter((recipe): recipe is NonNullable<typeof recipe> => Boolean(recipe)),
  ...modelRecipeIndex.recipes.filter(
    (recipe) =>
      recipe.status.group === 'verified' && !featuredModelIds.includes(recipe.id),
  ),
]
  .slice(0, 8)
  .map((recipe) => ({
    id: recipe.id,
    name: recipe.name,
    provider: recipe.provider,
    category: recipe.category,
    tasks: recipe.tasks,
    status: recipe.status.label,
    environment: recipe.runtime.environmentName,
    python: recipe.runtime.python,
    cuda: recipe.runtime.cudaLabel,
  }));

const communityLinks = [
  {
    title: 'Join Slack',
    description: 'Real-time help & discussions',
    href: WORLDFOUNDRY_SLACK_INVITE,
    Icon: MessagesSquare,
  },
  {
    title: 'GitHub Issues',
    description: 'Bug reports & feature requests',
    href: WORLDFOUNDRY_GITHUB_ISSUES,
    Icon: CircleDot,
  },
  {
    title: 'GitHub',
    description: 'Source, stars, and contributions',
    href: WORLDFOUNDRY_GITHUB_REPO,
    Icon: MessageCircle,
  },
];

const resourceLinks = [
  {
    label: 'Blog',
    text: 'Updates, technical notes, and release highlights.',
    href: '/blog',
  },
  {
    label: 'Events',
    text: 'Meetups, demos, benchmark sprints, and milestones.',
    href: '/events',
  },
  {
    label: 'OpenEnvision',
    text: 'The lab organization behind WorldFoundry.',
    href: '/openenvision',
  },
];

const ecosystemLinks = [
  {
    label: 'Awesome World Modeling',
    text: 'Curated papers, models, and resources for world modeling.',
    href: OPENENVISION_AWESOME_WORLD_MODELING,
    external: true,
  },
  {
    label: 'BlogrXiv',
    text: 'Curated index for technical AI research blogs and writing.',
    href: OPENENVISION_BLOGXIV_SITE,
    external: true,
  },
  {
    label: 'Gaia',
    text: 'Sibling open-vision project in the same organization.',
    href: OPENENVISION_GAIA_REPO,
    external: true,
  },
  {
    label: 'WorldFoundry on GitHub',
    text: 'Source, issues, discussions, and releases.',
    href: WORLDFOUNDRY_GITHUB_REPO,
    external: true,
  },
];

const workflowRail = ['Discover', 'Prepare', 'Run', 'Inspect', 'Evaluate'] as const;

export default function HomePage() {
  return (
    <main className="pi-home-shell wf-home-shell">
      <div className="wf-home-stage">
        <header className="wf-home-site-header">
          <div className="wf-home-site-header-inner">
            <div className="pi-doc-header-brand">
              <WorldFoundryWordmarkLink variant="compact" className="wf-home-wordmark" />
            </div>
            <div className="wf-home-site-header-tools">
              <SiteNav active="home" />
              <SiteSearchTrigger />
              <div className="pi-language-switch" aria-label="Language">
                <Link href="/" aria-current="true">
                  English
                </Link>
                <Link href="/zh/docs">中文</Link>
              </div>
            </div>
          </div>
        </header>

        <section className="wf-home-hero" aria-labelledby="wf-home-title">
          <HomeHeroMedia />
          <div className="wf-home-hero-scrim" aria-hidden="true" />
          <div className="wf-home-hero-grain" aria-hidden="true" />

          <div className="wf-home-hero-content">
            <h1 id="wf-home-title">WorldFoundry</h1>
            <p className="wf-home-hero-lead">
              Run, inspect, and evaluate world models in one reproducible workflow.
            </p>
            <div className="wf-home-hero-actions">
              <Link href="/docs/guides/supported-models" className="wf-home-button wf-home-button-primary">
                <span>Explore model recipes</span>
                <ArrowRight aria-hidden="true" size={16} strokeWidth={1.8} />
              </Link>
              <Link href="/docs/quickstart" className="wf-home-button wf-home-button-secondary">
                <span>Start the quickstart</span>
              </Link>
            </div>
            <Link className="wf-home-hero-catalog-link" href="/docs/guides/supported-models">
              {modelRecipeIndex.total} manifest-backed model recipes
              <ArrowRight aria-hidden="true" size={13} strokeWidth={1.7} />
            </Link>
          </div>

        </section>
      </div>

      <div className="wf-home-main">
        <section className="wf-home-configure wf-home-reveal" aria-labelledby="wf-configure-title">
          <header className="wf-home-configure-heading">
            <div>
              <h2 id="wf-configure-title">Configure a run</h2>
            </div>
            <p>
              Choose a real catalog entry. Runtime and version facts come from the repository
              manifests, and the command stays copyable.
            </p>
          </header>
          <HomeRunConfigurator models={featuredModels} />
        </section>

        <section className="wf-home-pillars wf-home-reveal" aria-labelledby="wf-pillars-title">
          <header className="wf-home-center-intro">
            <h2 id="wf-pillars-title" className="sr-only">
              Why WorldFoundry
            </h2>
            <p className="wf-home-center-lead">Infrastructure for operating world models.</p>
          </header>
          <div className="wf-home-pillar-grid">
            {pillars.map(({ title, description, Icon }) => (
              <article className="wf-home-pillar" key={title}>
                <span className="wf-home-pillar-icon" aria-hidden="true">
                  <Icon size={22} strokeWidth={1.7} />
                </span>
                <h3>{title}</h3>
                <p>{description}</p>
              </article>
            ))}
          </div>
          <div className="wf-home-center-action">
            <Link href="/docs/overview/why-worldfoundry" className="wf-home-text-link">
              Read the benefits and tradeoffs
              <ArrowRight aria-hidden="true" size={15} strokeWidth={1.8} />
            </Link>
          </div>
        </section>

        <section
          className="wf-home-catalog-section wf-home-reveal"
          aria-labelledby="wf-catalog-title"
        >
          <header className="wf-home-center-intro">
            <h2 id="wf-catalog-title">Universal Compatibility</h2>
            <p>One engine, endless possibilities. Browse integrated models and benchmarks.</p>
          </header>
          <CatalogCoverage />
        </section>

        <section
          className="wf-home-capabilities wf-home-reveal"
          aria-labelledby="wf-capabilities-title"
        >
          <header className="wf-home-center-intro">
            <h2 id="wf-capabilities-title">Operate end to end</h2>
            <p>
              Catalog, runtime, workspace, and evaluation share identities and durable contracts.
            </p>
          </header>
          <div className="wf-home-capability-grid">
            {capabilities.map((capability) => (
              <article className="wf-home-capability-card" key={capability.title}>
                <h3>{capability.title}</h3>
                <p>{capability.description}</p>
                <Link href={capability.href}>
                  {capability.link}
                  <ArrowRight aria-hidden="true" size={15} strokeWidth={1.8} />
                </Link>
              </article>
            ))}
          </div>
        </section>

        <section
          className="wf-home-workflow-section wf-home-reveal"
          aria-labelledby="wf-workflow-title"
        >
          <header className="wf-home-workflow-intro">
            <p className="wf-home-workflow-badge">
              <span aria-hidden="true" />
              Five-stage pipeline
            </p>
            <h2 id="wf-workflow-title">
              How it <span>works</span>
            </h2>
            <p className="wf-home-workflow-lead">
              Every surface follows the same sequence. The artifact is the handoff between model
              execution and benchmark evaluation.
            </p>
            <div className="wf-home-workflow-rail" aria-hidden="true">
              <span className="wf-home-workflow-rail-progress" />
              <ol>
                {workflowRail.map((label, index) => (
                  <li key={label} style={{ '--wf-rail-index': index } as React.CSSProperties}>
                    <span />
                    <strong>{label}</strong>
                  </li>
                ))}
              </ol>
            </div>
          </header>
          <div className="wf-home-workflow-frame">
            <div className="wf-home-workflow-ambient" aria-hidden="true">
              <span className="wf-home-workflow-ambient-glow" />
              <span className="wf-home-workflow-ambient-grid" />
            </div>
            <WorldFoundryWorkflow variant="home" />
          </div>
        </section>

        <section
          className="wf-home-community wf-home-reveal"
          aria-labelledby="wf-community-title"
        >
          <div className="wf-home-community-panel">
            <div className="wf-home-community-copy">
              <p className="wf-home-community-badge">
                <span aria-hidden="true" />
                Everyone welcome
              </p>
              <h2 id="wf-community-title">
                Got questions?
                <span> We&apos;re here to help.</span>
              </h2>
              <p>
                Whether you&apos;re just getting started or debugging a complex run, the community is
                open to everyone. No question is too basic.
              </p>
              <ul className="wf-home-community-signals">
                <li>
                  <MessageCircle aria-hidden="true" size={16} strokeWidth={1.8} />
                  Fast &amp; friendly responses
                </li>
                <li>
                  <UserPlus aria-hidden="true" size={16} strokeWidth={1.8} />
                  Active maintainers
                </li>
              </ul>
            </div>

            <div className="wf-home-community-actions">
              <ul className="wf-home-community-links">
                {communityLinks.map(({ title, description, href, Icon }) => (
                  <li key={title}>
                    <a href={href} target="_blank" rel="noreferrer" className="wf-home-community-link">
                      <Icon aria-hidden="true" size={20} strokeWidth={1.7} />
                      <span>
                        <strong>{title}</strong>
                        <em>{description}</em>
                      </span>
                      <ArrowRight aria-hidden="true" size={16} strokeWidth={1.8} />
                    </a>
                  </li>
                ))}
              </ul>

              <figure className="wf-home-community-qr">
                <img
                  src={withBasePath(WORLDFOUNDRY_WECHAT_QR)}
                  alt="WorldFoundry WeChat community QR code"
                  width={280}
                  height={280}
                />
                <figcaption>
                  <strong>WeChat Community</strong>
                  <span>Scan to join. The QR code is updated periodically if it expires.</span>
                </figcaption>
              </figure>
            </div>
          </div>
        </section>

        <section className="wf-home-resources wf-home-reveal" aria-labelledby="wf-resources-title">
          <header className="wf-home-center-intro">
            <h2 id="wf-resources-title">Resources</h2>
            <p>Notes, events, and lab context around the project.</p>
          </header>
          <div className="wf-home-resource-grid">
            {resourceLinks.map((item) => (
              <Link className="wf-home-resource-card" href={item.href} key={item.label}>
                <h3>{item.label}</h3>
                <p>{item.text}</p>
                <span aria-hidden="true">
                  <ArrowRight size={15} strokeWidth={1.8} />
                </span>
              </Link>
            ))}
          </div>
        </section>

        <section className="wf-home-ecosystem wf-home-reveal" aria-labelledby="wf-ecosystem-title">
          <header className="wf-home-center-intro">
            <h2 id="wf-ecosystem-title">Ecosystem</h2>
            <p>Sibling projects and source links in the OpenEnvision ecosystem.</p>
          </header>
          <ul className="wf-home-ecosystem-chips">
            {ecosystemLinks.map((item) => (
              <li key={item.label}>
                <a href={item.href} target="_blank" rel="noreferrer">
                  <strong>{item.label}</strong>
                  <span>{item.text}</span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </main>
  );
}
