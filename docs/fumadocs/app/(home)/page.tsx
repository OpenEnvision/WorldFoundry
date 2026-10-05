import Link from 'next/link';
import {
  ArrowRight,
  ArrowUpRight,
  BookMarked,
  CircleDot,
  ClipboardCheck,
  Layers3,
  MessageCircle,
  MessagesSquare,
  UserPlus,
} from 'lucide-react';
import { CatalogCoverage } from '@/components/catalog-coverage';
import { HomeConfigureSection } from '@/components/home-configure-section';
import { HomeHeroMedia } from '@/components/home-hero-media';
import { type HomeRecipeOption } from '@/components/home-run-configurator';
import { SiteFooter } from '@/components/site-footer';
import { SiteHeader } from '@/components/site-header';
import { WorldModelProgression } from '@/components/world-model-progression';
import { WorldFoundryWorkflow } from '@/components/worldfoundry-system-map';
import {
  OPENENVISION_AWESOME_WORLD_MODELING,
  OPENENVISION_BLOGXIV_SITE,
  WORLDFOUNDRY_GITHUB_ISSUES,
  WORLDFOUNDRY_GITHUB_REPO,
  WORLDFOUNDRY_SLACK_INVITE,
  WORLDFOUNDRY_WECHAT_QR,
} from '@/lib/site-links';
import { modelRecipeIndex } from '@/lib/model-recipe-index';
import { catalogCoverage } from '@/lib/catalog-coverage';
import { catalogReadinessLabel } from '@/lib/model-recipe-types';
import { withBasePath } from '@/lib/site-path';

const pillars = [
  {
    title: 'Cataloged',
    description:
      'Declarative model and benchmark manifests, plus explicit runtime and asset management, make heterogeneous systems discoverable before compute is spent.',
    Icon: BookMarked,
  },
  {
    title: 'Artifact-centric',
    description:
      'Durable request and result contracts decouple generation from scoring. Videos, geometry, actions, and traces can be inspected or rescored without reloading the model.',
    Icon: Layers3,
  },
  {
    title: 'Evidence-aware',
    description:
      'Scorecards keep declared support, operational readiness, and validated performance separate, so a catalog entry or a demo is never mistaken for a benchmark claim.',
    Icon: ClipboardCheck,
  },
];

const capabilities = [
  {
    index: '01',
    title: 'Know what exists',
    description:
      'Manifests expose stable IDs, sources, capabilities, assets, runtime bindings, readiness, and blockers before compute is allocated.',
    href: '/docs/overview/design',
    link: 'See what is included',
  },
  {
    index: '02',
    title: 'Run through shared boundaries',
    description:
      'Pipelines preserve model-native I/O while TUI, CLI, Studio, Python, and MCP reuse the same execution contracts.',
    href: '/docs/overview/design',
    link: 'Understand the design',
  },
  {
    index: '03',
    title: 'Inspect before scoring',
    description:
      'Videos, geometry, actions, trajectories, and traces stay visible on disk and in Studio instead of disappearing inside scripts.',
    href: '/docs/guides/studio',
    link: 'Explore Studio',
  },
  {
    index: '04',
    title: 'Evidence, not headlines',
    description:
      'Benchmark runners preserve per-sample outcomes, coverage, provenance, blockers, reports, and scorecards.',
    href: '/docs/evaluation',
    link: 'Read about evaluation',
  },
];

const featuredModelIds = [
  'wan2.2',
  'ltx-video',
  'cosmos-predict-2.5',
  'hunyuanvideo-1.5',
  'longcat-video',
  'lingbot-video',
  'bernini',
  'matrix-game-2',
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
    status: catalogReadinessLabel(recipe.status.group),
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
        <SiteHeader
          variant="hero"
          active="home"
          wordmarkClassName="wf-home-wordmark"
          languageLinks={[
            { href: '/', label: 'English', current: true },
            { href: '/zh/docs', label: '中文' },
          ]}
        />

        <section className="wf-home-hero" aria-labelledby="wf-home-title">
          <div className="wf-home-hero-showcase">
            <HomeHeroMedia>
              <div className="wf-home-hero-content">
                <p className="wf-home-hero-kicker">Toward a general infrastructure for world intelligence</p>
                <h1 id="wf-home-title">WorldFoundry</h1>
                <p className="wf-home-hero-lead">
                  Discover, run, inspect, and evaluate world models under one operational
                  definition — without flattening them into a single interface.
                </p>
                <div className="wf-home-hero-actions">
                  <Link href="/docs/quickstart" className="wf-home-button wf-home-button-primary">
                    <span>Get started</span>
                    <ArrowRight aria-hidden="true" size={16} strokeWidth={1.8} />
                  </Link>
                  <Link href="/docs/guides/supported-models" className="wf-home-button wf-home-button-secondary">
                    <span>Explore model recipes</span>
                  </Link>
                </div>
                <Link className="wf-home-hero-catalog-link" href="/docs/guides/supported-models">
                  {catalogCoverage.modelsTotal} model entries · {catalogCoverage.benchmarksTotal} browsable benchmarks
                  <ArrowRight aria-hidden="true" size={13} strokeWidth={1.7} />
                </Link>
              </div>
            </HomeHeroMedia>
          </div>
        </section>
      </div>

      <div className="wf-home-main">
        <p className="wf-home-bridge">
          World models stay heterogeneous. Shared catalogs, artifacts, and scorecards are what make
          them comparable.
        </p>
        <HomeConfigureSection models={featuredModels} />

        <section className="wf-home-pillars wf-home-reveal" aria-labelledby="wf-pillars-title">
          <header className="wf-home-center-intro">
            <p className="wf-home-section-badge">
              <span aria-hidden="true" />
              Why WorldFoundry
            </p>
            <h2 id="wf-pillars-title">
              Commensurability <span>without</span> homogenization.
            </h2>
            <p>
              Shared catalogs, durable artifacts, and evidence-aware scorecards — while native
              inputs, execution semantics, and evaluation protocols stay model-specific.
            </p>
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
            <Link href="/docs" className="wf-home-text-link">
              Read the introduction
              <ArrowRight aria-hidden="true" size={15} strokeWidth={1.8} />
            </Link>
          </div>
        </section>

        <section
          className="wf-home-catalog-section wf-home-reveal"
          aria-labelledby="wf-catalog-title"
        >
          <header className="wf-home-center-intro">
            <p className="wf-home-section-badge">
              <span aria-hidden="true" />
              Catalog
            </p>
            <h2 id="wf-catalog-title">
              {catalogCoverage.modelsTotal} model entries, <span>one</span> operational vocabulary.
            </h2>
            <p>
              Video, 3D/4D, interactive worlds, and embodied systems share manifests and readiness
              signals. Catalog inclusion is not uniform runtime maturity.
            </p>
          </header>
          <CatalogCoverage />
        </section>

        <section
          className="wf-home-taxonomy wf-home-reveal"
          aria-labelledby="wf-taxonomy-title"
        >
          <header className="wf-home-center-intro">
            <p className="wf-home-section-badge">
              <span aria-hidden="true" />
              Capability progression
            </p>
            <h2 id="wf-taxonomy-title">
              From generation to <span>world intelligence</span>.
            </h2>
            <p>
              Visual plausibility is not world modeling. A system exhibits stronger competence as
              representations become persistent, transitions become intervention-responsive, and
              predictions become useful in a closed loop.
            </p>
          </header>
          <WorldModelProgression locale="en" />
          <div className="wf-home-center-action">
            <Link href="/docs#operational-definition" className="wf-home-text-link">
              Read the operational definition
              <ArrowRight aria-hidden="true" size={15} strokeWidth={1.8} />
            </Link>
          </div>
        </section>

        <section
          className="wf-home-capabilities wf-home-reveal"
          aria-labelledby="wf-capabilities-title"
        >
          <header className="wf-home-center-intro">
            <p className="wf-home-section-badge">
              <span aria-hidden="true" />
              End to end
            </p>
            <h2 id="wf-capabilities-title">
              Operate with <span>shared</span> contracts.
            </h2>
            <p>
              Catalog, runtime, workspace, and evaluation share identities and durable outputs.
            </p>
          </header>
          <div className="wf-home-capability-grid">
            {capabilities.map((capability) => (
              <article className="wf-home-capability-card" key={capability.title}>
                <span className="wf-home-capability-index" aria-hidden="true">
                  {capability.index}
                </span>
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
                  <li
                    key={label}
                    data-wf-stage={index}
                    style={{ '--wf-rail-index': index } as React.CSSProperties}
                  >
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
            <p className="wf-home-section-badge">
              <span aria-hidden="true" />
              Around the project
            </p>
            <h2 id="wf-resources-title">
              Resources &amp; <span>ecosystem</span>
            </h2>
            <p>Notes, events, and sibling projects in the OpenEnvision network.</p>
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
            {ecosystemLinks.map((item) => (
              <a
                className="wf-home-resource-card"
                href={item.href}
                key={item.label}
                target="_blank"
                rel="noreferrer"
              >
                <h3>{item.label}</h3>
                <p>{item.text}</p>
                <span aria-hidden="true">
                  <ArrowUpRight size={15} strokeWidth={1.8} />
                </span>
              </a>
            ))}
          </div>
        </section>
      </div>

      <SiteFooter />
    </main>
  );
}
