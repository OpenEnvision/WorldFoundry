import {
  ArrowRight,
  BookOpen,
  Bot,
  CircleCheck,
  Clock3,
  Cpu,
  Cuboid,
  ExternalLink,
  FileOutput,
  Film,
  GitBranch,
  Globe,
  HardDrive,
  Orbit,
  Package,
  Play,
  ScanLine,
  ShieldCheck,
  TriangleAlert,
  type LucideIcon,
} from 'lucide-react';

type ModelKind = 'embodied' | 'three-d' | 'video' | 'vision' | 'world';
type ModelStatus = 'blocked' | 'integrated' | 'preview' | 'verified';
type FactIcon = 'artifact' | 'hardware' | 'input' | 'runtime' | 'scope' | 'storage';
type LinkIcon = 'docs' | 'paper' | 'project' | 'source' | 'weights';

export type ModelRecipeFact = {
  icon?: FactIcon;
  label: string;
  value: string;
};

export type ModelRecipeLink = {
  href: string;
  icon?: LinkIcon;
  label: string;
};

export type ModelRecipeAction = {
  href: string;
  label: string;
};

export type ModelRecipeHeaderProps = {
  actions?: ModelRecipeAction[];
  evidence?: string;
  facts: ModelRecipeFact[];
  kind?: ModelKind;
  links?: ModelRecipeLink[];
  modelId: string;
  provider: string;
  status: ModelStatus;
  statusLabel: string;
  title: string;
};

const kindIcons: Record<ModelKind, LucideIcon> = {
  embodied: Bot,
  'three-d': Cuboid,
  video: Film,
  vision: ScanLine,
  world: Orbit,
};

const factIcons: Record<FactIcon, LucideIcon> = {
  artifact: FileOutput,
  hardware: HardDrive,
  input: Film,
  runtime: Cpu,
  scope: ScanLine,
  storage: Package,
};

const linkIcons: Record<LinkIcon, LucideIcon> = {
  docs: BookOpen,
  paper: BookOpen,
  project: Globe,
  source: GitBranch,
  weights: Package,
};

const statusIcons: Record<ModelStatus, LucideIcon> = {
  blocked: TriangleAlert,
  integrated: ShieldCheck,
  preview: Clock3,
  verified: CircleCheck,
};

function isExternalHref(href: string) {
  return href.startsWith('https://') || href.startsWith('http://');
}

/**
 * Compact, reusable identity and readiness summary for model recipe pages.
 * Keep setup commands and model-specific caveats in the surrounding MDX.
 */
export function ModelRecipeHeader({
  actions = [],
  evidence,
  facts,
  kind = 'world',
  links = [],
  modelId,
  provider,
  status,
  statusLabel,
  title,
}: ModelRecipeHeaderProps) {
  const KindIcon = kindIcons[kind];
  const StatusIcon = statusIcons[status];

  return (
    <section className="wf-model-recipe" aria-label={`${title} runtime summary`}>
      <div className="wf-model-recipe-head">
        <span className="wf-model-recipe-mark" aria-hidden="true">
          <KindIcon size={23} strokeWidth={1.8} />
        </span>
        <div className="wf-model-recipe-identity">
          <p>{provider}</p>
          <code>{modelId}</code>
        </div>
        <span className={`wf-model-recipe-status wf-model-recipe-status-${status}`}>
          <StatusIcon size={14} strokeWidth={2} aria-hidden="true" />
          {statusLabel}
        </span>
      </div>

      <dl className="wf-model-recipe-facts">
        {facts.map((fact) => {
          const FactIcon = factIcons[fact.icon ?? 'scope'];
          return (
            <div key={`${fact.label}:${fact.value}`}>
              <dt>
                <FactIcon size={15} strokeWidth={1.8} aria-hidden="true" />
                <span>{fact.label}</span>
              </dt>
              <dd>{fact.value}</dd>
            </div>
          );
        })}
      </dl>

      {links.length > 0 || actions.length > 0 ? (
        <div className="wf-model-recipe-footer">
          {links.length > 0 ? (
            <nav className="wf-model-recipe-links" aria-label={`${title} official resources`}>
              {links.map((link) => {
                const LinkIcon = linkIcons[link.icon ?? 'docs'];
                const external = isExternalHref(link.href);
                return (
                  <a
                    href={link.href}
                    rel={external ? 'noreferrer' : undefined}
                    key={`${link.label}:${link.href}`}
                  >
                    <LinkIcon size={14} strokeWidth={1.8} aria-hidden="true" />
                    {link.label}
                    {external ? <ExternalLink size={11} strokeWidth={1.8} aria-hidden="true" /> : null}
                  </a>
                );
              })}
            </nav>
          ) : null}

          {actions.length > 0 ? (
            <nav className="wf-model-recipe-actions" aria-label={`${title} quick actions`}>
              {actions.slice(0, 2).map((action, index) => (
                <a
                  className={index === 0 ? 'wf-model-recipe-action-primary' : undefined}
                  href={action.href}
                  key={`${action.label}:${action.href}`}
                >
                  {index === 0 ? <Play size={13} fill="currentColor" aria-hidden="true" /> : null}
                  {action.label}
                  {index !== 0 ? <ArrowRight size={13} aria-hidden="true" /> : null}
                </a>
              ))}
            </nav>
          ) : null}
        </div>
      ) : null}

      {evidence ? (
        <p className="wf-model-recipe-evidence">
          <ShieldCheck size={14} strokeWidth={1.8} aria-hidden="true" />
          {evidence}
        </p>
      ) : null}
    </section>
  );
}
