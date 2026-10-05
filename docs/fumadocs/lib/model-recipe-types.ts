export type ModelRecipeStatusGroup =
  | 'verified'
  | 'integrated'
  | 'runtime_ported'
  | 'profile'
  | 'planned'
  | 'blocked';

export const catalogReadinessLabels: Record<ModelRecipeStatusGroup, string> = {
  verified: 'Runner verified',
  integrated: 'Integrated',
  runtime_ported: 'Runtime ported',
  profile: 'Profile only',
  planned: 'Planned',
  blocked: 'Blocked',
};

export function catalogReadinessLabel(group: ModelRecipeStatusGroup): string {
  return catalogReadinessLabels[group];
}

export type ModelRecipeStatus = {
  group: ModelRecipeStatusGroup;
  label: string;
  integration: string;
  runner: string;
  demo: string;
};

export type ModelRecipeCategory = {
  id: string;
  label: string;
  label_zh: string;
  description: string;
  count: number;
};

export type ModelRecipeRuntimeSummary = {
  profileId: string | null;
  environmentName: string | null;
  environmentKind: 'dedicated' | 'unified' | 'unrecorded';
  python: string | null;
  cudaLabel: string | null;
};

export type ModelRecipeIndexEntry = {
  id: string;
  name: string;
  category: string;
  categoryLabel: string;
  categoryLabelZh: string;
  provider: string;
  summary: string;
  aliases: string[];
  tasks: string[];
  status: ModelRecipeStatus;
  runtime: ModelRecipeRuntimeSummary;
  checkpoint: {
    id: string;
    revision?: string;
    license?: string;
    gated?: boolean;
    private?: boolean;
    role?: string;
    status?: string;
  } | null;
  teaser?: string | null;
};

export type ModelRecipeIndexData = {
  total: number;
  categories: ModelRecipeCategory[];
  recipes: ModelRecipeIndexEntry[];
};

export type ModelRecipeSource = {
  kind: 'project' | 'paper' | 'docs' | 'source' | 'weights';
  label: string;
  url: string;
  revision?: string;
};

export type ModelRecipeContractField = { field: string; detail: string };
export type ModelRecipeArtifact = { kind: string; filename: string };

export type ModelRecipeTaskField = {
  field: string;
  option?: string;
  detail: string;
  kind?: string;
  target?: string;
  required?: boolean;
  default?: unknown;
  choices?: string[];
  description?: string;
};

export type ModelRecipeTaskArtifact = {
  kind: string;
  filename: string;
  description?: string;
};

export type ModelRecipeTask = {
  id: string;
  label: string;
  description: string;
  source: 'inference_spec' | 'catalog' | 'cli';
  variantIds: string[];
  inputs: ModelRecipeTaskField[];
  artifacts: ModelRecipeTaskArtifact[];
};

export type ModelRecipeCheckpoint = {
  id: string;
  revision?: string;
  license?: string;
  gated?: boolean;
  private?: boolean;
  role?: string;
  status?: string;
  notes?: string[];
};

export type ModelRecipeVariant = {
  id: string;
  label: string;
  task: string;
  runtimeProfile: string;
  pipelineBinding: string;
  status: string;
  pipelineTarget: string | null;
  runner: string | null;
  loadingMethod: string | null;
  invocationMode: string | null;
  environmentName: string | null;
  environmentKind: 'dedicated' | 'unified' | 'unrecorded';
  python: string | null;
  cudaLabel: string | null;
  backendStage: string | null;
  runtimeStatus: string | null;
  inputContract: ModelRecipeContractField[];
  artifacts: ModelRecipeArtifact[];
};

export type ModelRecipeRuntime = ModelRecipeRuntimeSummary & {
  bindingId: string | null;
  runnerTarget: string | null;
  runner: string | null;
  pipelineTarget: string | null;
  loadingMethod: string | null;
  invocationMode: string | null;
  backendStage: string | null;
  runtimeStatus: string | null;
  environmentId: string | null;
  cudaProfile: string | null;
  driverStatus: string | null;
  condaPackages: string[];
  pipPackages: string[];
  packageVersions: Record<string, string>;
  validationImports: string[];
  notes: string[];
};

/**
 * Benchmark referenced from a model homepage. `source: 'docs'` means the
 * catalog manifest recommends it explicitly via the optional `docs:` block;
 * `source: 'manifest'` means the benchmark name appears in the manifest's
 * recorded evidence notes.
 */
export type ModelRecipeBenchmarkRef = {
  id: string;
  name: string;
  category: string;
  categoryZh: string;
  summary: string;
  summaryZh: string;
  href: string;
  source: 'docs' | 'manifest';
  reason: string;
  reasonZh: string;
};

/**
 * Narrative block that seeds a generated per-model MDX page. Populated from
 * the optional curated `docs:` / `homepage:` block in the catalog manifest
 * when present (`curated: true`), otherwise synthesized deterministically
 * from recorded catalog, runtime, binding, and evidence fields by
 * scripts/generate-model-recipes.py. Hand-written pages (`pageSource:
 * authored`) do not read this block at render time.
 */
/**
 * Formal publishing institution recorded in the catalog manifest.
 * Companies take precedence over universities/labs; GitHub usernames are
 * never used as a publisher identity.
 */
export type ModelRecipePublisher = {
  name: string;
  nameZh: string;
  kind: 'company' | 'university' | 'lab' | null;
};

export type ModelRecipeFigure = {
  src: string;
  poster?: string;
  kind?: 'image' | 'video';
  caption?: string;
  captionZh?: string;
};

export type ModelRecipePaper = {
  title: string;
  year?: number;
  venue?: string;
  arxivId?: string;
  abstract?: string;
};

export type ModelRecipeDocs = {
  curated: boolean;
  publisher: ModelRecipePublisher | null;
  overview: string[];
  overviewZh: string[];
  architecture: string[];
  architectureZh: string[];
  usageNotes: string[];
  usageNotesZh: string[];
  highlights: string[];
  highlightsZh: string[];
  modalities: { inputs: string[]; outputs: string[] };
  useCases: string[];
  useCasesZh: string[];
  hardware: {
    minVramGb: number | null;
    recommended: string | null;
    notes: string[];
  };
  benchmarks: ModelRecipeBenchmarkRef[];
  limitations: string[];
  limitationsZh: string[];
  paper: ModelRecipePaper | null;
  figures: ModelRecipeFigure[];
};

export type ModelRecipe = Omit<ModelRecipeIndexEntry, 'runtime' | 'checkpoint'> & {
  runtime: ModelRecipeRuntime;
  sources: ModelRecipeSource[];
  checkpoints: ModelRecipeCheckpoint[];
  variants: ModelRecipeVariant[];
  inferenceTasks: ModelRecipeTask[];
  inputContract: ModelRecipeContractField[];
  artifacts: ModelRecipeArtifact[];
  notes: string[];
  docs: ModelRecipeDocs;
  commands: {
    prepare: string;
    install: string;
    inspect: string;
    check: string;
    run: string;
  };
  catalogPath: string;
};

export type ModelRecipeData = {
  total: number;
  categories: ModelRecipeCategory[];
  recipes: ModelRecipe[];
};
