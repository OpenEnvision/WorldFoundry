export type ModelRecipeStatusGroup =
  | 'verified'
  | 'integrated'
  | 'runtime_ported'
  | 'profile'
  | 'planned'
  | 'blocked';

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
};

export type ModelRecipeRuntime = ModelRecipeRuntimeSummary & {
  bindingId: string | null;
  runnerTarget: string | null;
  runner: string | null;
  pipelineTarget: string | null;
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

export type ModelRecipe = Omit<ModelRecipeIndexEntry, 'runtime' | 'checkpoint'> & {
  runtime: ModelRecipeRuntime;
  sources: ModelRecipeSource[];
  checkpoints: ModelRecipeCheckpoint[];
  variants: ModelRecipeVariant[];
  inputContract: Array<{ field: string; detail: string }>;
  artifacts: Array<{ kind: string; filename: string }>;
  notes: string[];
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
