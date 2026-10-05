/**
 * Map catalog / manifest status dialect to the four reader-facing labels.
 * Keep in sync with docs/fumadocs/scripts/model_home_status.py.
 */

export type ReaderStatusKey = 'verified' | 'pending' | 'static' | 'planned';
export type StatusLocale = 'en' | 'zh';

const READER_LABELS: Record<StatusLocale, Record<ReaderStatusKey, string>> = {
  en: {
    verified: 'Verified',
    pending: 'Runner parity pending',
    static: 'Static load only',
    planned: 'Planned',
  },
  zh: {
    verified: '已验证',
    pending: 'Runner 一致性待确认',
    static: '仅静态加载',
    planned: '计划中',
  },
};

/** Longest-first so a static-verified token wins over a shorter runtime prefix. */
const TOKEN_KEYS: Array<[string, ReaderStatusKey]> = [
  ['native_checkpoint_structure_validated', 'static'],
  ['native_checkpoint_structure_verified', 'static'],
  ['in_tree_checkpoint_runtime_static_verified', 'static'],
  ['in_tree_checkpoint_runtime_static_validated', 'static'],
  ['static_runtime_verified_checkpoint_assets_staged', 'static'],
  ['static_runtime_verified', 'static'],
  ['official_in_process_runtime_static_checkpoint_validation_complete', 'static'],
  ['official_demo_parity_pending', 'pending'],
  ['profile_resolves_catalog_variant_runner_parity_pending', 'pending'],
  ['native_pipeline_cuda_artifact_pending', 'pending'],
  ['native_structure_and_scheduler_parity_validated_gpu_parity_pending', 'pending'],
  ['catalog_demo_and_runner_parity_recorded_checkpoint_runtime_required', 'verified'],
  ['all_released_checkpoints_gpu_validated', 'verified'],
  ['all_declared_variants_checkpoint_gpu_validated', 'verified'],
  ['in_tree_runtime_selected_checkpoint_gpu_validated', 'verified'],
  ['selected_checkpoint_gpu_validated', 'verified'],
  ['checkpoint_gpu_validated', 'verified'],
  ['checkpoint_backed_verified', 'verified'],
  ['full_default_gpu_parity_verified', 'verified'],
  ['full_demo_verified', 'verified'],
  ['converted_checkpoint_strict_restore_and_gpu_action_probe_validated', 'verified'],
  ['robotwin_and_umi_checkpoints_gpu_validated', 'verified'],
  ['in_tree_checkpoint_runtime_gpu_server_init_verified_rollout_pending', 'pending'],
  ['in_tree_checkpoint_runtime_gpu_forward_server_init_verified', 'pending'],
  ['in_tree_checkpoint_runtime_checkpoint_gpu_probe_verified', 'pending'],
  ['in_tree_vendored_import_verified', 'static'],
  ['in_tree_checkpoint_runtime', 'pending'],
  ['in_tree_in_process_checkpoint_runtime', 'pending'],
  ['in_tree_vendor_ready_checkpoint_pending', 'pending'],
  ['in_tree_inference_ready_checkpoint_pending', 'pending'],
  ['in_tree_import_verification_in_progress', 'planned'],
  ['checkpoint_assets_staged_cpu_schema_validated_gpu_pending', 'pending'],
  ['implemented_checkpoint_pending', 'pending'],
  ['implemented_checkpoint_required', 'pending'],
  ['implemented_pending_gpu_parity', 'pending'],
  ['implemented_pending_native_gpu_parity', 'pending'],
  ['implemented_public_checkpoint_not_staged', 'pending'],
  ['pending_official_sample_parity', 'pending'],
  ['pending_gpu_validation', 'pending'],
  ['pending_visual_qa', 'pending'],
  ['runtime_ported', 'pending'],
  ['checkpoint_backed_runtime_ready', 'pending'],
  ['native_checkpoint_layout_validated', 'static'],
  ['native_checkpoint_validated', 'static'],
  ['metadata_only', 'planned'],
  ['component_verified', 'pending'],
];

const VERIFIED_EXACT = new Set(['verified', 'validated', 'passed', 'checkpoint_verified']);
const STATIC_EXACT = new Set(['static', 'static_load', 'static_only', 'static_runtime']);
const PLANNED_EXACT = new Set([
  'planned',
  'todo',
  'proposed',
  'profile',
  'profile_only',
  'metadata_only',
  'not_started',
  'blocked',
]);
const PENDING_EXACT = new Set([
  'pending',
  'integrated',
  'runtime_ported',
  'ported',
  'implemented',
  'configured',
  'partial',
  'not_recorded',
  'not_applicable',
]);

export function normalizeStatusToken(value: string | null | undefined): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_');
}

export function classifyStatusToken(value: string | null | undefined): ReaderStatusKey {
  const raw = String(value || '').trim();
  if (!raw) return 'pending';
  const token = normalizeStatusToken(raw);
  if (token.startsWith('profile_resolves_')) return 'pending';
  for (const [needle, key] of TOKEN_KEYS) {
    if (token.includes(needle)) return key;
  }
  if (VERIFIED_EXACT.has(token)) return 'verified';
  if (STATIC_EXACT.has(token) || token.startsWith('static_')) return 'static';
  if (PLANNED_EXACT.has(token)) return 'planned';
  if (token.includes('static') && (token.includes('verified') || token.includes('validated') || token.includes('load'))) {
    return 'static';
  }
  if (['gpu_validated', 'parity_recorded', 'runner_verified'].some((part) => token.includes(part))) {
    return 'verified';
  }
  if (PENDING_EXACT.has(token) || token.includes('pending') || token.includes('ported')) {
    return 'pending';
  }
  if (token.includes('verified') || token.includes('validated') || token === 'passed') {
    return 'verified';
  }
  return 'pending';
}

export function readerStatusLabel(
  value: string | null | undefined,
  locale: StatusLocale = 'en',
  empty = locale === 'zh' ? '未记录' : 'Not recorded',
): string {
  if (!value || value === 'not_recorded') return empty;
  return READER_LABELS[locale][classifyStatusToken(value)];
}
