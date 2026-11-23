import type { Locale } from '@/lib/i18n';
import { defaultLocale } from '@/lib/i18n';

type LocalizedLabel = Record<Locale, string>;

export type MetricQuickNavItem = {
  id: LocalizedLabel;
  label: LocalizedLabel;
};

export type MetricQuickNavGroup = {
  id: string;
  page: MetricQuickNavPageId;
  label: LocalizedLabel;
  items: MetricQuickNavItem[];
};

export type MetricQuickNavPageId = 'scorers' | 'distribution' | 'perceptual';

export const metricQuickNavLabels: Record<
  Locale,
  { title: string; summary: string; hubTitle: string }
> = {
  en: {
    title: 'Jump to a metric',
    summary: '{count} ids · click to expand',
    hubTitle: 'Metric pages',
  },
  zh: {
    title: '跳转到指标',
    summary: '{count} 个 id · 点击展开',
    hubTitle: '指标子页面',
  },
};

export const metricQuickNavPageMeta: Record<
  MetricQuickNavPageId,
  { slug: string; label: LocalizedLabel; description: LocalizedLabel }
> = {
  scorers: {
    slug: '/evaluation/metrics/scorers',
    label: { en: 'Scorers', zh: 'Scorer 指标' },
    description: {
      en: 'CLIPScore, VQAScore, ITMScore, LAION aesthetic, FaceScore, ArtScore, OpenS2V, and editing/alignment scorers',
      zh: 'CLIPScore、VQAScore、ITMScore、LAION 美学、FaceScore、ArtScore、OpenS2V 与编辑/对齐 scorer',
    },
  },
  distribution: {
    slug: '/evaluation/metrics/distribution',
    label: { en: 'Distribution', zh: '分布指标' },
    description: {
      en: 'FID, IS, KID, CMMD, Clean-FID, diversity/coverage, FVD, FVMD, and JEDi',
      zh: 'FID、IS、KID、CMMD、Clean-FID、多样性/覆盖度、FVD、FVMD 与 JEDi',
    },
  },
  perceptual: {
    slug: '/evaluation/metrics/perceptual',
    label: { en: 'Perceptual', zh: '感知指标' },
    description: {
      en: 'LPIPS, SSIM, MS-SSIM, PSNR, DINO, DreamSim, FSIM, CPBD, IRS, masks, detection, and layout',
      zh: 'LPIPS、SSIM、MS-SSIM、PSNR、DINO、DreamSim、FSIM、CPBD、IRS、mask、检测与布局',
    },
  },
};

/** Anchor ids match fumadocs slug output for `### \`metric\` (↑/↓)` headings. */
export const metricQuickNavGroups: MetricQuickNavGroup[] = [
  {
    id: 'multimodal-scorers',
    page: 'scorers',
    label: { en: 'Multimodal scorers', zh: '多模态 scorer' },
    items: [
      { id: { en: 'clip_score-', zh: 'clip_score-' }, label: { en: 'clip_score', zh: 'clip_score' } },
      { id: { en: 'vqa_score-', zh: 'vqa_score-' }, label: { en: 'vqa_score', zh: 'vqa_score' } },
      { id: { en: 'itm_score-', zh: 'itm_score-' }, label: { en: 'itm_score', zh: 'itm_score' } },
    ],
  },
  {
    id: 'quality-reward',
    page: 'scorers',
    label: { en: 'Quality & reward', zh: '质量与 reward' },
    items: [
      {
        id: { en: 'laion_aesthetic-', zh: 'laion_aesthetic-' },
        label: { en: 'laion_aesthetic', zh: 'laion_aesthetic' },
      },
      { id: { en: 'facescore-', zh: 'facescore-' }, label: { en: 'facescore', zh: 'facescore' } },
      { id: { en: 'artscore-', zh: 'artscore-' }, label: { en: 'artscore', zh: 'artscore' } },
      { id: { en: 'facesim_cur-', zh: 'facesim_cur-' }, label: { en: 'facesim_cur', zh: 'facesim_cur' } },
      {
        id: { en: 'gme_score--nexus_score--natural_score-', zh: 'gme_score--nexus_score--natural_score-' },
        label: { en: 'gme / nexus / natural', zh: 'gme / nexus / natural' },
      },
    ],
  },
  {
    id: 'editing-alignment-scorers',
    page: 'scorers',
    label: { en: 'Editing & alignment', zh: '编辑与语义对齐' },
    items: [
      { id: { en: 'semsr-', zh: 'semsr-' }, label: { en: 'semsr', zh: 'semsr' } },
      {
        id: { en: 'manipulation_direction-', zh: 'manipulation_direction-' },
        label: { en: 'manipulation_direction', zh: 'manipulation_direction' },
      },
      { id: { en: 'vs_similarity-', zh: 'vs_similarity-' }, label: { en: 'vs_similarity', zh: 'vs_similarity' } },
      { id: { en: 'quality_loss-', zh: 'quality_loss-' }, label: { en: 'quality_loss', zh: 'quality_loss' } },
    ],
  },
  {
    id: 'image-distribution',
    page: 'distribution',
    label: { en: 'Image distribution', zh: '图像分布' },
    items: [
      { id: { en: 'fid-', zh: 'fid-' }, label: { en: 'fid', zh: 'fid' } },
      { id: { en: 'scene_fid-', zh: 'scene_fid-' }, label: { en: 'scene_fid', zh: 'scene_fid' } },
      { id: { en: 'inception_score-', zh: 'inception_score-' }, label: { en: 'inception_score', zh: 'inception_score' } },
      { id: { en: 'kid-', zh: 'kid-' }, label: { en: 'kid', zh: 'kid' } },
      { id: { en: 'precision_recall-', zh: 'precision_recall-' }, label: { en: 'precision_recall', zh: 'precision_recall' } },
      {
        id: { en: 'improved_precision_recall-', zh: 'improved_precision_recall-' },
        label: { en: 'improved_precision_recall', zh: 'improved_precision_recall' },
      },
      {
        id: { en: 'fwd--cmmd--clean_fid--mind--trend-', zh: 'fwd--cmmd--clean_fid--mind--trend-' },
        label: { en: 'fwd / cmmd / clean_fid / mind / trend', zh: 'fwd / cmmd / clean_fid / mind / trend' },
      },
      { id: { en: 'ppl-', zh: 'ppl-' }, label: { en: 'ppl', zh: 'ppl' } },
      {
        id: { en: 'vendi_score--rke--rnd-', zh: 'vendi_score--rke--rnd-' },
        label: { en: 'vendi_score / rke / rnd', zh: 'vendi_score / rke / rnd' },
      },
      { id: { en: 'rarity_score-', zh: 'rarity_score-' }, label: { en: 'rarity_score', zh: 'rarity_score' } },
      { id: { en: 'fld-', zh: 'fld-' }, label: { en: 'fld', zh: 'fld' } },
      { id: { en: 'multimodal_mid-', zh: 'multimodal_mid-' }, label: { en: 'multimodal_mid', zh: 'multimodal_mid' } },
      { id: { en: 'fjd-', zh: 'fjd-' }, label: { en: 'fjd', zh: 'fjd' } },
      { id: { en: 'crosslid-', zh: 'crosslid-' }, label: { en: 'crosslid', zh: 'crosslid' } },
      { id: { en: 'cfid-', zh: 'cfid-' }, label: { en: 'cfid', zh: 'cfid' } },
      { id: { en: 'ssd-', zh: 'ssd-' }, label: { en: 'ssd', zh: 'ssd' } },
      {
        id: { en: 'linear_separability-', zh: 'linear_separability-' },
        label: { en: 'linear_separability', zh: 'linear_separability' },
      },
      { id: { en: 'fdd-', zh: 'fdd-' }, label: { en: 'fdd', zh: 'fdd' } },
      { id: { en: 'cis-', zh: 'cis-' }, label: { en: 'cis', zh: 'cis' } },
      { id: { en: 'cas-', zh: 'cas-' }, label: { en: 'cas', zh: 'cas' } },
      {
        id: { en: 'attribute_sad--attribute_pad-', zh: 'attribute_sad--attribute_pad-' },
        label: { en: 'attribute_sad / attribute_pad', zh: 'attribute_sad / attribute_pad' },
      },
    ],
  },
  {
    id: 'video-distribution',
    page: 'distribution',
    label: { en: 'Video distribution', zh: '视频分布' },
    items: [
      { id: { en: 'fvd-', zh: 'fvd-' }, label: { en: 'fvd', zh: 'fvd' } },
      { id: { en: 'fvmd-', zh: 'fvmd-' }, label: { en: 'fvmd', zh: 'fvmd' } },
      { id: { en: 'jedi-', zh: 'jedi-' }, label: { en: 'jedi', zh: 'jedi' } },
    ],
  },
  {
    id: 'perceptual-similarity',
    page: 'perceptual',
    label: { en: 'Pairwise & quality', zh: '成对与图像质量' },
    items: [
      {
        id: { en: 'lpips--ssim--ms_ssim--psnr-mixed', zh: 'lpips--ssim--ms_ssim--psnr-mixed' },
        label: { en: 'lpips / ssim / ms_ssim / psnr', zh: 'lpips / ssim / ms_ssim / psnr' },
      },
      {
        id: { en: 'dino_similarity--dreamsim--fsim--cpbd-mixed', zh: 'dino_similarity--dreamsim--fsim--cpbd-mixed' },
        label: { en: 'dino / dreamsim / fsim / cpbd', zh: 'dino / dreamsim / fsim / cpbd' },
      },
      { id: { en: 'irs-', zh: 'irs-' }, label: { en: 'irs', zh: 'irs' } },
    ],
  },
  {
    id: 'perceptual-spatial',
    page: 'perceptual',
    label: { en: 'Masks, detection & layout', zh: 'Mask、检测与布局' },
    items: [
      { id: { en: 'mask_accuracy-', zh: 'mask_accuracy-' }, label: { en: 'mask_accuracy', zh: 'mask_accuracy' } },
      { id: { en: 'object_detection-', zh: 'object_detection-' }, label: { en: 'object_detection', zh: 'object_detection' } },
      { id: { en: 'lqs-', zh: 'lqs-' }, label: { en: 'lqs', zh: 'lqs' } },
      {
        id: { en: 'object_wise_consistency-', zh: 'object_wise_consistency-' },
        label: { en: 'object_wise_consistency', zh: 'object_wise_consistency' },
      },
    ],
  },
];

export function metricDocsPath(locale: Locale, slug: string) {
  return locale === defaultLocale ? `/docs${slug}` : `/${locale}/docs${slug}`;
}

export function metricQuickNavGroupsForPage(page: MetricQuickNavPageId) {
  return metricQuickNavGroups.filter((group) => group.page === page);
}

export type MetricQuickNavResolution = {
  variant: 'hub' | 'page';
  page?: MetricQuickNavPageId;
};

export function resolveMetricQuickNavFromSlugs(
  slugs: readonly string[],
): MetricQuickNavResolution | null {
  if (slugs[0] !== 'evaluation' || slugs[1] !== 'metrics') {
    return null;
  }
  if (slugs.length === 2 || slugs[2] === 'reference') {
    return { variant: 'hub' };
  }
  if (
    slugs[2] === 'scorers'
    || slugs[2] === 'distribution'
    || slugs[2] === 'perceptual'
  ) {
    return { variant: 'page', page: slugs[2] };
  }
  return null;
}
