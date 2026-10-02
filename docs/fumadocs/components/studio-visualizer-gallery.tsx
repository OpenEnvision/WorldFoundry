import { withBasePath } from '@/lib/site-path';

type StudioGalleryLocale = 'en' | 'zh';

type StudioVisualizerDemo = {
  id: string;
  title: Record<StudioGalleryLocale, string>;
  subtitle: Record<StudioGalleryLocale, string>;
  aliases: Record<StudioGalleryLocale, string>;
  demoUrl?: string;
  port: string;
  artifacts: string;
  command: string;
};

const labels = {
  en: {
    eyebrow: 'Studio visualizer demos',
    overviewCaption: 'WorldFoundry Workspace · Visualizers tab',
    overviewAlt: 'WorldFoundry Workspace visualizers overview',
    port: 'Port',
    aliases: 'Aliases',
    artifacts: 'Artifacts',
    launch: 'Launch',
    demo: 'Live demo',
  },
  zh: {
    eyebrow: 'Studio 可视化 Demo',
    overviewCaption: 'WorldFoundry Workspace · Visualizers 页面',
    overviewAlt: 'WorldFoundry Workspace 可视化入口总览',
    port: '端口',
    aliases: 'Alias',
    artifacts: '产物',
    launch: '启动',
    demo: '实时演示',
  },
} satisfies Record<StudioGalleryLocale, Record<string, string>>;

const demos = [
  {
    id: 'world',
    title: {
      en: 'World Rollout Viewer',
      zh: '世界模型分段 Rollout',
    },
    subtitle: {
      en: 'Buffered realtime target for action-conditioned roaming. Controls are sampled continuously while generated chunks are prefetched.',
      zh: '面向实时漫游的缓冲式动作条件 rollout。前端持续采样控制输入并预取生成片段。',
    },
    aliases: {
      en: 'interactive-world, world-model, world-rollout',
      zh: 'interactive-world, world-model, world-rollout',
    },
    port: '7868',
    artifacts: 'image, video, action_trace',
    command: 'python -m worldfoundry.studio.cli matrix-game-2 --frontend world --port 7868',
  },
  {
    id: 'viser',
    title: {
      en: 'Viser Geometry Viewer',
      zh: 'Viser Geometry Viewer',
    },
    subtitle: {
      en: 'Point clouds, depth-derived point sets, meshes, camera poses, and trajectories.',
      zh: '点云、depth 转 point set、mesh、camera pose 和 trajectory。',
    },
    aliases: {
      en: 'viser, geometry, pointcloud',
      zh: 'viser, geometry, pointcloud',
    },
    port: '18590',
    artifacts: 'ply, pcd, xyz, glb, gltf, obj, npz',
    command: 'python -m worldfoundry.studio.cli pi3 --frontend points --asset /path/to/scene.ply',
  },
  {
    id: 'rerun',
    title: {
      en: 'Rerun Timeline Viewer',
      zh: 'Rerun Timeline Viewer',
    },
    subtitle: {
      en: '`.rrd` timelines and recordings with synchronized cameras, frames, tracks, and 3D entities.',
      zh: '`.rrd` timeline / recording，同步 camera、frame、track 与 3D entity。',
    },
    aliases: {
      en: 'rrd',
      zh: 'rrd',
    },
    port: '9876',
    artifacts: 'rrd',
    command: 'python -m worldfoundry.studio.cli vggt --frontend rerun --asset /path/to/recording.rrd',
  },
  {
    id: 'spark',
    title: {
      en: 'Spark Gaussian Splat Viewer',
      zh: 'Spark Gaussian Splat Viewer',
    },
    subtitle: {
      en: '3D Gaussian splats via the in-tree Spark frontend. Live preview uses the Spark.js Hello World butterfly demo.',
      zh: '通过仓内 Spark frontend 查看 3D Gaussian splat。实时预览来自 Spark.js Hello World butterfly demo。',
    },
    aliases: {
      en: '3dgs, splat',
      zh: '3dgs, splat',
    },
    demoUrl: 'https://sparkjs.dev/examples/#hello-world',
    port: '8765',
    artifacts: 'splat, spz, ksplat, sog, splat-ply',
    command: 'python -m worldfoundry.studio.cli vggt --frontend spark --asset /path/to/scene.splat',
  },
] satisfies StudioVisualizerDemo[];

export function StudioVisualizerGallery({ locale = 'en' }: { locale?: StudioGalleryLocale }) {
  const t = labels[locale];

  return (
    <section className="pi-studio-viz-gallery not-prose" aria-label={t.eyebrow}>
      <figure className="pi-studio-viz-overview">
        <div className="pi-studio-viz-overview-media">
          <img
            src={withBasePath('/images/studio/visualizers/live/workspace-overview.png')}
            alt={t.overviewAlt}
            className="pi-studio-viz-overview-image"
            loading="lazy"
          />
        </div>
        <figcaption className="pi-studio-viz-overview-caption">{t.overviewCaption}</figcaption>
      </figure>

      <ul className="pi-studio-viz-list">
        {demos.map((demo) => (
          <li className="pi-studio-viz-item" key={demo.id}>
            <h3 className="pi-studio-viz-item-title">{demo.title[locale]}</h3>
            <p className="pi-studio-viz-item-desc">{demo.subtitle[locale]}</p>
            <p className="pi-studio-viz-item-facts">
              <span>
                {t.port} {demo.port}
              </span>
              <span>
                {t.artifacts} {demo.artifacts}
              </span>
            </p>
            <p className="pi-studio-viz-item-cmd">
              <span className="pi-studio-viz-card-fact-label">{t.launch}</span>
              <code>{demo.command}</code>
            </p>
            {demo.demoUrl ? (
              <p className="pi-studio-viz-item-demo">
                <a href={demo.demoUrl} rel="noreferrer" target="_blank">
                  {t.demo}
                </a>
              </p>
            ) : null}
          </li>
        ))}
      </ul>
    </section>
  );
}
