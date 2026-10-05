import Image from 'next/image';
import Link from 'next/link';

import { SiteHeader } from '@/components/site-header';
import {
  OPENENVISION_AWESOME_WORLD_MODELING,
  OPENENVISION_BLOGXIV_SITE,
} from '@/lib/site-links';
import { withBasePath } from '@/lib/site-path';
import type { Metadata } from 'next';

const openEnvisionLogoSrc = withBasePath('/openenvision-logo.png') ?? '/openenvision-logo.png';

const projects = [
  {
    name: 'ScholarTube',
    category: 'Research videos',
    description: 'A curated, source-linked video knowledge index for AI researchers.',
    descriptionZh: '面向 AI 研究者的视频知识索引，精选研究视频并保留来源链接。',
    repo: 'https://github.com/OpenEnvision/ScholarTube',
    site: 'https://openenvision.github.io/ScholarTube/',
  },
  {
    name: 'Awesome Multimodal Modeling',
    category: 'Multimodal research',
    description: 'Explore research on multimodal large language models, unified multimodal modeling, and native multimodal modeling.',
    descriptionZh: '多模态研究资源合集，覆盖多模态大语言模型、统一多模态建模与原生多模态建模。',
    repo: 'https://github.com/OpenEnvision/Awesome-Multimodal-Modeling',
    site: 'https://openenvision.github.io/Awesome-Multimodal-Modeling/',
  },
  {
    name: 'BlogrXiv',
    category: 'Research blogs',
    description: 'Discover AI research blogs and technical writing beyond the paper.',
    descriptionZh: '发现 AI 研究博客与技术文章，了解论文之外的研究思路和实践经验。',
    repo: 'https://github.com/OpenEnvision/BlogrXiv',
    site: OPENENVISION_BLOGXIV_SITE,
  },
  {
    name: 'Awesome World Modeling',
    category: 'World models',
    description: 'A curated collection of representational, generative, and agentic world-model research.',
    descriptionZh: '世界模型研究资源合集，覆盖表征式、生成式与智能体世界模型。',
    repo: OPENENVISION_AWESOME_WORLD_MODELING,
    site: 'https://openenvision.github.io/Awesome-World-Modeling/',
  },
];

export const metadata: Metadata = {
  title: 'OpenEnvision',
  description: 'Explore OpenEnvision research tools and open resources: ScholarTube, Awesome Multimodal Modeling, BlogrXiv, Awesome World Modeling, and WorldFoundry.',
};

export default function OpenEnvisionPage() {
  return (
    <main className="pi-home-shell wf-home-shell">
      <SiteHeader
        variant="solid"
        active="openenvision"
        languageLinks={[
          { href: '/openenvision', label: 'English', current: true },
          { href: '/zh/docs', label: '中文' },
        ]}
      />

      <div className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8 md:py-12">
        <section className="pi-open-hero" aria-labelledby="openenvision-title">
          <Image
            src={openEnvisionLogoSrc}
            alt="OpenEnvision logo"
            className="pi-open-logo"
            width={148}
            height={148}
            priority
          />
          <div>
            <p className="pi-label">GitHub organization</p>
            <h1 id="openenvision-title">OpenEnvision</h1>
            <p>
              OpenEnvision Lab is a joint research lab advancing open vision intelligence through
              academia-industry collaboration.
            </p>
            <p>
              OpenEnvision Lab 是一个通过产学协作推进 open vision intelligence 的联合研究实验室。
            </p>
          </div>
        </section>

        <section className="pi-open-section" aria-labelledby="openenvision-projects">
          <h2 id="openenvision-projects">Explore our work</h2>
          <p className="pi-open-project-intro">
            Open tools and shared resources for discovering, understanding, and building AI.
            <span lang="zh-CN"> 从研究视频、技术博客到模型资源，一起探索和构建 AI。</span>
          </p>
          <div className="pi-open-project-grid">
            {projects.map((project) => (
              <article className="pi-open-project-card" key={project.name}>
                <p className="pi-label">{project.category}</p>
                <h3>{project.name}</h3>
                <p>{project.description}</p>
                <p lang="zh-CN">{project.descriptionZh}</p>
                <div className="pi-open-project-links">
                  <a href={project.site} target="_blank" rel="noreferrer"
                    aria-label={`Explore ${project.name}`}>
                    Explore project <span aria-hidden="true">↗</span>
                  </a>
                  <a href={project.repo} target="_blank" rel="noreferrer"
                    aria-label={`${project.name} on GitHub`}>
                    GitHub <span aria-hidden="true">↗</span>
                  </a>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="pi-open-section" aria-labelledby="openenvision-links">
          <h2 id="openenvision-links">Project Links</h2>
          <table className="pi-open-table">
            <tbody>
              <tr>
                <th>Organization</th>
                <td>
                  <a href="https://github.com/OpenEnvision">https://github.com/OpenEnvision</a>
                </td>
              </tr>
              <tr>
                <th>WorldFoundry repo</th>
                <td>
                  <a href="https://github.com/OpenEnvision/WorldFoundry">
                    https://github.com/OpenEnvision/WorldFoundry
                  </a>
                </td>
              </tr>
              <tr>
                <th>Clone URL</th>
                <td>
                  <code>https://github.com/OpenEnvision/WorldFoundry.git</code>
                </td>
              </tr>
            </tbody>
          </table>
        </section>

        <section className="pi-open-section" aria-labelledby="clone-command">
          <h2 id="clone-command">Clone</h2>
          <div className="pi-command" aria-label="WorldFoundry OpenEnvision clone command">
            <code>git clone https://github.com/OpenEnvision/WorldFoundry.git</code>
          </div>
        </section>

        <footer className="pi-footer">
          <p>OpenEnvision</p>
          <div>
            <Link href="/docs">Docs</Link>
            <Link href="/docs/guides/supported-models">Supported Models</Link>
          </div>
        </footer>
      </div>
    </main>
  );
}
