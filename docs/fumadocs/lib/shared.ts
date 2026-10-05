export const appName = 'WorldFoundry';
export const docsRoute = '/docs';
export const docsImageRoute = '/og/docs';
export const docsContentRoute = '/llms.mdx/docs';

export const gitConfig = {
  user: 'OpenEnvision',
  repo: 'WorldFoundry',
  branch: 'main',
};

export function getGithubContentUrls(pagePath: string) {
  const file = `docs/fumadocs/content/docs/${pagePath}`;
  const repoUrl = `https://github.com/${gitConfig.user}/${gitConfig.repo}`;

  return {
    blob: `${repoUrl}/blob/${gitConfig.branch}/${file}`,
    edit: `${repoUrl}/edit/${gitConfig.branch}/${file}`,
    raw: `https://raw.githubusercontent.com/${gitConfig.user}/${gitConfig.repo}/${gitConfig.branch}/${file}`,
  };
}
