import katex from 'katex';

export function MathBlock({ tex }: { tex: string }) {
  const html = katex.renderToString(tex, {
    displayMode: true,
    throwOnError: false,
  });

  return <div className="katex-display" dangerouslySetInnerHTML={{ __html: html }} />;
}

export function MathInline({ tex }: { tex: string }) {
  const html = katex.renderToString(tex, {
    displayMode: false,
    throwOnError: false,
  });

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
