'use client';

import { useState } from 'react';

import { HomeRunConfigurator, type HomeRecipeOption } from '@/components/home-run-configurator';

export function HomeConfigureSection({ models }: { models: HomeRecipeOption[] }) {
  const [modelId, setModelId] = useState(models[0]?.id ?? '');

  return (
    <section className="wf-home-configure wf-home-reveal" aria-labelledby="wf-configure-title">
      <header className="wf-home-configure-heading">
        <div>
          <p className="wf-home-section-badge">
            <span aria-hidden="true" />
            Try it out
          </p>
          <h2 id="wf-configure-title">
            Configure a <span>run</span>
          </h2>
        </div>
        <p>
          Choose a real catalog entry. Runtime and version facts come from the repository
          manifests, and the command stays copyable.
        </p>
      </header>
      <HomeRunConfigurator
        models={models}
        selectedId={modelId}
        onSelectedIdChange={setModelId}
      />
    </section>
  );
}
