# Component Parity and First-Divergence Diagnosis

Use this reference for checkpoint conversion or a failing numerical comparison.
Run commands from the repository root in the prepared model environment. The
[validation guide](../../../docs/fumadocs/content/docs/reference/validation.mdx)
owns accepted references, tolerances, and required replay gates.

## Align the Comparison

Pin the upstream code and checkpoint revisions, component configuration,
inputs, seed, schedule, precision, and executed attention backend. A seed alone
does not guarantee identical initial latents when preprocessing consumes RNG;
compare or supply the same noise and conditioning tensors.

Start with the smallest real component that preserves the failing computation.
Use evaluation/inference mode and FP32 where supported to isolate conversion
from lower-precision kernel differences. Then test the deployed dtype and shape.
Load one side at a time if needed, retaining an independent copy of its output.
Keep reference imports in test/debug code, never the production runtime.

## Verify Loading Before Activations

Follow the declared checkpoint roles through
[checkpoint specifications](../../../worldfoundry/base_models/diffusion_model/loaders/checkpoints.py)
and the canonical resolver/loader. Check expected, missing, and unexpected
keys, tensor shapes, dtypes, and any explicitly ignored source tensors.

For the transforms actually used by the port, inspect:

- Q/K/V packing or splitting, head order, and weight/bias pairing;
- modulation scale/shift ordering and linear-weight transposition;
- latent scaling, offsets, and per-channel normalization;
- tokenizer/config sidecars and checkpoint partition or expert selection.

Compare mapped tensor values, not just key counts. If the integration persists
a converted checkpoint, reload it through the public loader and compare the
reloaded tensors and forward output. Synthetic round trips verify loading
mechanics; released-weight component comparisons verify the actual conversion.
Both are narrower than full-pipeline acceptance.

## Find the First Broken Boundary

1. Compare tokenizer IDs/masks and encoded conditioning, latent initialization,
   timestep/sigma values, and model-input scaling.
2. Compare the first denoiser prediction before scheduler integration. If it
   matches, compare the scheduler update and continue to the first failed step.
3. If the denoiser differs, compare block outputs, then only the first divergent
   block's norm, attention, RoPE, modulation, or MLP boundaries.
4. If latents match but media differs, decode identical latents. Align decoder
   RNG, normalization, frame ordering, range, and output serialization.

At each boundary, record shape, dtype, finite-value checks, maximum and mean
absolute differences, and the applicable tolerance. Tensor statistics help
localize a failure but do not replace elementwise output comparison. For
nondeterministic paths, measure reference self-run variation as the validation
guide requires; an existing exact gate remains exact.

For forward outputs exposed by a pipeline, use the existing `capture_modules`
case field in the
[replay harness](../../../tests/manual/geometry_regression.py). For denoiser and
scheduler-step boundaries, use instance-local
[DiffusionExtension hooks](../../../worldfoundry/base_models/diffusion_model/extensions/base.py)
in the debug harness. Record matching component, branch, step, and tensor
boundaries on both sides. Copy observations without modifying live tensors.
Remove temporary forward hooks in `finally`; leave production code free of
diagnostic prints, class patches, and trace toggles.

## Return Actionable Evidence

Report the first divergent boundary and source `path:line`, aligned settings,
measured drift, and trace paths under `tmp/`. Label whether the passing check
covered a direct component, the public loader, or the complete pipeline. Run
the affected real-checkpoint gates after the fix; component parity does not
replace them. Missing weights or reference dependencies remain blockers.

The component-first and first-divergence procedures draw on the model/parity
skills in [Diffusers](https://github.com/huggingface/diffusers) and
[FastVideo](https://github.com/hao-ai-lab/FastVideo), adapted to WorldFoundry's
loading and evidence contracts.
