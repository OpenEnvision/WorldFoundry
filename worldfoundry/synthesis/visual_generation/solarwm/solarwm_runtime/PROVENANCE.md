# SolarWM inference runtime

`solarwm/` is adapted from Junchao-cs/SolarWM, revision
`2fe1ea9aca072f787bbba07ee9380b0e1ea7563b` (Apache License 2.0). `ltx_core/`,
`ltx_pipelines/` and `ltx_trainer/` are adapted from Lightricks/LTX-2 tag `v1.2.0`, revision
`d151147788a9284cca791edc6ce898007e727fe6` (LTX-2.x Community License Agreement). See root
`THIRD-PARTY-NOTICES` for both licenses and the SolarWM notice.

The directory holds the import closure of `solarwm infer` for the Wan2.2, LTX-2.5 and
MiniMax-H3 backends. Every file was modified by WorldFoundry: imports are package-relative,
the code is formatted to the repository style without file-wide lint suppressions, backend and
LTX provider modules resolve relative to their package, the vendored LTX packages report their
release version instead of an installed distribution, `ltx_pipelines/__init__.py` keeps only its
package docstring, `ltx_trainer/__init__.py` no longer edits `sys.path`, and launch provenance
records the Git state of this directory.

Only inference is included. The backends' `train` and `preencode` actions and the LTX provider's
training and preencoding runtimes raise `BackendContractError`; the `solarwm.training` package,
the stage training runtimes, preencoding writers and their helpers are omitted, as are the
training-only definitions of `wan22/runtime/stage0p5.py`, `wan22/runtime/stage2.py`,
`minimax_h3/runtime.py`, `minimax_h3/stage1.py` (`H3Stage1Core.forward_loss`) and
`ltx25/runtime.py`. `ShardedEMA`, which stage-2 inference uses to swap in EMA weights, moved from
`solarwm/training/ema.py` to `solarwm/runtime/ema.py`. The released inference configurations are bundled under
`worldfoundry/data/models/runtime/configs/solarwm/examples/`.
