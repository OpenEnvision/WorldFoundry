# Echo-WM inference runtime

Adapted from jd-opensource/JoyAI-Echo, revision `a08f1274573d7cd6ec66719f204951f4227a5173`,
directory `echo_wm/` (LTX-2 Community License Agreement; see root `THIRD-PARTY-NOTICES`).
`ltx_core/`, `ltx_pipelines/` and `ltx_causal/` come from `echo_wm/ltx-core/src`,
`echo_wm/ltx-pipelines/src` and `echo_wm/ltx-causal/src`; `helpers/`, `inference_wm.py` and
`inference_wm_causal.py` come from `echo_wm/`.

The directory holds the import closure of the Base (`inference_wm`) and Flash
(`inference_wm_causal`) entrypoints. Every file was modified by WorldFoundry: imports are
package-relative and the code is formatted to the repository style without file-wide lint
suppressions. The entrypoints no longer edit `sys.path`, read their default configurations from
`worldfoundry/data/models/runtime/configs/joyai-echo-wm/`, and start the MoGe field-of-view helper
as a package module. `ltx_pipelines/__init__.py` keeps only a package docstring, so the upstream
pipelines that neither entrypoint imports are omitted. `CausalTI2VidPipeline` takes its default
cache configuration from a module-level constant.
