# Echo-LongVideo inference runtime

Adapted from jd-opensource/JoyAI-Echo, revision `0b931fd42f5ef51410019ae41b261eb3b8936410`,
directory `echo_longvideo/` (LTX-2 Community License Agreement; see root `THIRD-PARTY-NOTICES`).
`ltx_core/`, `ltx_pipelines/` and `ltx_distillation/` come from `echo_longvideo/ltx-core/src`,
`echo_longvideo/ltx-pipelines/src` and `echo_longvideo/ltx-distillation/src`; `inference.py`,
`r2v_schema.py` and `scripts/precompute_conditioning.py` come from `echo_longvideo/`.

The directory holds the import closure of the R2V entrypoint (`inference`). Every file was
modified by WorldFoundry: imports are package-relative and the code is formatted to the repository
style without file-wide lint suppressions. The entrypoint and the conditioning precompute script no
longer edit `sys.path`, and the default configuration is read from
`worldfoundry/data/models/runtime/configs/joyai-echo-longvideo/`, where the released
configurations are bundled without their MSST settings. `ltx_pipelines/__init__.py`,
`ltx_distillation/__init__.py` and `ltx_distillation/models/__init__.py` keep only package
docstrings, so the training exports and unused upstream pipelines are omitted.

The production voice filter extracts the speech stem of memory audio with the Bandit separator of
MSST-WebUI, a separate GNU AGPL-3.0 work that upstream loads from a local checkout. It is not part
of this runtime: `msst_speech_filter.py` is omitted, and `audio_voice_filter.py` passes memory audio
through when `memory.voice_filter.enabled` is false and rejects non-empty memory audio when it is
true.
