# AlayaWorld v1.1 inference runtime

Adapted from AlayaLab/AlayaWorld, revision `ea03cfbb2e4c4e9102ed8ea8562e0b5370ca9b79`
(LTX-2 Community License Agreement; see root `THIRD-PARTY-NOTICES`). The `alaya/`,
`ltx2/` and `fastvideo/` modules and `prepare_i2v_inputs.py` are the inference subset of
that revision. Every file in this directory was modified by WorldFoundry: imports are
package-relative, the code is formatted to the repository style, ViGeo loads from
`worldfoundry.base_models.three_dimensions.depth.vigeo`, the Depth-Anything-3 depth
backend and its configuration fields are removed, and `prepare_i2v_inputs.main` accepts
an argument list. No training entrypoints are included.
