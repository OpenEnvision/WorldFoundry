"""Reusable Wan optimization policies.

Family-specific opt-in helpers (TeaCache coefficients, CFG-skip decorator).
The cross-family TeaCache threshold is resolved in
:mod:`...models.denoisers.graph_wrapped`; this package keeps the original
polynomial coefficients and the batched CFG-skip wrapper used by some Wan
graphs.  Neither is installed unless a recipe or variant opts in.
"""
