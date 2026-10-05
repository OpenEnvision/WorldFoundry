"""Immutable architecture data for specialized Wan recipes.

This package holds EasyDict graphs copied from official Wan 2.1 / 2.2
release configs: T5 / VAE checkpoint names, DiT width, patch size,
and inference geometry (fps, shift, CFG).  Subpackages ``wan21`` and
``wan22`` pin one identity per file.  Action / LingBot overlays live
at this level and deepcopy a base config.

Values are data, not runtime objects.  Changing a field here changes
every recipe that imports that identity.
"""
