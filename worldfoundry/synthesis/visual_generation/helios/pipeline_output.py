"""Compatibility import; implementation is shared in base_models."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module('worldfoundry.base_models.diffusion_model.runners.helios_output')
