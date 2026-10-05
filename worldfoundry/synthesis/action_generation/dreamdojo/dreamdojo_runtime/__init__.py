"""DreamDojo runtime package with Cosmos sources promoted to base_models."""

from __future__ import annotations

import sys

from worldfoundry.base_models.diffusion_model.models.networks import cosmos2 as _cosmos_predict2

from . import cosmos_oss as _cosmos_oss
from . import external as _external
from . import groot_dreams as _groot_dreams

sys.modules.setdefault("cosmos_oss", _cosmos_oss)
sys.modules.setdefault("cosmos_predict2", _cosmos_predict2)
sys.modules.setdefault("external", _external)
sys.modules.setdefault("groot_dreams", _groot_dreams)
