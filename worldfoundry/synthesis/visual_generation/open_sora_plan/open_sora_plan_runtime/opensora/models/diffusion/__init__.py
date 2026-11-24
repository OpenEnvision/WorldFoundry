from .opensora_v1_3.modeling_inpaint import OpenSoraInpaint_v1_3_models
from .opensora_v1_3.modeling_opensora import OpenSora_v1_3_models

Diffusion_models = {}
Diffusion_models.update(OpenSora_v1_3_models)
Diffusion_models.update(OpenSoraInpaint_v1_3_models)

from .opensora_v1_3.modeling_inpaint import (  # noqa: E402 - Follows a module-level setup call, as upstream.
    OpenSoraInpaint_v1_3_models_class,
)
from .opensora_v1_3.modeling_opensora import (  # noqa: E402 - Follows a module-level setup call, as upstream.
    OpenSora_v1_3_models_class,
)

Diffusion_models_class = {}
Diffusion_models_class.update(OpenSora_v1_3_models_class)
Diffusion_models_class.update(OpenSoraInpaint_v1_3_models_class)
