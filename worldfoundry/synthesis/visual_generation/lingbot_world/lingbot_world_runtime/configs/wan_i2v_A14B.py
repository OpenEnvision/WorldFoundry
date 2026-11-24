from easydict import EasyDict

from .shared_config import _load_yaml_config, wan_shared_cfg

# ------------------------ Wan I2V A14B ------------------------#

i2v_A14B = EasyDict(__name__="Config: Wan I2V A14B")
i2v_A14B.update(wan_shared_cfg)
_i2v_payload = _load_yaml_config("wan_i2v_A14B.yaml")
for key in ("vae_stride", "patch_size", "window_size", "sample_guide_scale"):
    if key in _i2v_payload:
        _i2v_payload[key] = tuple(_i2v_payload[key])
i2v_A14B.update(_i2v_payload)
