from omegaconf import OmegaConf

# Define the resolver function
def replace_slash(value: str) -> str:
    return value.replace('/', '_')

# Register the resolver with Hydra
OmegaConf.register_new_resolver("replace_slash", replace_slash)
