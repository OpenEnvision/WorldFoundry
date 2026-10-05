"""
This module provides a dynamic import mechanism, allowing specific objects (e.g., classes, functions)
to be lazily loaded from submodules on demand.

It defines a mapping of user-facing names to their respective module paths. When an attribute
is accessed on this module that matches one of the defined names, the corresponding submodule
is imported, the requested object is retrieved, cached, and then returned. This avoids
eagerly importing all submodules when the main module is loaded.
"""

from importlib import import_module

# Defines a mapping of top-level export names to their relative module paths.
# When an attribute matching a key is accessed on this module, the module
# specified by the corresponding value will be imported, and the object
# with that key's name will be loaded from it.
_EXPORTS = {
    "MGVideoDiffusionTransformerI2V": ".dit",
    "MatrixGame1Synthesis": ".matrix_game_1_synthesis",
    "MatrixGame2Synthesis": ".matrix_game_2_synthesis",
    "MatrixGame3Synthesis": ".matrix_game_3_synthesis",
    "MatrixGame35FirstPersonSynthesis": ".matrix_game_3p5_synthesis",
    "MatrixGame35Synthesis": ".matrix_game_3p5_synthesis",
    "MatrixGame35ThirdPersonSynthesis": ".matrix_game_3p5_synthesis",
}

# Specifies the list of public names that are exported by this module when
# using `from your_module import *`. It is derived from the keys in `_EXPORTS`.
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """
    Implements a module-level dynamic attribute loader.

    This special function is called when an attribute `name` is accessed on
    this module that cannot be found in its `globals()` dictionary. It attempts
    to lazily load the requested object from a submodule based on the `_EXPORTS`
    mapping.

    Args:
        name: The name of the attribute being accessed (e.g., "MGVideoDiffusionTransformerI2V").

    Returns:
        The dynamically loaded object (e.g., a class or function) from the
        corresponding submodule.

    Raises:
        AttributeError: If the `name` is not found in the `_EXPORTS` mapping,
                        indicating that the attribute is not defined by this
                        dynamic loading mechanism.
    """
    if name not in _EXPORTS:
        # If the requested name is not in our known exports, raise an error
        # indicating that the attribute does not exist.
        raise AttributeError(name)

    # Import the target module specified in _EXPORTS relative to the current module.
    module = import_module(_EXPORTS[name], __name__)
    # Retrieve the actual object (class, function, etc.) from the newly imported module.
    value = getattr(module, name)
    # Cache the loaded object in the module's globals. This ensures that
    # subsequent accesses to 'name' will retrieve the object directly
    # without re-invoking __getattr__, improving performance.
    globals()[name] = value
    return value
