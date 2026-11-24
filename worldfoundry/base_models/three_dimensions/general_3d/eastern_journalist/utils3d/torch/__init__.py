import itertools
from typing import TYPE_CHECKING

from ..helpers import lazy_import_all_from

module_members = {}

for module_name in ["utils", "transforms", "mesh", "maps", "rasterization"]:
    module_members[module_name] = lazy_import_all_from(globals(), "." + module_name)

__all__ = list(itertools.chain(*module_members.values()))


if TYPE_CHECKING:
    from .maps import *  # noqa: F403 - Upstream star import kept as written.
    from .mesh import *  # noqa: F403 - Upstream star import kept as written.
    from .rasterization import *  # noqa: F403 - Upstream star import kept as written.
    from .transforms import *  # noqa: F403 - Upstream star import kept as written.
    from .utils import *  # noqa: F403 - Upstream star import kept as written.
