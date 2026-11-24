"""
Exposes the LingBotVASynthesis class directly under the package namespace.

This __init__.py file simplifies imports by allowing users to access
LingBotVASynthesis directly from the package, rather than needing to
import it from its specific submodule.
"""

from .lingbot_va_synthesis import LingBotVASynthesis

__all__ = ["LingBotVASynthesis"]
