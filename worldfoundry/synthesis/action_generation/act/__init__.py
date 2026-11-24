"""
This module serves as the public interface for the `act_synthesis` subpackage.

It primarily re-exports the `ACTSynthesis` class, making it directly accessible
when importing from the parent package, simplifying access to the main action
synthesis component.
"""

from .act_synthesis import ACTSynthesis

# Define the public API of this module for 'from module import *' statements.
__all__ = ["ACTSynthesis"]
