"""
This module serves as a public interface for the policy synthesis functionalities within the package.

It re-exports the `OfficialPolicySynthesis` class, making it directly accessible when importing from the package
(e.g., `from your_package import OfficialPolicySynthesis`).
"""

from .official_policy_synthesis import OfficialPolicySynthesis

# Define the public API of this module.
# This list specifies which names should be imported when a client does 'from package import *'.
__all__ = ["OfficialPolicySynthesis"]
