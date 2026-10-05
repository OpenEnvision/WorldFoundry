# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""WorldCrafter uses the shared Helios Euler/UniPC/DMD scheduler unchanged."""
from .helios import HeliosScheduler as WorldCrafterScheduler
from .helios import HeliosSchedulerOutput as WorldCrafterSchedulerOutput
