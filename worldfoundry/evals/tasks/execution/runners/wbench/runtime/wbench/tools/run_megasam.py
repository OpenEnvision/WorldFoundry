"""WBench MegaSAM precompute entry point backed by WorldFoundry base_models."""

from __future__ import annotations

from worldfoundry.base_models.three_dimensions.slam.megasam import main

if __name__ == "__main__":
    raise SystemExit(main())
