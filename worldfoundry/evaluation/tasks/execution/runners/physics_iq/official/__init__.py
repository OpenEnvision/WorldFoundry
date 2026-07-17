"""Vendored official Physics-IQ metric implementation.

The source algorithms are derived from google-deepmind/physics-IQ-benchmark at
revision b02cf26dc15d559d0ca4f63a6917070312dde185. Individual modules retain the
upstream Apache-2.0 copyright headers. Benchmark prompt material is CC-BY-4.0.
"""

from .scoring import IQTable

__all__ = ["IQTable"]
