"""Small inference-time compatibility subset for optional PyTorch3D imports."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
