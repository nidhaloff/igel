"""Top-level package for igel."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

from .igel import Igel, metrics_dict, models_dict

__version__ = version(__name__)
__author__ = "Nidhal Baccouri"
__email__ = "nidhalbacc@gmail.com"
