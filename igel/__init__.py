"""Top-level package for igel."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

from .igel import Igel, metrics_dict, models_dict

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.4.0"


__author__ = "Nidhal Baccouri"
__email__ = "nidhalbacc@gmail.com"
