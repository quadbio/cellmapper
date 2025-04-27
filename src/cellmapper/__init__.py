from importlib.metadata import version

from .cellmapper import CellMapper
from .logging import logger

__all__ = ["logger", "CellMapper"]

__version__ = version("cellmapper")
