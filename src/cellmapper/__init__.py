from importlib.metadata import version

from .logging import logger
from .model.cellmapper import CellMapper
from .model.neighbors import Neighbors

__all__ = ["logger", "CellMapper", "Neighbors"]

__version__ = version("cellmapper")
