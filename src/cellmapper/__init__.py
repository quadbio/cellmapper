from importlib.metadata import version

from .cellmapper import CellMapper
from .knn import Neighbors
from .logging import logger

__all__ = ["logger", "CellMapper", "Neighbors"]

__version__ = version("cellmapper")
