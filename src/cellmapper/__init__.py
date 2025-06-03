from importlib.metadata import version

from .logging import logger

# Legacy import for backward compatibility (deprecated)
from .model.cellmapper import CellMapper
from .model.knn import Neighbors
from .model.obs_mapper import ObsMapper
from .model.var_mapper import VarMapper

__all__ = [
    "logger",
    "ObsMapper",
    "VarMapper",
    "Neighbors",
]

__version__ = version("cellmapper")
