"""
scToolsRNA — a workbench of general-purpose single-cell RNA-seq utilities.

Maintained by the Wagner Lab (UCSF). Functions cover preprocessing, feature
selection and dimensionality reduction, differential expression, label transfer,
trajectory analysis, plotting, and I/O for scanpy/AnnData workflows.

All functions are re-exported at the top level for backward compatibility, so
both ``import scToolsRNA as sct; sct.get_variable_genes(...)`` and
``from scToolsRNA import get_variable_genes`` continue to work.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("sctoolsrna")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .classification import *
from .diffexp import *
from .dimensionality import *
from .plotting import *
from .utils import *
from .network import *
from .preprocess import *
from .readwrite import *
from .sparse import *
from .trajectories import *
from .colormaps import *
from .workflows import *
from .stitch import *
from .knn import *
from .labeltransfer import *
