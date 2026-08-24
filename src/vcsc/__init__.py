"""VCSC: a Value-Compressed Sparse Column/Row overlay for AnnData.

See :class:`~vcsc.VCSCArray` and :class:`~vcsc.VCSRArray` for the array
types, :func:`~vcsc.from_anndata` to build one from an
:class:`anndata.AnnData` object, and :class:`~vcsc.VCSCAnnData` for an
AnnData subclass that holds a VCSC/VCSR array as ``X`` directly.
"""

from vcsc._anndata import from_anndata, to_layer
from vcsc._anndata_class import VCSCAnnData
from vcsc._base import VCSCArray, VCSRArray
from vcsc._ivcs import IVCSCArray, IVCSRArray
from vcsc._rapid_load import load_and_normalize, load_packed
from vcsc._vcs_norm import VCSCArrayNormalized, VCSRArrayNormalized

__all__ = [
    "IVCSCArray",
    "IVCSRArray",
    "VCSCAnnData",
    "VCSCArray",
    "VCSCArrayNormalized",
    "VCSRArray",
    "VCSRArrayNormalized",
    "from_anndata",
    "load_and_normalize",
    "load_packed",
    "to_layer",
]

__version__ = "0.1.0"
