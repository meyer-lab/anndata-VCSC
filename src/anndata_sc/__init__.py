"""VCSC: a Value-Compressed Sparse Column/Row overlay for AnnData.

See :class:`~vcsc.VCSCArray` and :class:`~vcsc.VCSRArray` for the array
types, :func:`~anndata_sc.from_anndata` to build one from an
:class:`anndata.AnnData` object, and :class:`~vcsc.VCSCAnnData` for an
AnnData subclass that holds a VCSC/VCSR array as ``X`` directly.
"""

from anndata_sc._anndata import from_anndata, to_layer
from anndata_sc._anndata_class import VCSCAnnData
from anndata_sc._base import VCSCArray, VCSRArray
from anndata_sc._rapid_load import load_and_normalize, load_packed
from anndata_sc._vcs_norm import VCSCArrayNormalized, VCSRArrayNormalized

__all__ = [
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
