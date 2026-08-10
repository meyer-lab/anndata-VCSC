"""VCSC: a Value-Compressed Sparse Column/Row overlay for AnnData.

See :class:`~vcsc.VCSCArray` and :class:`~vcsc.VCSRArray` for the array
types, and :func:`~vcsc.from_anndata` to build one from an
:class:`anndata.AnnData` object.
"""

from vcsc._anndata import from_anndata, to_layer
from vcsc._base import VCSCArray, VCSRArray

__all__ = ["VCSCArray", "VCSRArray", "from_anndata", "to_layer"]

__version__ = "0.1.0"
