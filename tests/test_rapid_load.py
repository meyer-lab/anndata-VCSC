from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from vcsc import VCSCAnnData
from vcsc._rapid_load import load_and_normalize


def _reference_prepare(dense: np.ndarray, min_cell_counts: float, gene_threshold: float):
    """Direct translation of parafac2.normalize.prepare_dataset's math, on a plain array."""
    X = sp.csr_array(dense)
    cell_mask = np.ravel(X.sum(axis=1)) > min_cell_counts
    gene_mask = np.ravel(X.sum(axis=0)) > (gene_threshold * X.shape[0])
    Xf = sp.csr_array(X[cell_mask][:, gene_mask])
    Xf.data = Xf.data.astype(np.float32)

    counts_per_cell = np.ravel(Xf.sum(axis=1)).astype(np.float32)
    counts_per_cell /= np.median(counts_per_cell)
    Xf.data /= np.repeat(counts_per_cell, np.diff(Xf.indptr))

    gene_sums = np.ravel(Xf.sum(axis=0)).astype(np.float32)
    Xf.data /= gene_sums[Xf.indices]

    Xf.data *= np.float32(1000.0)
    Xf.data += np.float32(1.0)
    np.log10(Xf.data, out=Xf.data)

    return Xf, np.nonzero(cell_mask)[0], np.nonzero(gene_mask)[0]


def _write_ivcsr(tmp_path, dense: np.ndarray, name="data.ivcsr.h5ad"):
    adata = ad.AnnData(X=sp.csr_array(dense))
    vad = VCSCAnnData.from_anndata(adata, format="csr")
    path = tmp_path / name
    vad.write_h5ad(path, format="ivcsc")
    return path


@pytest.mark.parametrize(
    ("min_cell_counts", "gene_threshold"),
    [(-1.0, 0.0), (10.0, 0.05), (3.0, 0.2)],
)
def test_matches_reference_normalization(tmp_path, rng, min_cell_counts, gene_threshold):
    dense = rng.integers(0, 8, size=(120, 40)).astype(np.float64)
    dense[rng.random(dense.shape) < 0.5] = 0.0
    path = _write_ivcsr(tmp_path, dense)

    result = load_and_normalize(
        path, min_cell_counts=min_cell_counts, gene_threshold=gene_threshold
    )
    ref_X, ref_cells, ref_genes = _reference_prepare(dense, min_cell_counts, gene_threshold)

    np.testing.assert_array_equal(result.kept_cells, ref_cells)
    np.testing.assert_array_equal(result.kept_genes, ref_genes)
    assert result.X.shape == ref_X.shape
    np.testing.assert_allclose(result.X.toarray(), ref_X.toarray(), rtol=1e-5, atol=1e-5)


def test_no_filtering_keeps_everything(tmp_path, rng):
    dense = rng.integers(0, 5, size=(30, 10)).astype(np.float64)
    path = _write_ivcsr(tmp_path, dense)

    result = load_and_normalize(path, min_cell_counts=-1.0, gene_threshold=0.0)

    assert result.X.shape == dense.shape
    np.testing.assert_array_equal(result.kept_cells, np.arange(dense.shape[0]))
    np.testing.assert_array_equal(result.kept_genes, np.arange(dense.shape[1]))
