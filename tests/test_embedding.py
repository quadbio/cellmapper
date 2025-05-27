import numpy as np
import pytest


class TestEmbedding:
    """Class to test embedding functionality in CellMapper."""

    @pytest.mark.parametrize(
        "zero_center,implicit",
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_fast_cca_embedding(self, cmap, zero_center, implicit):
        n_comps = 10
        cmap.compute_fast_cca(n_comps=n_comps, key_added="X_cca", zero_center=zero_center, implicit=implicit)
        # Check embeddings exist
        assert "X_cca" in cmap.query.obsm
        assert "X_cca" in cmap.reference.obsm
        Xq = cmap.query.obsm["X_cca"]
        Xr = cmap.reference.obsm["X_cca"]
        # Check shapes
        assert Xq.shape[1] == n_comps
        assert Xr.shape[1] == n_comps
        # Check not all zeros or NaNs
        assert not np.allclose(Xq, 0)
        assert not np.isnan(Xq).any()
        assert not np.allclose(Xr, 0)
        assert not np.isnan(Xr).any()

    def test_fast_cca_vs_joint_pca(self, cmap):
        n_comps = 8
        cmap.compute_fast_cca(n_comps=n_comps, key_added="X_cca")
        cmap.compute_joint_pca(n_comps=n_comps, key_added="X_pca")
        cca = cmap.query.obsm["X_cca"]
        pca = cmap.query.obsm["X_pca"]
        assert cca.shape == pca.shape
        # They should not be (almost) identical
        assert not np.allclose(cca, pca, rtol=1e-2, atol=1e-2)

    def test_joint_pca_embedding(self, cmap):
        n_comps = 12
        cmap.compute_joint_pca(n_comps=n_comps, key_added="X_pca")
        assert "X_pca" in cmap.query.obsm
        assert "X_pca" in cmap.reference.obsm
        Xq = cmap.query.obsm["X_pca"]
        Xr = cmap.reference.obsm["X_pca"]
        assert Xq.shape[1] == n_comps
        assert Xr.shape[1] == n_comps
        assert not np.allclose(Xq, 0)
        assert not np.isnan(Xq).any()
        assert not np.allclose(Xr, 0)
        assert not np.isnan(Xr).any()
