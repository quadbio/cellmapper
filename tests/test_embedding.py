import pytest

from cellmapper import CellMapper


class TestEmbedding:
    """Class to test embedding functionality in CellMapper."""

    @pytest.mark.parametrize(
        "n_comps,fallback_representation,fallback_kwargs,key_added",
        [
            (10, "joint_pca", {"svd_solver": "arpack", "key_added": "X_pca"}, "X_pca"),
            (30, "fast_cca", {"scale_with_singular": True, "l2_scale": True, "key_added": "X_cca"}, "X_cca"),
            (10, "fast_cca", {"key_added": "X_fast_cca"}, "X_fast_cca"),
        ],
    )
    def test_compute_neighbors_fallback(self, cmap, n_comps, fallback_representation, fallback_kwargs, key_added):
        cmap.compute_neighbors(
            n_neighbors=3,
            use_rep=None,
            n_comps=n_comps,
            fallback_representation=fallback_representation,
            fallback_kwargs=fallback_kwargs,
        )
        assert key_added in cmap.reference.obsm
        assert key_added in cmap.query.obsm
        assert cmap.reference.obsm[key_added].shape[1] == n_comps
        assert cmap.query.obsm[key_added].shape[1] == n_comps

    def test_self_mapping_without_rep(self, adata_pbmc3k):
        """Test self-mapping when use_rep=None, testing automatic PCA computation."""
        # Initialize with self-mapping
        cm = CellMapper(adata_pbmc3k)

        # Test with no representation provided
        cm.compute_neighbors(n_neighbors=5, use_rep=None, n_comps=10)
        cm.compute_mappping_matrix(method="gaussian")

        # Verify joint PCA was computed
        assert "X_pca" in adata_pbmc3k.obsm
        assert adata_pbmc3k.obsm["X_pca"].shape[1] == 10

        # Test rest of pipeline
        cm.transfer_labels(obs_keys="leiden")
        assert "leiden_pred" in cm.query.obs
