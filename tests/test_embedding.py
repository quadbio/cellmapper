import pytest

from cellmapper import CellMapper


class TestEmbedding:
    """Class to test embedding functionality in CellMapper."""

    @pytest.mark.parametrize(
        "joint_pca_key,n_pca_components,pca_kwargs",
        [
            ("pca_joint", 10, {}),
            ("custom_pca", 5, {"svd_solver": "arpack"}),
        ],
    )
    def test_compute_neighbors_joint_pca(self, cmap, joint_pca_key, n_pca_components, pca_kwargs):
        cmap.compute_neighbors(
            n_neighbors=3,
            use_rep=None,
            joint_pca_key=joint_pca_key,
            n_pca_components=n_pca_components,
            pca_kwargs=pca_kwargs,
        )
        assert joint_pca_key in cmap.reference.obsm
        assert joint_pca_key in cmap.query.obsm
        assert cmap.reference.obsm[joint_pca_key].shape[1] == n_pca_components
        assert cmap.query.obsm[joint_pca_key].shape[1] == n_pca_components

    def test_self_mapping_without_rep(self, adata_pbmc3k):
        """Test self-mapping when use_rep=None, testing automatic PCA computation."""
        # Initialize with self-mapping
        cm = CellMapper(adata_pbmc3k)

        # Test with no representation provided
        cm.compute_neighbors(n_neighbors=5, use_rep=None, fallback_kwargs={"n_components": 10})
        cm.compute_mappping_matrix(method="gaussian")

        # Verify joint PCA was computed
        assert "X_pca" in adata_pbmc3k.obsm
        assert adata_pbmc3k.obsm["X_pca"].shape[1] == 10

        # Test rest of pipeline
        cm.transfer_labels(obs_keys="leiden")
        assert "leiden_pred" in cm.query.obs
