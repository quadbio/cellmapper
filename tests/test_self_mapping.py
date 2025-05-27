import pytest
import scanpy as sc

from cellmapper.cellmapper import CellMapper


class TestSelfMapping:
    """Tests for self-mapping functionality in CellMapper."""

    def test_self_mapping_initialization(self, adata_pbmc3k):
        """Test that self-mapping mode is correctly detected when reference=None."""
        # Initialize with only reference
        cm = CellMapper(adata_pbmc3k)
        assert cm._is_self_mapping
        assert cm.reference is adata_pbmc3k
        assert cm.query is adata_pbmc3k

    def test_identity_mapping(self, adata_pbmc3k):
        """Test that with n_neighbors=1, self-mapping preserves original labels exactly."""
        # Initialize with self-mapping
        cm = CellMapper(adata_pbmc3k)
        cm.fit(
            knn_method="sklearn",
            mapping_method="jaccard",
            obs_keys="leiden",
            use_rep="X_pca",
            n_neighbors=1,
            prediction_postfix="transfer",
        )

        # With n_neighbors=1, labels should be perfectly preserved
        assert "leiden_transfer" in adata_pbmc3k.obs
        assert len(adata_pbmc3k.obs["leiden_transfer"]) == len(adata_pbmc3k.obs["leiden"])
        # Check that all predicted labels are valid categories
        assert set(adata_pbmc3k.obs["leiden_transfer"].cat.categories) <= set(adata_pbmc3k.obs["leiden"].cat.categories)
        # Labels should match exactly when n_neighbors=1
        assert adata_pbmc3k.obs["leiden_transfer"].equals(adata_pbmc3k.obs["leiden"])

    def test_all_operations_self_mapping(self, adata_pbmc3k):
        """Test the full pipeline in self-mapping mode."""
        # Initialize with self-mapping
        cm = CellMapper(adata_pbmc3k)

        # Test with typical parameters
        cm.compute_neighbors(n_neighbors=5, use_rep="X_pca")
        cm.compute_mappping_matrix(method="gaussian")

        # Test label transfer
        cm.transfer_labels(obs_keys="leiden")
        assert "leiden_pred" in cm.query.obs
        # With n_neighbors>1, self-mapped labels might not be 100% identical

        # Test embedding transfer
        cm.transfer_embeddings(obsm_keys="X_pca")
        assert "X_pca_pred" in cm.query.obsm

        # Test expression transfer
        cm.transfer_expression(layer_key="X")
        assert cm.query_imputed is not None

        # Test evaluation functions
        cm.evaluate_label_transfer(label_key="leiden")
        assert cm.label_transfer_metrics is not None

        cm.evaluate_expression_transfer(layer_key="X", method="pearson")
        assert cm.expression_transfer_metrics is not None

    @pytest.mark.parametrize("n_neighbors", [5, 15, 30])
    def test_load_scanpy_distances(self, adata_spatial, n_neighbors):
        """Test loading distances computed with scanpy.pp.neighbors."""

        # Compute neighbors with scanpy
        sc.pp.neighbors(adata_spatial, n_neighbors=n_neighbors, use_rep="X_pca")

        # Initialize CellMapper in self-mapping mode
        cm = CellMapper(adata_spatial)

        # Load precomputed distances
        cm.load_precomputed_distances(distances_key="distances")

        # Verify the neighbors were properly loaded. Note that scanpy will return values for n_neighbors-1 neighbors.
        assert cm.knn is not None
        assert cm.knn.xx.n_neighbors + 1 == n_neighbors

        # Test the full pipeline with precomputed distances
        cm.compute_mappping_matrix(method="gaussian")
        cm.transfer_labels(obs_keys="leiden")

        assert "leiden_pred" in cm.query.obs
        assert "leiden_conf" in cm.query.obs

    @pytest.mark.parametrize(
        "squidpy_params",
        [
            # Test basic KNN approach
            {
                "n_neighs": 10,
                "library_key": None,
            },
            # Test with library_key
            {"n_neighs": 8, "library_key": "batch"},
            # Test Delaunay triangulation
            {
                "delaunay": True,
                "library_key": None,
            },
            # Test radius with set_diag=True
            {"radius": 10.0, "set_diag": True, "library_key": "batch", "coord_type": "generic"},
            # Test percentile with library_key
            {"percentile": 99.0, "library_key": "batch"},
        ],
    )
    def test_load_squidpy_distances(self, adata_spatial, squidpy_params):
        """Test loading distances computed with squidpy.gr.spatial_neighbors with various configurations."""
        # Skip test if squidpy is not installed
        pytest.importorskip("squidpy")
        import squidpy as sq

        # Compute spatial neighbors with squidpy using the provided parameters
        sq.gr.spatial_neighbors(adata_spatial, spatial_key="spatial", **squidpy_params)

        # Initialize CellMapper in self-mapping mode
        cm = CellMapper(adata_spatial)

        print(adata_spatial)

        # Load precomputed distances
        cm.load_precomputed_distances(distances_key="spatial_distances")

        # Verify the neighbors were properly loaded
        assert cm.knn is not None

        # Additional checks based on specific parameters
        if "delaunay" in squidpy_params and squidpy_params["delaunay"]:
            # Delaunay triangulation typically has more connections
            assert cm.knn.xx.n_neighbors >= 3, "Delaunay should create at least a few connections per cell"

        if "set_diag" in squidpy_params and squidpy_params["set_diag"]:
            assert (
                adata_spatial.obsp["spatial_connectivities"].diagonal()
                == cm.knn.xx.boolean_adjacency(set_diag=True).diagonal()
            ).all()

        # Test the mapping pipeline
        cm.compute_mappping_matrix(method="gaussian")
        cm.transfer_labels(obs_keys="leiden")

        assert "leiden_pred" in cm.query.obs
        assert "leiden_conf" in cm.query.obs

    @pytest.mark.parametrize("include_self", [True, False])
    def test_load_distances_with_include_self(self, adata_spatial, include_self):
        """Test loading precomputed distances with and without self-connections."""

        # Compute neighbors with scanpy
        sc.pp.neighbors(adata_spatial, n_neighbors=10, use_rep="X_pca")

        # Initialize CellMapper in self-mapping mode
        cm_with_self = CellMapper(adata_spatial)
        cm_without_self = CellMapper(adata_spatial)

        # Load precomputed distances with different include_self settings
        cm_with_self.load_precomputed_distances(distances_key="distances", include_self=True)
        cm_without_self.load_precomputed_distances(distances_key="distances", include_self=False)

        # Verify that neighbors were loaded with or without self
        assert cm_with_self.knn is not None
        assert cm_without_self.knn is not None

        # Check that with include_self=True, each cell has itself as a neighbor
        for i in range(min(10, cm_with_self.knn.xx.n_samples)):  # Check first 10 cells
            assert i in cm_with_self.knn.xx.indices[i]

        # Check that with include_self=False, no cell has itself as a neighbor
        for i in range(min(10, cm_without_self.knn.xx.n_samples)):  # Check first 10 cells
            assert i not in cm_without_self.knn.xx.indices[i]

        # Both should work with the rest of the pipeline
        cm_with_self.compute_mappping_matrix(method="gaussian")
        cm_without_self.compute_mappping_matrix(method="gaussian")

        # Compute label transfer for both
        cm_with_self.transfer_labels(obs_keys="leiden", prediction_postfix="with_self")
        cm_without_self.transfer_labels(obs_keys="leiden", prediction_postfix="without_self")

        # Both should have created prediction columns
        assert "leiden_with_self" in adata_spatial.obs
        assert "leiden_without_self" in adata_spatial.obs

        # The results should be different (excluding self changes the neighborhood)
        assert not adata_spatial.obs["leiden_with_self"].equals(adata_spatial.obs["leiden_without_self"])

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
