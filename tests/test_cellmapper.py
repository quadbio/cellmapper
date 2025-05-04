import numpy as np
import pytest
import scanpy as sc
from scipy.sparse import issparse

from cellmapper.cellmapper import CellMapper


def assert_metrics_close(actual: dict, expected: dict, atol=1e-3):
    for key, exp in expected.items():
        act = actual[key]
        if isinstance(exp, float | int | np.floating | np.integer):
            assert np.isclose(act, exp, atol=atol), f"{key}: {act} != {exp}"
        else:
            assert act == exp, f"{key}: {act} != {exp}"


class TestQueryToReferenceMapping:
    """Tests for query-to-reference mapping functionality in CellMapper."""

    def test_label_transfer(self, cmap, expected_label_transfer_metrics):
        cmap.transfer_labels(obs_keys="leiden")
        cmap.evaluate_label_transfer(label_key="leiden")
        assert_metrics_close(cmap.label_transfer_metrics, expected_label_transfer_metrics)

    def test_embedding_transfer(self, cmap):
        cmap.transfer_embeddings(obsm_keys="X_pca")
        assert "X_pca_pred" in cmap.query.obsm
        assert cmap.query.obsm["X_pca_pred"].shape[0] == cmap.query.n_obs

    def test_expression_transfer(self, cmap, expected_expression_transfer_metrics):
        cmap.transfer_expression(layer_key="X")
        cmap.evaluate_expression_transfer(layer_key="X", method="pearson")
        assert_metrics_close(cmap.expression_transfer_metrics, expected_expression_transfer_metrics)

    @pytest.mark.parametrize("method", ["gaussian", "scarches", "random", "inverse_distance", "jaccard", "hnoca"])
    def test_compute_mapping_matrix_all_methods(self, cmap, method):
        cmap.compute_mappping_matrix(method=method)
        assert cmap.mapping_matrix is not None

    @pytest.mark.parametrize("layer_key", ["X", "counts"])
    def test_expression_transfer_layers(self, cmap, layer_key):
        cmap.transfer_expression(layer_key=layer_key)
        assert cmap.query_imputed is not None
        assert cmap.query_imputed.X.shape[0] == cmap.query.n_obs

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

    @pytest.mark.parametrize(
        "obs_keys,obsm_keys,layer_key",
        [
            ("leiden", None, None),
            (None, "X_pca", None),
            (None, None, "X"),
            ("leiden", "X_pca", None),
            ("leiden", None, "X"),
            (None, "X_pca", "X"),
            ("leiden", "X_pca", "X"),
        ],
    )
    def test_fit_various_combinations(self, cmap, obs_keys, obsm_keys, layer_key):
        cmap.fit(obs_keys=obs_keys, obsm_keys=obsm_keys, layer_key=layer_key)
        if obs_keys is not None:
            keys = [obs_keys] if isinstance(obs_keys, str) else obs_keys
            for key in keys:
                assert f"{key}_pred" in cmap.query.obs
        if obsm_keys is not None:
            keys = [obsm_keys] if isinstance(obsm_keys, str) else obsm_keys
            for key in keys:
                assert f"{key}_pred" in cmap.query.obsm
        if layer_key is not None:
            assert cmap.query_imputed is not None
            assert cmap.query_imputed.X.shape[0] == cmap.query.n_obs

    def test_transfer_labels_self_mapping(self, query_reference_adata):
        """Check mapping to self."""
        _, reference = query_reference_adata
        cm = CellMapper(reference, reference)
        cm.fit(
            knn_method="sklearn",
            mapping_method="jaccard",
            obs_keys="leiden",
            use_rep="X_pca",
            n_neighbors=1,
            prediction_postfix="transfer",
        )
        assert "leiden_transfer" in reference.obs
        assert len(reference.obs["leiden_transfer"]) == len(reference.obs["leiden"])
        # Check that all predicted labels are valid categories
        assert set(reference.obs["leiden_transfer"].cat.categories) <= set(reference.obs["leiden"].cat.categories)
        # If mapping to self, labels should match
        assert reference.obs["leiden_transfer"].equals(reference.obs["leiden"])

    def test_query_imputed_property_numpy_array(self, cmap, random_imputed_data):
        """Test setting query_imputed with a numpy array."""
        # Set the imputed data using the property setter
        cmap.query_imputed = random_imputed_data

        # Verify the AnnData object was created correctly
        assert cmap.query_imputed is not None
        assert cmap.query_imputed.X.shape == random_imputed_data.shape
        assert np.allclose(cmap.query_imputed.X, random_imputed_data)

        # Verify metadata was copied correctly
        assert cmap.query_imputed.obs.equals(cmap.query.obs)
        assert cmap.query_imputed.var.equals(cmap.reference.var)

        # Test evaluation works with custom imputed data
        cmap.evaluate_expression_transfer(layer_key="X", method="pearson")
        assert cmap.expression_transfer_metrics is not None
        assert cmap.expression_transfer_metrics["method"] == "pearson"

    def test_query_imputed_property_sparse_matrix(self, cmap, sparse_imputed_data):
        """Test setting query_imputed with a sparse matrix."""
        # Set the imputed data using the property setter
        cmap.query_imputed = sparse_imputed_data

        # Verify the AnnData object was created correctly
        assert cmap.query_imputed is not None
        assert cmap.query_imputed.X.shape == sparse_imputed_data.shape

        # Test sparse format is preserved
        assert issparse(cmap.query_imputed.X)

        # Test evaluation works with custom sparse imputed data
        cmap.evaluate_expression_transfer(layer_key="X", method="spearman")
        assert cmap.expression_transfer_metrics is not None
        assert cmap.expression_transfer_metrics["method"] == "spearman"

    def test_query_imputed_property_dataframe(self, cmap, dataframe_imputed_data):
        """Test setting query_imputed with a pandas DataFrame."""
        # Set the imputed data using the property setter
        cmap.query_imputed = dataframe_imputed_data

        # Verify the AnnData object was created correctly
        assert cmap.query_imputed is not None
        assert cmap.query_imputed.X.shape == dataframe_imputed_data.shape
        assert np.allclose(cmap.query_imputed.X, dataframe_imputed_data.values)

        # Test evaluation works with DataFrame-sourced imputed data
        cmap.evaluate_expression_transfer(layer_key="X", method="js")
        assert cmap.expression_transfer_metrics is not None
        assert cmap.expression_transfer_metrics["method"] == "js"

    def test_query_imputed_property_anndata(self, cmap, custom_anndata_imputed):
        """Test setting query_imputed with a pre-made AnnData object."""
        # Set the imputed data using the property setter
        cmap.query_imputed = custom_anndata_imputed

        # Verify the AnnData object was set correctly
        assert cmap.query_imputed is custom_anndata_imputed

        # Test evaluation works with custom AnnData
        cmap.evaluate_expression_transfer(layer_key="X", method="rmse")
        assert cmap.expression_transfer_metrics is not None
        assert cmap.expression_transfer_metrics["method"] == "rmse"

    def test_query_imputed_invalid_shape(self, cmap, invalid_shape_data):
        """Test that setting query_imputed with wrong shape raises an error."""
        # Wrong shape - too few cells
        with pytest.raises(ValueError, match="shape mismatch"):
            cmap.query_imputed = invalid_shape_data["too_few_cells"]

        # Wrong shape - too few genes
        with pytest.raises(ValueError, match="shape mismatch"):
            cmap.query_imputed = invalid_shape_data["too_few_genes"]

    def test_query_imputed_invalid_type(self, cmap):
        """Test that setting query_imputed with invalid type raises an error."""
        # Invalid type - a string
        with pytest.raises(TypeError):
            cmap.query_imputed = "not a valid imputed data type"

        # Invalid type - a list
        with pytest.raises(TypeError):
            cmap.query_imputed = [1, 2, 3]

    def test_query_imputed_integration_with_transfer_expression(self, cmap, random_imputed_data):
        """Test that transfer_expression correctly uses the query_imputed property."""
        # First check query_imputed is None
        assert cmap.query_imputed is None

        # Transfer expression
        cmap.transfer_expression(layer_key="X")

        # Verify query_imputed was set
        assert cmap.query_imputed is not None
        assert cmap.query_imputed.X.shape == (cmap.query.n_obs, cmap.reference.n_vars)

        if issparse(cmap.query_imputed.X):
            # For sparse data, we'll convert a small subset to dense for comparison
            original_data_sample = cmap.query_imputed.X[:5, :5].toarray()
        else:
            original_data_sample = cmap.query_imputed.X[:5, :5].copy()

        # Set new random data
        cmap.query_imputed = random_imputed_data

        # Verify the data was updated (using sample to avoid sparse matrix issues)
        if issparse(cmap.query_imputed.X):
            new_data_sample = cmap.query_imputed.X[:5, :5].toarray()
        else:
            new_data_sample = cmap.query_imputed.X[:5, :5]

        # The samples should be different since we're using random data
        assert not np.allclose(original_data_sample, new_data_sample)

    @pytest.mark.parametrize("method", ["pearson", "spearman", "js", "rmse"])
    def test_evaluate_with_custom_imputation(self, cmap, random_imputed_data, method):
        """Test evaluation with imputed data from an alternative method."""
        # Set the imputed data
        cmap.query_imputed = random_imputed_data

        # Evaluate using the specified method
        cmap.evaluate_expression_transfer(layer_key="X", method=method)
        assert cmap.expression_transfer_metrics is not None
        assert cmap.expression_transfer_metrics["method"] == method

    def test_imputation_without_copying(self, cmap, random_imputed_data):
        """Test that query_imputed correctly reflects metadata changes."""
        # Set the imputed data
        cmap.query_imputed = random_imputed_data

        # Check that obs and var have the same contents
        assert cmap.query_imputed.obs.equals(cmap.query.obs)
        assert cmap.query_imputed.var.equals(cmap.reference.var)

        # Modify query.obs
        test_key = "_test_metadata_update"
        cmap.query.obs[test_key] = 1

        # Set imputed data again to trigger copy of updated metadata
        cmap.query_imputed = random_imputed_data

        # Check that new metadata is reflected in the imputed data
        assert test_key in cmap.query_imputed.obs
        assert cmap.query_imputed.obs[test_key].iloc[0] == 1


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

    def test_self_mapping_without_rep(self, adata_pbmc3k):
        """Test self-mapping when use_rep=None, testing automatic PCA computation."""
        # Initialize with self-mapping
        cm = CellMapper(adata_pbmc3k)

        # Test with no representation provided
        cm.compute_neighbors(n_neighbors=5, use_rep=None, n_pca_components=10)
        cm.compute_mappping_matrix(method="gaussian")

        # Verify joint PCA was computed
        assert "X_pca" in adata_pbmc3k.obsm
        assert adata_pbmc3k.obsm["X_pca"].shape[1] == 10

        # Test rest of pipeline
        cm.transfer_labels(obs_keys="leiden")
        assert "leiden_pred" in cm.query.obs

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
