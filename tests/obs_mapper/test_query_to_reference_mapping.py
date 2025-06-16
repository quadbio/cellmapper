import numpy as np
import pytest
from scipy.sparse import issparse
from scipy.stats import pearsonr

from cellmapper.model.cellmapper import CellMapper


def assert_metrics_close(actual: dict, expected: dict, atol=1e-3):
    for key, exp in expected.items():
        act = actual[key]
        if isinstance(exp, float | int | np.floating | np.integer):
            assert np.isclose(act, exp, atol=atol), f"{key}: {act} != {exp}"
        else:
            assert act == exp, f"{key}: {act} != {exp}"


class TestQueryToReferenceMapping:
    """Tests for query-to-reference mapping functionality in CellMapper."""

    def test_label_transfer(self, obs_mapper, expected_label_transfer_metrics):
        obs_mapper.map_obs(key="leiden")
        obs_mapper.evaluate_label_transfer(label_key="leiden")
        assert_metrics_close(obs_mapper.label_transfer_metrics, expected_label_transfer_metrics)

    def test_embedding_transfer(self, obs_mapper):
        obs_mapper.map_obsm(key="X_pca")
        assert "X_pca_pred" in obs_mapper.query.obsm
        assert obs_mapper.query.obsm["X_pca_pred"].shape[0] == obs_mapper.query.n_obs

    def test_expression_transfer(self, obs_mapper, expected_expression_transfer_metrics):
        obs_mapper.map_layers(key="X")
        obs_mapper.evaluate_expression_transfer(layer_key="X", method="pearson")
        assert_metrics_close(obs_mapper.expression_transfer_metrics, expected_expression_transfer_metrics)

    @pytest.mark.parametrize(
        "method", ["gaussian", "scarches", "random", "inverse_distance", "jaccard", "hnoca", "equal"]
    )
    def test_compute_mapping_matrix_all_methods(self, obs_mapper, method):
        obs_mapper.compute_mapping_matrix(method=method)
        assert obs_mapper.mapping_matrix is not None

    @pytest.mark.parametrize("layer_key", ["X", "counts"])
    def test_expression_transfer_layers(self, obs_mapper, layer_key):
        obs_mapper.map_layers(key=layer_key)
        assert obs_mapper.query_imputed is not None
        assert obs_mapper.query_imputed.X.shape[0] == obs_mapper.query.n_obs

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
    def test_fit_various_combinations(self, obs_mapper, obs_keys, obsm_keys, layer_key):
        obs_mapper.map(obs_keys=obs_keys, obsm_keys=obsm_keys, layer_key=layer_key)
        if obs_keys is not None:
            keys = [obs_keys] if isinstance(obs_keys, str) else obs_keys
            for key in keys:
                assert f"{key}_pred" in obs_mapper.query.obs
        if obsm_keys is not None:
            keys = [obsm_keys] if isinstance(obsm_keys, str) else obsm_keys
            for key in keys:
                assert f"{key}_pred" in obs_mapper.query.obsm
        if layer_key is not None:
            assert obs_mapper.query_imputed is not None
            assert obs_mapper.query_imputed.X.shape[0] == obs_mapper.query.n_obs

    def test_map_obs_self_mapping(self, query_reference_adata):
        """Check mapping to self."""
        _, reference = query_reference_adata
        cm = CellMapper(reference, reference)
        cm.map(
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

    def test_query_imputed_property_numpy_array(self, obs_mapper, random_imputed_data):
        """Test setting query_imputed with a numpy array."""
        # Set the imputed data using the property setter
        obs_mapper.query_imputed = random_imputed_data

        # Verify the AnnData object was created correctly
        assert obs_mapper.query_imputed is not None
        assert obs_mapper.query_imputed.X.shape == random_imputed_data.shape
        assert np.allclose(obs_mapper.query_imputed.X, random_imputed_data)

        # Verify metadata was copied correctly
        assert obs_mapper.query_imputed.obs.equals(obs_mapper.query.obs)
        assert obs_mapper.query_imputed.var.equals(obs_mapper.reference.var)

        # Test evaluation works with custom imputed data
        obs_mapper.evaluate_expression_transfer(layer_key="X", method="pearson")
        assert obs_mapper.expression_transfer_metrics is not None
        assert obs_mapper.expression_transfer_metrics["method"] == "pearson"

    def test_query_imputed_property_sparse_matrix(self, obs_mapper, sparse_imputed_data):
        """Test setting query_imputed with a sparse matrix."""
        # Set the imputed data using the property setter
        obs_mapper.query_imputed = sparse_imputed_data

        # Verify the AnnData object was created correctly
        assert obs_mapper.query_imputed is not None
        assert obs_mapper.query_imputed.X.shape == sparse_imputed_data.shape

        # Test sparse format is preserved
        assert issparse(obs_mapper.query_imputed.X)

        # Test evaluation works with custom sparse imputed data
        obs_mapper.evaluate_expression_transfer(layer_key="X", method="spearman")
        assert obs_mapper.expression_transfer_metrics is not None
        assert obs_mapper.expression_transfer_metrics["method"] == "spearman"

    def test_query_imputed_property_dataframe(self, obs_mapper, dataframe_imputed_data):
        """Test setting query_imputed with a pandas DataFrame."""
        # Set the imputed data using the property setter
        obs_mapper.query_imputed = dataframe_imputed_data

        # Verify the AnnData object was created correctly
        assert obs_mapper.query_imputed is not None
        assert obs_mapper.query_imputed.X.shape == dataframe_imputed_data.shape
        assert np.allclose(obs_mapper.query_imputed.X, dataframe_imputed_data.values)

        # Test evaluation works with DataFrame-sourced imputed data
        obs_mapper.evaluate_expression_transfer(layer_key="X", method="js")
        assert obs_mapper.expression_transfer_metrics is not None
        assert obs_mapper.expression_transfer_metrics["method"] == "js"

    def test_query_imputed_property_anndata(self, obs_mapper, custom_anndata_imputed):
        """Test setting query_imputed with a pre-made AnnData object."""
        # Set the imputed data using the property setter
        obs_mapper.query_imputed = custom_anndata_imputed

        # Verify the AnnData object was set correctly
        assert obs_mapper.query_imputed is custom_anndata_imputed

        # Test evaluation works with custom AnnData
        obs_mapper.evaluate_expression_transfer(layer_key="X", method="rmse")
        assert obs_mapper.expression_transfer_metrics is not None
        assert obs_mapper.expression_transfer_metrics["method"] == "rmse"

    def test_query_imputed_invalid_shape(self, obs_mapper, invalid_shape_data):
        """Test that setting query_imputed with wrong shape raises an error."""
        # Wrong shape - too few cells
        with pytest.raises(ValueError, match="shape mismatch"):
            obs_mapper.query_imputed = invalid_shape_data["too_few_cells"]

        # Wrong shape - too few genes
        with pytest.raises(ValueError, match="shape mismatch"):
            obs_mapper.query_imputed = invalid_shape_data["too_few_genes"]

    def test_query_imputed_invalid_type(self, obs_mapper):
        """Test that setting query_imputed with invalid type raises an error."""
        # Invalid type - a string
        with pytest.raises(TypeError):
            obs_mapper.query_imputed = "not a valid imputed data type"

        # Invalid type - a list
        with pytest.raises(TypeError):
            obs_mapper.query_imputed = [1, 2, 3]

    def test_query_imputed_integration_with_map_layers(self, obs_mapper, random_imputed_data):
        """Test that map_layers correctly uses the query_imputed property."""
        # First check query_imputed is None
        assert obs_mapper.query_imputed is None

        # Map expression
        obs_mapper.map_layers(key="X")

        # Verify query_imputed was set
        assert obs_mapper.query_imputed is not None
        assert obs_mapper.query_imputed.X.shape == (obs_mapper.query.n_obs, obs_mapper.reference.n_vars)

        if issparse(obs_mapper.query_imputed.X):
            # For sparse data, we'll convert a small subset to dense for comparison
            original_data_sample = obs_mapper.query_imputed.X[:5, :5].toarray()
        else:
            original_data_sample = obs_mapper.query_imputed.X[:5, :5].copy()

        # Set new random data
        obs_mapper.query_imputed = random_imputed_data

        # Verify the data was updated (using sample to avoid sparse matrix issues)
        if issparse(obs_mapper.query_imputed.X):
            new_data_sample = obs_mapper.query_imputed.X[:5, :5].toarray()
        else:
            new_data_sample = obs_mapper.query_imputed.X[:5, :5]

        # The samples should be different since we're using random data
        assert not np.allclose(original_data_sample, new_data_sample)

    @pytest.mark.parametrize("method", ["pearson", "spearman", "js", "rmse"])
    def test_evaluate_with_custom_imputation(self, obs_mapper, random_imputed_data, method):
        """Test evaluation with imputed data from an alternative method."""
        # Set the imputed data
        obs_mapper.query_imputed = random_imputed_data

        # Evaluate using the specified method
        obs_mapper.evaluate_expression_transfer(layer_key="X", method=method)
        assert obs_mapper.expression_transfer_metrics is not None
        assert obs_mapper.expression_transfer_metrics["method"] == method

    def test_imputation_without_copying(self, obs_mapper, random_imputed_data):
        """Test that query_imputed correctly reflects metadata changes."""
        # Set the imputed data
        obs_mapper.query_imputed = random_imputed_data

        # Check that obs and var have the same contents
        assert obs_mapper.query_imputed.obs.equals(obs_mapper.query.obs)
        assert obs_mapper.query_imputed.var.equals(obs_mapper.reference.var)

        # Modify query.obs
        test_key = "_test_metadata_update"
        obs_mapper.query.obs[test_key] = 1

        # Set imputed data again to trigger copy of updated metadata
        obs_mapper.query_imputed = random_imputed_data

        # Check that new metadata is reflected in the imputed data
        assert test_key in obs_mapper.query_imputed.obs
        assert obs_mapper.query_imputed.obs[test_key].iloc[0] == 1

    @pytest.mark.parametrize(
        "n_comps,fallback_representation,fallback_kwargs,key_added",
        [
            (10, "joint_pca", {"svd_solver": "arpack", "key_added": "X_pca"}, "X_pca"),
            (30, "fast_cca", {"scale_with_singular": True, "l2_scale": True, "key_added": "X_cca"}, "X_cca"),
            (10, "fast_cca", {"key_added": "X_fast_cca"}, "X_fast_cca"),
        ],
    )
    def test_compute_neighbors_fallback(self, obs_mapper, n_comps, fallback_representation, fallback_kwargs, key_added):
        obs_mapper.compute_neighbors(
            n_neighbors=3,
            use_rep=None,
            n_comps=n_comps,
            fallback_representation=fallback_representation,
            fallback_kwargs=fallback_kwargs,
        )
        assert key_added in obs_mapper.reference.obsm
        assert key_added in obs_mapper.query.obsm
        assert obs_mapper.reference.obsm[key_added].shape[1] == n_comps
        assert obs_mapper.query.obsm[key_added].shape[1] == n_comps

    def test_map_obs_numerical_data_type_detection(self, query_reference_adata):
        """Test that numerical data types are correctly detected in map_obs."""
        query, reference = query_reference_adata

        # Add some numerical data to reference
        reference.obs["numerical_score"] = np.random.rand(reference.n_obs)
        reference.obs["integer_score"] = np.random.randint(0, 100, reference.n_obs)

        # Create CellMapper and compute mapping matrix
        obs_mapper = CellMapper(query=query, reference=reference)
        obs_mapper.compute_neighbors(n_neighbors=30, use_rep="X_pca", method="sklearn")
        obs_mapper.compute_mapping_matrix(method="gaussian")

        # Test float and integer data
        for key in ["numerical_score", "integer_score"]:
            obs_mapper.map_obs(key=key)
            assert f"{key}_pred" in obs_mapper.query.obs
            assert obs_mapper.query.obs[f"{key}_pred"].dtype.kind == "f"

    def test_map_obs_pseudotime_cross_mapping(self, query_reference_adata):
        """Test mapping pseudotime values in cross-mapping mode - should still have reasonable correlation."""
        query, reference = query_reference_adata

        # Create CellMapper and compute mapping matrix
        obs_mapper = CellMapper(query=query, reference=reference)
        obs_mapper.compute_neighbors(n_neighbors=30, use_rep="X_pca", method="sklearn")
        obs_mapper.compute_mapping_matrix(method="gaussian")

        # Map pseudotime
        obs_mapper.map_obs(key="dpt_pseudotime")

        # Check that pseudotime was mapped
        assert "dpt_pseudotime_pred" in obs_mapper.query.obs
        assert obs_mapper.query.obs["dpt_pseudotime_pred"].dtype == reference.obs["dpt_pseudotime"].dtype

        # Check correlation between actual and predicted pseudotime in query subset
        # (Note: query is a subset of the original data, so we can compare)
        query_original_pt = query.obs["dpt_pseudotime"]
        query_predicted_pt = obs_mapper.query.obs["dpt_pseudotime_pred"]

        correlation, _ = pearsonr(query_original_pt, query_predicted_pt)

        # Cross-mapping should still have reasonably high correlation, though lower than self-mapping
        assert correlation > 0.99, f"Cross-mapping pseudotime correlation too low: {correlation}"

        # Verify no confidence scores for numerical data
        assert "dpt_pseudotime_conf" not in obs_mapper.query.obs
