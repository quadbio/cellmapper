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

    def test_label_transfer(self, cmap, expected_label_transfer_metrics):
        cmap.map_obs(key="leiden")
        cmap.evaluate_label_transfer(label_key="leiden")
        assert_metrics_close(cmap.label_transfer_metrics, expected_label_transfer_metrics)

    def test_embedding_transfer(self, cmap):
        cmap.map_obsm(key="X_pca")
        assert "X_pca_pred" in cmap.query.obsm
        assert cmap.query.obsm["X_pca_pred"].shape[0] == cmap.query.n_obs

    def test_expression_transfer(self, cmap, expected_expression_transfer_metrics):
        cmap.map_layers(key="X")
        cmap.evaluate_expression_transfer(layer_key="X", comparison_method="pearson")
        assert_metrics_close(cmap.expression_transfer_metrics, expected_expression_transfer_metrics)

    @pytest.mark.parametrize(
        "kernel_method", ["gauss", "scarches", "random", "inverse_distance", "jaccard", "hnoca", "equal"]
    )
    def test_compute_mapping_matrix_all_methods(self, cmap, kernel_method):
        cmap.compute_mapping_matrix(kernel_method=kernel_method)
        assert cmap.mapping_operator.matrix is not None

    @pytest.mark.parametrize("layer_key", ["X", "counts"])
    def test_expression_transfer_layers(self, cmap, layer_key):
        cmap.map_layers(key=layer_key)
        assert cmap.query_imputed is not None
        assert cmap.query_imputed.X.shape[0] == cmap.query.n_obs

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
        cmap.map(obs_keys=obs_keys, obsm_keys=obsm_keys, layer_key=layer_key)
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

    def test_map_obs_self_mapping(self, query_reference_adata):
        """Check mapping to self."""
        _, reference = query_reference_adata
        cm = CellMapper(reference, reference)
        cm.map(
            knn_method="sklearn",
            kernel_method="jaccard",
            obs_keys="leiden",
            use_rep="X_pca",
            n_neighbors=1,
            prediction_postfix="transfer",
            # For n_neighbors=1 identity mapping, we need self-edges and no symmetrization
            symmetrize=False,
            self_edges=True,
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
        cmap.evaluate_expression_transfer(layer_key="X", comparison_method="pearson")
        assert cmap.expression_transfer_metrics is not None
        assert cmap.expression_transfer_metrics["comparison_method"] == "pearson"

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
        cmap.evaluate_expression_transfer(layer_key="X", comparison_method="spearman")
        assert cmap.expression_transfer_metrics is not None
        assert cmap.expression_transfer_metrics["comparison_method"] == "spearman"

    def test_query_imputed_property_dataframe(self, cmap, dataframe_imputed_data):
        """Test setting query_imputed with a pandas DataFrame."""
        # Set the imputed data using the property setter
        cmap.query_imputed = dataframe_imputed_data

        # Verify the AnnData object was created correctly
        assert cmap.query_imputed is not None
        assert cmap.query_imputed.X.shape == dataframe_imputed_data.shape
        assert np.allclose(cmap.query_imputed.X, dataframe_imputed_data.values)

        # Test evaluation works with DataFrame-sourced imputed data
        cmap.evaluate_expression_transfer(layer_key="X", comparison_method="js")
        assert cmap.expression_transfer_metrics is not None
        assert cmap.expression_transfer_metrics["comparison_method"] == "js"

    def test_query_imputed_property_anndata(self, cmap, custom_anndata_imputed):
        """Test setting query_imputed with a pre-made AnnData object."""
        # Set the imputed data using the property setter
        cmap.query_imputed = custom_anndata_imputed

        # Verify the AnnData object was set correctly
        assert cmap.query_imputed is custom_anndata_imputed

        # Test evaluation works with custom AnnData
        cmap.evaluate_expression_transfer(layer_key="X", comparison_method="rmse")
        assert cmap.expression_transfer_metrics is not None
        assert cmap.expression_transfer_metrics["comparison_method"] == "rmse"

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

    def test_query_imputed_integration_with_map_layers(self, cmap, random_imputed_data):
        """Test that map_layers correctly uses the query_imputed property."""
        # First check query_imputed is None
        assert cmap.query_imputed is None

        # Map expression
        cmap.map_layers(key="X")

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

    @pytest.mark.parametrize("comparison_method", ["pearson", "spearman", "js", "rmse"])
    def test_evaluate_with_custom_imputation(self, cmap, random_imputed_data, comparison_method):
        """Test evaluation with imputed data from an alternative method."""
        # Set the imputed data
        cmap.query_imputed = random_imputed_data

        # Evaluate using the specified method
        cmap.evaluate_expression_transfer(layer_key="X", comparison_method=comparison_method)
        assert cmap.expression_transfer_metrics is not None
        assert cmap.expression_transfer_metrics["comparison_method"] == comparison_method

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

    def test_map_obs_numerical_data_type_detection(self, query_reference_adata):
        """Test that numerical data types are correctly detected in map_obs."""
        query, reference = query_reference_adata

        # Add some numerical data to reference
        reference.obs["numerical_score"] = np.random.rand(reference.n_obs)
        reference.obs["integer_score"] = np.random.randint(0, 100, reference.n_obs)

        # Create CellMapper and compute mapping matrix
        cmap = CellMapper(query=query, reference=reference)
        cmap.compute_neighbors(n_neighbors=30, use_rep="X_pca", knn_method="sklearn")
        cmap.compute_mapping_matrix(kernel_method="gauss")

        # Test float and integer data
        for key in ["numerical_score", "integer_score"]:
            cmap.map_obs(key=key)
            assert f"{key}_pred" in cmap.query.obs
            assert cmap.query.obs[f"{key}_pred"].dtype.kind == "f"

    def test_map_obs_pseudotime_cross_mapping(self, query_reference_adata):
        """Test mapping pseudotime values in cross-mapping mode - should still have reasonable correlation."""
        query, reference = query_reference_adata

        # Create CellMapper and compute mapping matrix
        cmap = CellMapper(query=query, reference=reference)
        cmap.compute_neighbors(n_neighbors=30, use_rep="X_pca", knn_method="sklearn")
        cmap.compute_mapping_matrix(kernel_method="gauss")

        # Map pseudotime
        cmap.map_obs(key="dpt_pseudotime")

        # Check that pseudotime was mapped
        assert "dpt_pseudotime_pred" in cmap.query.obs
        assert cmap.query.obs["dpt_pseudotime_pred"].dtype == reference.obs["dpt_pseudotime"].dtype

        # Check correlation between actual and predicted pseudotime in query subset
        # (Note: query is a subset of the original data, so we can compare)
        query_original_pt = query.obs["dpt_pseudotime"]
        query_predicted_pt = cmap.query.obs["dpt_pseudotime_pred"]

        correlation, _ = pearsonr(query_original_pt, query_predicted_pt)

        # Cross-mapping should still have reasonably high correlation, though lower than self-mapping
        assert correlation > 0.99, f"Cross-mapping pseudotime correlation too low: {correlation}"

        # Verify no confidence scores for numerical data
        assert "dpt_pseudotime_conf" not in cmap.query.obs

    def test_map_obs_subset_categories(self, query_reference_adata):
        """Test mapping with subset_categories parameter for categorical data."""
        query, reference = query_reference_adata

        # Create CellMapper and compute mapping matrix
        cmap = CellMapper(query=query, reference=reference)
        cmap.compute_neighbors(n_neighbors=30, use_rep="X_pca", knn_method="sklearn")
        cmap.compute_mapping_matrix(kernel_method="gauss")

        # Get available leiden categories in reference
        available_categories = list(reference.obs["leiden"].cat.categories)

        # Test with subset of categories
        subset_cats = available_categories[:2]  # Take first 2 categories
        cmap.map_obs(key="leiden", subset_categories=subset_cats)

        # Check that mapping was performed
        assert "leiden_pred" in cmap.query.obs
        assert "leiden_conf" in cmap.query.obs

        # Check that predictions only contain subset categories (or might be missing if no assignment)
        predicted_categories = set(cmap.query.obs["leiden_pred"].dropna().unique())
        assert predicted_categories.issubset(set(subset_cats)), (
            f"Predicted categories {predicted_categories} not subset of {subset_cats}"
        )

    def test_map_obs_subset_categories_single_string(self, query_reference_adata):
        """Test mapping with subset_categories as single string."""
        query, reference = query_reference_adata

        # Create CellMapper and compute mapping matrix
        cmap = CellMapper(query=query, reference=reference)
        cmap.compute_neighbors(n_neighbors=30, use_rep="X_pca", knn_method="sklearn")
        cmap.compute_mapping_matrix(kernel_method="gauss")

        # Get first available category
        first_category = reference.obs["leiden"].cat.categories[0]

        # Test with single category as string
        cmap.map_obs(key="leiden", subset_categories=first_category)

        # Check that mapping was performed and only contains the specified category
        assert "leiden_pred" in cmap.query.obs
        predicted_categories = set(cmap.query.obs["leiden_pred"].dropna().unique())
        assert predicted_categories.issubset({first_category}), (
            f"Predicted categories {predicted_categories} not subset of {first_category}"
        )

    def test_map_obs_subset_categories_invalid_categories(self, query_reference_adata, caplog):
        """Test mapping with some invalid categories in subset_categories."""
        query, reference = query_reference_adata

        # Create CellMapper and compute mapping matrix
        cmap = CellMapper(query=query, reference=reference)
        cmap.compute_neighbors(n_neighbors=30, use_rep="X_pca", knn_method="sklearn")
        cmap.compute_mapping_matrix(kernel_method="gauss")

        # Mix valid and invalid categories
        valid_category = reference.obs["leiden"].cat.categories[0]
        invalid_categories = ["nonexistent1", "nonexistent2"]
        mixed_categories = [valid_category] + invalid_categories

        # Test with mixed valid/invalid categories - should work without errors
        cmap.map_obs(key="leiden", subset_categories=mixed_categories)

        # Check that mapping still worked with valid categories
        assert "leiden_pred" in cmap.query.obs
        predicted_categories = set(cmap.query.obs["leiden_pred"].dropna().unique())
        assert predicted_categories.issubset({valid_category})

    def test_map_obs_subset_categories_all_invalid(self, query_reference_adata, caplog):
        """Test mapping with all invalid categories in subset_categories."""
        query, reference = query_reference_adata

        # Create CellMapper and compute mapping matrix
        cmap = CellMapper(query=query, reference=reference)
        cmap.compute_neighbors(n_neighbors=30, use_rep="X_pca", knn_method="sklearn")
        cmap.compute_mapping_matrix(kernel_method="gauss")

        # Use only invalid categories
        invalid_categories = ["nonexistent1", "nonexistent2"]

        # Test with all invalid categories - should fallback to using all categories
        cmap.map_obs(key="leiden", subset_categories=invalid_categories)

        # Check that mapping still worked with all categories (fallback)
        assert "leiden_pred" in cmap.query.obs
        # Should have predictions from all available categories since it fell back
        predicted_categories = set(cmap.query.obs["leiden_pred"].dropna().unique())
        available_categories = set(reference.obs["leiden"].cat.categories)
        # At least one category should be predicted (could be subset due to k-NN mapping)
        assert len(predicted_categories) > 0
        assert predicted_categories.issubset(available_categories)

    def test_map_obs_subset_categories_numerical_warning(self, query_reference_adata, caplog):
        """Test that subset_categories generates warning for numerical data."""
        query, reference = query_reference_adata

        # Create CellMapper and compute mapping matrix
        cmap = CellMapper(query=query, reference=reference)
        cmap.compute_neighbors(n_neighbors=30, use_rep="X_pca", knn_method="sklearn")
        cmap.compute_mapping_matrix(kernel_method="gauss")

        # Test with numerical data and subset_categories - should work and ignore the parameter
        cmap.map_obs(key="dpt_pseudotime", subset_categories=["some_category"])

        # Check that mapping still worked normally (parameter was ignored)
        assert "dpt_pseudotime_pred" in cmap.query.obs
        # Confidence scores should not be created for numerical data
        assert "dpt_pseudotime_conf" not in cmap.query.obs

    def test_map_method_with_subset_categories(self, query_reference_adata):
        """Test that subset_categories parameter works through the high-level map method."""
        query, reference = query_reference_adata

        # Create CellMapper
        cmap = CellMapper(query=query, reference=reference)

        # Get available categories
        available_categories = list(reference.obs["leiden"].cat.categories)
        subset_cats = available_categories[:2]

        # Test high-level map method with subset_categories
        cmap.map(
            obs_keys="leiden", n_neighbors=30, use_rep="X_pca", kernel_method="gauss", subset_categories=subset_cats
        )

        # Check that mapping was performed with subset
        assert "leiden_pred" in cmap.query.obs
        predicted_categories = set(cmap.query.obs["leiden_pred"].dropna().unique())
        assert predicted_categories.issubset(set(subset_cats))
