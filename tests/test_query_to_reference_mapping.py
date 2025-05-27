import numpy as np
import pytest
from scipy.sparse import issparse

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

    @pytest.mark.parametrize(
        "method", ["gaussian", "scarches", "random", "inverse_distance", "jaccard", "hnoca", "equal"]
    )
    def test_compute_mapping_matrix_all_methods(self, cmap, method):
        cmap.compute_mappping_matrix(method=method)
        assert cmap.mapping_matrix is not None

    @pytest.mark.parametrize("layer_key", ["X", "counts"])
    def test_expression_transfer_layers(self, cmap, layer_key):
        cmap.transfer_expression(layer_key=layer_key)
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
