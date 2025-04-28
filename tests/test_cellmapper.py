import numpy as np
import pytest


def assert_metrics_close(actual: dict, expected: dict, atol=1e-3):
    for key, exp in expected.items():
        act = actual[key]
        if isinstance(exp, float | int | np.floating | np.integer):
            assert np.isclose(act, exp, atol=atol), f"{key}: {act} != {exp}"
        else:
            assert act == exp, f"{key}: {act} != {exp}"


class TestCellMapper:
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

    @pytest.mark.parametrize("eval_layer", ["X", "counts"])
    @pytest.mark.parametrize("method", ["pearson", "spearman", "js", "rmse"])
    def test_evaluate_expression_transfer_layers_and_methods(self, cmap, eval_layer, method):
        cmap.transfer_expression(layer_key="X")
        cmap.evaluate_expression_transfer(layer_key=eval_layer, method=method)
        metrics = cmap.expression_transfer_metrics
        assert metrics["method"] == method
        assert metrics["n_valid_genes"] > 0
