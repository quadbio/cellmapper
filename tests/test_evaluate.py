import pytest


class TestEvaluate:
    @pytest.mark.parametrize("eval_layer", ["X", "counts"])
    @pytest.mark.parametrize("method", ["pearson", "spearman", "js", "rmse"])
    @pytest.mark.parametrize("groupby", ["batch", "modality"])
    def test_evaluate_expression_transfer_layers_and_methods(self, cmap, eval_layer, method, groupby):
        cmap.transfer_expression(layer_key="X")
        cmap.evaluate_expression_transfer(layer_key=eval_layer, method=method, groupby=groupby)
        metrics = cmap.expression_transfer_metrics
        assert metrics["method"] == method
        assert metrics["n_valid_genes"] > 0
        assert cmap.query_imputed is not None
        assert cmap.query.var[f"metric_{method}"] is not None
        if groupby == "batch":
            assert cmap.query.varm[f"metric_{method}"] is not None
