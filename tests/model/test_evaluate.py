import numpy as np
import pandas as pd
import pytest


class TestEvaluate:
    @pytest.mark.parametrize("eval_layer", ["X", "counts"])
    @pytest.mark.parametrize("comparison_method", ["pearson", "spearman", "js", "rmse"])
    @pytest.mark.parametrize("groupby", ["batch", "modality"])
    def test_evaluate_expression_transfer_layers_and_methods(self, cmap, eval_layer, comparison_method, groupby):
        cmap.map_layers(key="X")
        cmap.evaluate_expression_transfer(layer_key=eval_layer, comparison_method=comparison_method, groupby=groupby)
        metrics = cmap.expression_transfer_metrics
        assert metrics["comparison_method"] == comparison_method
        assert metrics["n_test_genes"] > 0
        assert cmap.query_imputed is not None
        assert cmap.query.var[f"metric_{comparison_method}"] is not None
        if groupby == "batch":
            assert cmap.query.varm[f"metric_{comparison_method}"] is not None

    @pytest.mark.parametrize(
        "log,percentile",
        [
            (False, (0, 100)),
            (True, (0, 100)),
            (False, (5, 95)),
            (True, (1, 99)),
        ],
    )
    def test_presence_score_overall(self, cmap, log, percentile):
        cmap.compute_presence_score(log=log, percentile=percentile)
        assert "presence_score" in cmap.reference.obs
        scores = cmap.reference.obs["presence_score"]
        assert isinstance(scores, pd.Series | np.ndarray)
        assert np.all((scores >= 0) & (scores <= 1))
        assert not np.all(scores == 0)  # Should not be all zeros

    @pytest.mark.parametrize("groupby", ["batch", "modality"])
    def test_presence_score_groupby(self, cmap, groupby):
        cmap.compute_presence_score(groupby=groupby)
        # Overall score should always be present in .obs
        assert "presence_score" in cmap.reference.obs
        # Per-group scores should be present in .obsm
        assert "presence_score" in cmap.reference.obsm
        df = cmap.reference.obsm["presence_score"]
        assert isinstance(df, pd.DataFrame)
        assert all(np.all((df[col] >= 0) & (df[col] <= 1)) for col in df.columns)
        # Columns should match group names
        groups = cmap.query.obs[groupby].unique()
        assert set(df.columns) == set(groups)
