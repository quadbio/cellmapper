import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing


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
        "log,percentile,minmax",
        [
            (False, (0, 100), True),
            (True, (0, 100), True),
            (False, (5, 95), True),
            (True, (1, 99), True),
            (False, (0, 100), False),
            (True, (0, 100), False),
            (False, (5, 95), False),
            (True, (1, 99), False),
        ],
    )
    def test_presence_score_overall(self, cmap, log, percentile, minmax):
        cmap.compute_presence_score(log=log, percentile=percentile, minmax=minmax)
        assert "presence_score" in cmap.reference.obs
        scores = cmap.reference.obs["presence_score"]
        assert isinstance(scores, pd.Series | np.ndarray)
        if minmax:
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


class TestConfusionMatrix:
    """Tests for plot_confusion_matrix method."""

    def test_plot_confusion_matrix_basic(self, cmap):
        """Test basic confusion matrix plotting without colors."""
        cmap.map_obs(key="leiden")
        ax = cmap.plot_confusion_matrix("leiden", show_annotation_colors=False)
        assert ax is not None
        plt.close()

    def test_plot_confusion_matrix_with_colors(self, cmap):
        """Test confusion matrix plotting with annotation colors."""
        cmap.map_obs(key="leiden")

        # Set explicit colors for testing
        query_cats = cmap.query.obs["leiden"].cat.categories
        ref_cats = cmap.reference.obs["leiden"].cat.categories

        # Generate distinct colors for query and reference
        query_colors = [f"#{i * 30:02x}0000" for i in range(len(query_cats))]  # Red shades
        ref_colors = [f"#0000{i * 30:02x}" for i in range(len(ref_cats))]  # Blue shades

        cmap.query.uns["leiden_colors"] = query_colors
        cmap.reference.uns["leiden_colors"] = ref_colors

        print(f"Query uns keys: {list(cmap.query.uns.keys())}")
        print(f"Reference uns keys: {list(cmap.reference.uns.keys())}")
        print(f"Query leiden categories: {list(query_cats)}")
        print(f"Query colors: {query_colors}")
        print(f"Reference leiden categories: {list(ref_cats)}")
        print(f"Reference colors: {ref_colors}")

        ax = cmap.plot_confusion_matrix("leiden", show_annotation_colors=True)
        assert ax is not None

        # Check that annotation strips were added (patches on the axes)
        patches = [p for p in ax.patches if hasattr(p, "get_facecolor")]
        n_rows = len(cmap.query.obs["leiden"].unique())
        n_cols = len(cmap.reference.obs["leiden"].unique())
        print(f"Number of patches: {len(patches)} (expected at least {n_rows + n_cols})")
        assert len(patches) >= n_rows + n_cols, f"Should have at least {n_rows + n_cols} patches"
        plt.close()

    def test_get_category_colors_helper(self, cmap):
        """Test _get_category_colors helper function directly."""
        from cellmapper.model.evaluate import _get_category_colors

        cmap.map_obs(key="leiden")

        # Get categories from confusion matrix
        y_true = cmap.query.obs["leiden"].astype(str)
        y_pred = cmap.query.obs["leiden_pred"].astype(str)
        cm = pd.crosstab(y_true, y_pred)

        # Test getting colors for row categories (true labels from query)
        row_cats = list(cm.index)
        row_colors = _get_category_colors(cmap.query, "leiden", row_cats)
        print(f"Row categories: {row_cats}")
        print(f"Row colors: {row_colors}")
        assert len(row_colors) == len(row_cats)

        # Test getting colors for col categories (pred labels from reference)
        col_cats = list(cm.columns)
        col_colors = _get_category_colors(cmap.reference, "leiden", col_cats)
        print(f"Col categories: {col_cats}")
        print(f"Col colors: {col_colors}")
        assert len(col_colors) == len(col_cats)

        # Check that we're not getting all gray (which would mean colors not found)
        has_real_colors_row = any(c != "gray" for c in row_colors)
        has_real_colors_col = any(c != "gray" for c in col_colors)
        print(f"Row has real colors: {has_real_colors_row}")
        print(f"Col has real colors: {has_real_colors_col}")

        # If colors exist in .uns, they should be found
        if "leiden_colors" in cmap.query.uns:
            assert has_real_colors_row, "Should find colors in query.uns"
        if "leiden_colors" in cmap.reference.uns:
            assert has_real_colors_col, "Should find colors in reference.uns"

    def test_get_category_colors_with_explicit_colors(self, cmap):
        """Test _get_category_colors when colors are explicitly set."""
        from cellmapper.model.evaluate import _get_category_colors

        cmap.map_obs(key="leiden")

        # Explicitly set colors in query and reference
        query_cats = cmap.query.obs["leiden"].cat.categories
        ref_cats = cmap.reference.obs["leiden"].cat.categories

        # Generate colors for query
        import matplotlib.pyplot as plt

        query_colors = plt.cm.tab10.colors[: len(query_cats)]
        cmap.query.uns["leiden_colors"] = [
            f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b in query_colors
        ]

        # Generate different colors for reference
        ref_colors = plt.cm.Set3.colors[: len(ref_cats)]
        cmap.reference.uns["leiden_colors"] = [
            f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b in ref_colors
        ]

        print(f"Query categories: {list(query_cats)}")
        print(f"Query colors: {cmap.query.uns['leiden_colors']}")
        print(f"Reference categories: {list(ref_cats)}")
        print(f"Reference colors: {cmap.reference.uns['leiden_colors']}")

        # Get categories from confusion matrix
        y_true = cmap.query.obs["leiden"].astype(str)
        y_pred = cmap.query.obs["leiden_pred"].astype(str)
        cm = pd.crosstab(y_true, y_pred)

        # Test getting colors for row categories (true labels from query)
        row_cats = list(cm.index)
        row_colors = _get_category_colors(cmap.query, "leiden", row_cats)
        print(f"CM row categories: {row_cats}")
        print(f"Row colors from query: {row_colors}")

        # Test getting colors for col categories (pred labels from reference)
        col_cats = list(cm.columns)
        col_colors = _get_category_colors(cmap.reference, "leiden", col_cats)
        print(f"CM col categories: {col_cats}")
        print(f"Col colors from reference: {col_colors}")

        # Verify colors were found (not gray)
        assert all(c != "gray" for c in row_colors), f"Row colors should not be gray: {row_colors}"
        assert all(c != "gray" for c in col_colors), f"Col colors should not be gray: {col_colors}"

        # Verify colors match the expected colors from their source adata
        for cat, color in zip(row_cats, row_colors, strict=True):
            cat_idx = list(query_cats).index(cat)
            expected = cmap.query.uns["leiden_colors"][cat_idx]
            assert color == expected, f"Row color mismatch for {cat}: {color} != {expected}"

        for cat, color in zip(col_cats, col_colors, strict=True):
            cat_idx = list(ref_cats).index(cat)
            expected = cmap.reference.uns["leiden_colors"][cat_idx]
            assert color == expected, f"Col color mismatch for {cat}: {color} != {expected}"

    def test_get_category_colors_mismatched_categories(self, cmap):
        """Test when query and reference have different category orders or subsets."""
        from cellmapper.model.evaluate import _get_category_colors

        cmap.map_obs(key="leiden")

        # Check actual category differences between query and reference
        query_cats = set(cmap.query.obs["leiden"].cat.categories)
        ref_cats = set(cmap.reference.obs["leiden"].cat.categories)
        print(f"Query categories: {sorted(query_cats)}")
        print(f"Reference categories: {sorted(ref_cats)}")
        print(f"Only in query: {query_cats - ref_cats}")
        print(f"Only in reference: {ref_cats - query_cats}")

        # Set colors with same categories but DIFFERENT ORDER in .uns
        # This simulates what happens when scanpy generates colors independently
        query_cat_list = list(cmap.query.obs["leiden"].cat.categories)
        ref_cat_list = list(cmap.reference.obs["leiden"].cat.categories)

        # Query colors in natural order

        query_colors = [f"#query{i:02d}" for i in range(len(query_cat_list))]
        cmap.query.uns["leiden_colors"] = query_colors

        # Reference colors - same categories but colors assigned to different indices
        ref_colors = [f"#ref{i:02d}" for i in range(len(ref_cat_list))]
        cmap.reference.uns["leiden_colors"] = ref_colors

        print("\nQuery category -> color mapping:")
        for cat, col in zip(query_cat_list, query_colors, strict=True):
            print(f"  {cat} -> {col}")

        print("\nReference category -> color mapping:")
        for cat, col in zip(ref_cat_list, ref_colors, strict=True):
            print(f"  {cat} -> {col}")

        # Now test color retrieval
        test_cats = sorted(query_cats | ref_cats)
        row_colors = _get_category_colors(cmap.query, "leiden", test_cats)
        col_colors = _get_category_colors(cmap.reference, "leiden", test_cats)

        print(f"\nRetrieved colors for test_cats={test_cats}:")
        print(f"From query (rows):     {row_colors}")
        print(f"From reference (cols): {col_colors}")

        # Verify each category maps to the correct color from its source
        for i, cat in enumerate(test_cats):
            if cat in query_cat_list:
                expected_row = query_colors[query_cat_list.index(cat)]
                assert row_colors[i] == expected_row, f"Query color wrong for {cat}"
            else:
                assert row_colors[i] == "gray", f"Missing query cat {cat} should be gray"

            if cat in ref_cat_list:
                expected_col = ref_colors[ref_cat_list.index(cat)]
                assert col_colors[i] == expected_col, f"Reference color wrong for {cat}"
            else:
                assert col_colors[i] == "gray", f"Missing ref cat {cat} should be gray"

    def test_get_category_colors_type_error(self):
        """Test that _get_category_colors raises TypeError for wrong input."""
        from cellmapper.model.evaluate import _get_category_colors

        with pytest.raises(TypeError, match="Expected AnnData"):
            _get_category_colors([1, 2, 3], "leiden", ["A", "B"])

    def test_plot_confusion_matrix_partial_colors(self, cmap):
        """Test when only reference has colors but query doesn't."""
        cmap.map_obs(key="leiden")

        # Only set colors for reference (simulating common scenario where
        # query is spatial data without colors and reference is atlas with colors)
        ref_cats = cmap.reference.obs["leiden"].cat.categories
        ref_colors = [f"#00{i * 30:02x}00" for i in range(len(ref_cats))]  # Green shades
        cmap.reference.uns["leiden_colors"] = ref_colors

        # Make sure query doesn't have colors
        if "leiden_colors" in cmap.query.uns:
            del cmap.query.uns["leiden_colors"]

        print(f"Query has leiden_colors: {'leiden_colors' in cmap.query.uns}")
        print(f"Reference has leiden_colors: {'leiden_colors' in cmap.reference.uns}")

        ax = cmap.plot_confusion_matrix("leiden", show_annotation_colors=True)
        patches = [p for p in ax.patches if hasattr(p, "get_facecolor")]

        # Should still have patches (some gray, some colored)
        n_rows = len(cmap.query.obs["leiden"].unique())
        n_cols = len(cmap.reference.obs["leiden"].unique())
        print(f"Number of patches: {len(patches)}")
        assert len(patches) >= n_rows + n_cols
        plt.close()
