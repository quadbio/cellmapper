from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from cellmapper._docs import d
from cellmapper.logging import logger


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the Jensen-Shannon divergence between two expression vectors.

    Parameters
    ----------
    p, q
        Expression vectors.

    Returns
    -------
    The Jensen-Shannon divergence between p and q.
    """
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    return jensenshannon(p, q, base=10)


def _rmse_zscore(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the RMSE between z-scored versions of two arrays (per gene).

    Both a and b are 1D arrays of the same length (spots/cells for a single gene).

    Parameters
    ----------
    a, b
        Expression vectors (1D arrays) to compare.

    Returns
    -------
    The RMSE between the z-scored versions of a and b.
    """

    def zscore(x):
        mean = np.mean(x)
        std = np.std(x, ddof=0)
        if std == 0:
            std = 1
        return (x - mean) / std

    a_z = zscore(a)
    b_z = zscore(b)
    return np.sqrt(np.mean((a_z - b_z) ** 2))


class EvaluationMixin:
    """Mixin class for evaluation-related methods for CellMapper."""

    def register_external_predictions(
        self, label_key: str, prediction_postfix: str = "pred", confidence_postfix: str = "conf"
    ) -> None:
        """
        Register externally computed predictions for evaluation.

        Parameters
        ----------
        label_key
            Base key in .obs for the label (e.g., 'cell_type').
        prediction_postfix
            Postfix for prediction column in .obs (e.g., 'pred').
            The full column name should be f"{label_key}_{prediction_postfix}".
        confidence_postfix
            Postfix for confidence column in .obs (e.g., 'conf').
            The full column name should be f"{label_key}_{confidence_postfix}".

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``prediction_postfix``: Postfix for prediction column.
        - ``confidence_postfix``: Postfix for confidence column.
        """
        # Verify that the expected columns exist
        pred_col = f"{label_key}_{prediction_postfix}"
        conf_col = f"{label_key}_{confidence_postfix}"

        if pred_col not in self.query.obs.columns:
            raise ValueError(f"Prediction column '{pred_col}' not found in query.obs")
        if conf_col not in self.query.obs.columns:
            raise ValueError(f"Confidence column '{conf_col}' not found in query.obs")

        # Register the postfixes
        self.prediction_postfix = prediction_postfix
        self.confidence_postfix = confidence_postfix

        logger.info(
            "External predictions registered with prediction_postfix='%s' and confidence_postfix='%s'",
            prediction_postfix,
            confidence_postfix,
        )

    def evaluate_label_transfer(
        self,
        label_key: str,
        prediction_postfix: str | None = None,
        confidence_postfix: str | None = None,
        confidence_cutoff: float = 0.0,
        zero_division: int | Literal["warn"] = 0,
    ) -> None:
        """
        Evaluate label transfer using a k-NN classifier or externally computed predictions.

        Parameters
        ----------
        label_key
            Key in .obs storing ground-truth cell type annotations.
        prediction_postfix
            Postfix for prediction column in .obs. If None, uses self.prediction_postfix.
        confidence_postfix
            Postfix for confidence column in .obs. If None, uses self.confidence_postfix.
        confidence_cutoff
            Minimum confidence score required to include a cell in the evaluation.
        zero_division
            How to handle zero divisions in sklearn metrics computation.

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``label_transfer_metrics``: Dictionary containing accuracy, precision, recall, F1 scores, and excluded fraction.
        """
        # Use provided postfixes if given, otherwise fall back to instance attributes
        pred_postfix = prediction_postfix or self.prediction_postfix
        conf_postfix = confidence_postfix or self.confidence_postfix

        if pred_postfix is None or conf_postfix is None:
            raise ValueError(
                "Label transfer has not been performed. Either call map_obs() first "
                "or provide prediction_postfix and confidence_postfix parameters."
            )

        # Extract ground-truth and predicted labels
        y_true = self.query.obs[label_key].dropna()
        y_pred = self.query.obs.loc[y_true.index, f"{label_key}_{self.prediction_postfix}"]
        confidence = self.query.obs.loc[y_true.index, f"{label_key}_{self.confidence_postfix}"]

        # Apply confidence cutoff
        valid_indices = confidence >= confidence_cutoff
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]
        excluded_fraction = 1 - valid_indices.mean()

        # Compute classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=zero_division)

        # Log and store results
        self.label_transfer_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "excluded_fraction": excluded_fraction,
        }
        logger.info(
            "Accuracy: %.4f, Precision: %.4f, Recall: %.4f, Weighted F1-Score: %.4f, Macro F1-Score: %.4f, Excluded Fraction: %.4f",
            accuracy,
            precision,
            recall,
            f1_weighted,
            f1_macro,
            excluded_fraction,
        )

        # Optional: Save a detailed classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=zero_division)
        self.label_transfer_report = pd.DataFrame(report).transpose()

    def plot_confusion_matrix(
        self, label_key: str, figsize=(10, 8), cmap="viridis", save: str | Path | None = None, **kwargs
    ) -> None:
        """
        Plot the confusion matrix as a heatmap using sklearn's ConfusionMatrixDisplay.

        Parameters
        ----------
        figsize
            Size of the figure (width, height). Default is (10, 8).
        cmap
            Colormap to use for the heatmap. Default is "viridis".
        label_key
            Key in .obs storing ground-truth cell type annotations.
        **kwargs
            Additional keyword arguments to pass to ConfusionMatrixDisplay.
        """
        if self.prediction_postfix is None or self.confidence_postfix is None:
            raise ValueError("Label transfer has not been performed. Call map_obs() first.")

        # Extract true and predicted labels
        y_true = self.query.obs[label_key].dropna()
        y_pred = self.query.obs.loc[y_true.index, f"{label_key}_pred"]

        # Plot confusion matrix using sklearn's ConfusionMatrixDisplay
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=cmap, xticks_rotation="vertical", ax=ax, **kwargs)
        plt.title("Confusion Matrix")

        if save:
            plt.savefig(save, bbox_inches="tight")

    @d.dedent
    def evaluate_expression_transfer(
        self,
        layer_key: str = "X",
        comparison_method: Literal["pearson", "spearman", "js", "rmse"] = "pearson",
        groupby: str | None = None,
        test_var_key: str | None = None,
    ) -> None:
        """
        Evaluate the agreement between imputed and original expression in the query dataset, optionally per group.

        These metrics are inspired by :cite:`li2022benchmarking`.

        Parameters
        ----------
        %(layer_key)s
        %(comparison_method)s
        groupby
            Column in self.query.obs to group query cells by (e.g., cell type, batch). If None, computes a single score for all query cells.
        test_var_key
            Optional key in self.query.var where True marks test genes. If provided, average metrics are computed only over test genes.

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``expression_transfer_metrics``: Dictionary containing the average metric and number of genes used for the evaluation.
        - ``query.var[metric_name]``: Per-gene metric values (overall, across all cells).
        - ``query.varm[metric_name]``: Per-gene, per-group metric values (if groupby is provided).
        """
        imputed_x, original_x, shared_genes = self._get_aligned_expression_arrays(layer_key)

        # Select metric function
        if comparison_method == "pearson":
            metric_func = lambda a, b: pearsonr(a, b)[0]
        elif comparison_method == "spearman":
            metric_func = lambda a, b: spearmanr(a, b)[0]
        elif comparison_method in ("js", "jensen-shannon"):
            metric_func = _jensen_shannon_divergence
        elif comparison_method == "rmse":
            metric_func = _rmse_zscore
        else:
            raise NotImplementedError(f"Method '{comparison_method}' is not implemented.")

        # Helper to compute metrics for a given mask of cells
        def compute_metrics(mask):
            # Explicitly return as float32 to match DataFrame's dtype
            return np.array(
                [metric_func(original_x[mask, i], imputed_x[mask, i]) for i in range(imputed_x.shape[1])],
                dtype=np.float32,
            )

        # Compute metrics for all cells
        overall_mask = np.ones(original_x.shape[0], dtype=bool)
        overall_metrics = compute_metrics(overall_mask)
        self._store_expression_metric(
            shared_genes,
            overall_metrics,
            comparison_method,
            test_var_key,
        )

        if groupby is not None:
            # Prepare DataFrame to store per-group metrics
            group_labels = self.query.obs[groupby]
            groups = group_labels.unique()
            metrics_df = pd.DataFrame(
                np.full((self.query.n_vars, len(groups)), np.nan, dtype=np.float32),
                index=self.query.var_names,
                columns=groups,
            )

            # Compute and store metrics for each group
            for group in groups:
                mask = group_labels == group
                metrics_df.loc[shared_genes, group] = compute_metrics(mask.values)
            self.query.varm[f"metric_{comparison_method}"] = metrics_df

            logger.info(
                "Metrics per group defined in `query.obs['%s']` computed and stored in `query.varm['%s']`",
                groupby,
                f"metric_{comparison_method}",
            )

    @d.dedent
    def _get_aligned_expression_arrays(self, layer_key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Extract and align imputed and original expression arrays for shared genes between query_imputed and query.

        Parameters
        ----------
        %(layer_key)s

        Returns
        -------
        imputed_x, original_x, shared_genes
        """
        if self.query_imputed is None:
            raise ValueError(
                "Imputed query data not found. Either run map_layers() first or set query_imputed manually."
            )
        shared_genes = list(self.query_imputed.var_names.intersection(self.query.var_names))
        if len(shared_genes) == 0:
            raise ValueError("No shared genes between query_imputed and query.")
        imputed_x = self.query_imputed[:, shared_genes].X
        if layer_key == "X":
            original_x = self.query[:, shared_genes].X
        else:
            original_x = self.query[:, shared_genes].layers[layer_key]
        if issparse(imputed_x):
            imputed_x = imputed_x.toarray()
        if issparse(original_x):
            original_x = original_x.toarray()
        return imputed_x, original_x, shared_genes

    def _store_expression_metric(
        self,
        shared_genes: list[str],
        values: np.ndarray,
        comparison_method: str,
        test_var_key: str | None = None,
    ) -> None:
        """
        Store per-gene and summary expression transfer metrics in the query AnnData object and log the results.

        Parameters
        ----------
        shared_genes
            List of shared gene names.
        values
            Array of per-gene metric values (e.g., correlation, JSD) or 2D array (genes x groups).
        %(comparison_method)s
        test_var_key
            Optional key in self.query.var where True marks test genes. If provided, average metrics are computed only over test genes.
        """
        # Store overall metric in .var
        self.query.var[f"metric_{comparison_method}"] = np.nan
        self.query.var.loc[shared_genes, f"metric_{comparison_method}"] = values

        # Create a mask for valid (non-nan) values
        valid_mask = ~np.isnan(values)

        # Create a mask for valid test genes - by default, all non-nan values are valid
        self.query.var[f"_is_valid_test_gene_{comparison_method}"] = False
        self.query.var.loc[shared_genes, f"_is_valid_test_gene_{comparison_method}"] = valid_mask

        # If test_var_key provided, intersect with test gene mask
        n_test_genes = np.sum(valid_mask)
        if test_var_key is not None:
            # Update valid test genes to be both non-nan AND marked as test genes
            test_mask = self.query.var[test_var_key].astype(bool)
            valid_test_mask = pd.Series(False, index=self.query.var_names)
            valid_test_mask.loc[shared_genes] = valid_mask

            # Combine the masks
            self.query.var[f"_is_valid_test_gene_{comparison_method}"] = (
                self.query.var[f"_is_valid_test_gene_{comparison_method}"] & test_mask
            )

            n_test_genes = self.query.var[f"_is_valid_test_gene_{comparison_method}"].sum()
            if n_test_genes == 0:
                raise ValueError(f"No valid test genes found using '{test_var_key}'")

        # Get valid values using the combined mask
        valid_values = self.query.var.loc[
            self.query.var[f"_is_valid_test_gene_{comparison_method}"], f"metric_{comparison_method}"
        ]

        # Compute average metric
        avg_value = float(np.mean(valid_values))

        # Store metrics
        self.expression_transfer_metrics = {
            "comparison_method": comparison_method,
            "average": avg_value,
            "n_shared_genes": len(shared_genes),
            "n_test_genes": n_test_genes,
        }

        logger.info(
            "Expression transfer evaluation (%s): average value = %.4f (n_shared_genes=%d, n_test_genes=%d)",
            comparison_method,
            avg_value,
            len(shared_genes),
            n_test_genes,
        )

    def compute_presence_score(
        self,
        groupby: str | None = None,
        key_added: str = "presence_score",
        log: bool = False,
        percentile: tuple[float, float] = (1, 99),
    ):
        """
        Estimate raw presence scores for each reference cell based on query-to-reference connectivities.

        Adapted from the HNOCA-tools package :cite:`he2024integrated`.

        Parameters
        ----------
        groupby
            Column in self.query.obs to group query cells by (e.g., cell type, batch). If None, computes a single score for all query cells.
        key_added
            Key to store the presence score: always writes the score across all query cells to self.reference.obs[key_added].
            If groupby is not None, also writes per-group scores as a DataFrame to self.reference.obsm[key_added].
        log
            Whether to apply log1p transformation to the scores.
        percentile
            Tuple of (low, high) percentiles for clipping scores before normalization.
        """
        if self.knn is None or self.knn.yx is None:
            raise ValueError("Neighbors must be computed before estimating presence scores.")

        conn = self.knn.yx.knn_graph_connectivities()
        reference_names = self.reference.obs_names

        # Always compute and post-process the overall score (all query cells)
        scores_all = np.array(conn.sum(axis=0)).flatten()
        df_all = pd.DataFrame({"all": scores_all}, index=reference_names)
        df_all_processed = process_presence_scores(df_all, log=log, percentile=percentile)
        self.reference.obs[key_added] = df_all_processed["all"]
        logger.info("Presence score across all query cells computed and stored in `reference.obs['%s']`", key_added)

        # If groupby, also compute and post-process per-group scores
        if groupby is not None:
            group_labels = self.query.obs[groupby]
            groups = group_labels.unique()
            score_matrix = np.zeros((len(reference_names), len(groups)), dtype=np.float32)
            for i, group in enumerate(groups):
                mask = group_labels == group
                group_conn = conn[mask.values, :]
                score_matrix[:, i] = np.array(group_conn.sum(axis=0)).flatten()
            df_groups = pd.DataFrame(score_matrix, index=reference_names, columns=groups)
            df_groups_processed = process_presence_scores(df_groups, log=log, percentile=percentile)
            self.reference.obsm[key_added] = df_groups_processed

            logger.info(
                "Presence scores per group defined in `query.obs['%s']` computed and stored in `reference.obsm['%s']`",
                groupby,
                key_added,
            )


def process_presence_scores(
    scores: pd.DataFrame,
    log: bool = False,
    percentile: tuple[float, float] = (1, 99),
) -> pd.DataFrame:
    """
    Post-process presence scores with log1p, percentile clipping, and min-max normalization.

    Parameters
    ----------
    scores
        DataFrame of raw presence scores (rows: reference cells, columns: groups or 'all').
    log
        Whether to apply log1p transformation to the scores.
    percentile
        Tuple of (low, high) percentiles for clipping scores before normalization.

    Returns
    -------
    pd.DataFrame
        Post-processed presence scores, same shape as input.
    """
    # Log1p transformation (optional)
    if log:
        scores = np.log1p(scores)

    # Percentile clipping (optional)
    if percentile != (0, 100):
        low, high = percentile
        scores = scores.apply(lambda x: np.clip(x, np.percentile(x, low), np.percentile(x, high)), axis=0)

    # Min-max normalization (always)
    def minmax(x):
        min_val, max_val = np.min(x), np.max(x)
        return (x - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(x)

    scores = scores.apply(minmax, axis=0)

    return scores
