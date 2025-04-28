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
    p = p / p.sum()
    q = q / q.sum()
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


class CellMapperEvaluationMixin:
    """Mixin class for evaluation-related methods for CellMapper."""

    def evaluate_label_transfer(
        self,
        label_key: str,
        confidence_cutoff: float = 0.0,
        zero_division: int | Literal["warn"] = 0,
    ) -> None:
        """
        Evaluate label transfer using a k-NN classifier.

        Parameters
        ----------
        label_key
            Key in .obs storing ground-truth cell type annotations.
        confidence_cutoff
            Minimum confidence score required to include a cell in the evaluation.
        zero_divisions
            How to handle zero divisions in sklearn metrics comptuation.

        Returns
        -------
        Nothing, but updates the following attributes:
        label_transfer_metrics
            Dictionary containing accuracy, precision, recall, F1 scores, and excluded fraction.
        """
        if self.prediction_postfix is None or self.confidence_postfix is None:
            raise ValueError("Label transfer has not been performed. Call transfer_labels() first.")

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
            raise ValueError("Label transfer has not been performed. Call transfer_labels() first.")

        # Extract true and predicted labels
        y_true = self.query.obs[label_key].dropna()
        y_pred = self.query.obs.loc[y_true.index, f"{label_key}_pred"]

        # Plot confusion matrix using sklearn's ConfusionMatrixDisplay
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=cmap, xticks_rotation="vertical", ax=ax, **kwargs)
        plt.title("Confusion Matrix")

        if save:
            plt.savefig(save, bbox_inches="tight")

    def evaluate_expression_transfer(
        self,
        layer_key: str = "X",
        method: Literal["pearson", "spearman", "js", "rmse"] = "pearson",
    ) -> None:
        """
        Evaluate the agreement between imputed and original expression in the query dataset.

        These metrics are inspired by Li et al., Nature Methods 2022 (https://www.nature.com/articles/s41592-022-01480-9).

        Parameters
        ----------
        layer_key
            Key in `self.query.layers` to use as the original expression. Use "X" to use `self.query.X`.
        method
            Method to quantify agreement. Currently supported: "pearson" (average Pearson correlation across shared genes).

        Returns
        -------
        Nothing, but updates the following attributes:
        expression_transfer_metrics
            Dictionary containing the average correlation and number of genes used for the evaluation.
        """
        imputed_x, original_x, shared_genes = self._get_aligned_expression_arrays(layer_key)
        log_msg = "Expression transfer evaluation (%s): average value = %.4f (n_genes=%d, n_valid_genes=%d)"
        if method == "pearson" or method == "spearman":
            if method == "pearson":
                corr_func = pearsonr
            elif method == "spearman":
                corr_func = spearmanr

            # Compute per-gene correlation
            corrs = np.array([corr_func(original_x[:, i], imputed_x[:, i])[0] for i in range(imputed_x.shape[1])])

            # Store per-gene correlation in query_imputed.var, only for shared genes
            self._store_expression_metric(
                shared_genes,
                corrs,
                f"metric_{method}",
                "average_correlation",
                method,
                log_msg,
            )
        elif method in ("js", "jensen-shannon"):
            jsds = np.array(
                [_jensen_shannon_divergence(imputed_x[:, i], original_x[:, i]) for i in range(imputed_x.shape[1])]
            )
            self._store_expression_metric(
                shared_genes,
                jsds,
                "metric_js",
                "average_jsd",
                "js",
                log_msg,
            )
        elif method == "rmse":
            rmses = np.array([_rmse_zscore(imputed_x[:, i], original_x[:, i]) for i in range(imputed_x.shape[1])])
            self._store_expression_metric(
                shared_genes,
                rmses,
                "metric_rmse",
                "average_rmse",
                "rmse",
                log_msg,
            )
        else:
            raise NotImplementedError(f"Method '{method}' is not implemented.")

    def _get_aligned_expression_arrays(self, layer_key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Extract and align imputed and original expression arrays for shared genes between query_imputed and query.

        Parameters
        ----------
        layer_key : Key in self.query.layers to use as the original expression. Use "X" to use self.query.X.

        Returns
        -------
        imputed_x, original_x, shared_genes
        """
        if self.query_imputed is None:
            raise ValueError("Imputed query data not found. Run transfer_expression() first.")
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
        metric_name: str,
        summary_key: str,
        method_label: str,
        log_msg: str,
    ) -> None:
        """
        Store per-gene and summary expression transfer metrics in the query_imputed AnnData object and log the results.

        Parameters
        ----------
        shared_genes
            List of shared gene names.
        values
            Array of per-gene metric values (e.g., correlation, JSD).
        metric_name
            Name of the column to store per-gene values in query_imputed.var.
        summary_key
            Key for the average metric in self.expression_transfer_metrics.
        method_label
            Name of the method/metric (for logging and summary dict).
        log_msg
            Logging message format string, should accept (avg_value, n_genes, n_valid_genes).

        Returns
        -------
        Nothing, but updates the following attributes:
        query_imputed.var[metric_name]
            DataFrame column with per-gene metric values.
        expression_transfer_metrics
            Dictionary containing the average metric value and number of genes used for the evaluation.
        """
        self.query_imputed.var[metric_name] = None
        self.query_imputed.var.loc[shared_genes, metric_name] = values
        valid_values = values[~np.isnan(values)]
        if valid_values.size == 0:
            raise ValueError(f"No valid genes for {method_label} calculation.")
        avg_value = float(np.mean(valid_values))
        self.expression_transfer_metrics = {
            "method": method_label,
            summary_key: avg_value,
            "n_genes": len(shared_genes),
            "n_valid_genes": int(valid_values.size),
        }
        logger.info(
            log_msg,
            method_label,
            avg_value,
            len(shared_genes),
            int(valid_values.size),
        )
