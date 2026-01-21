from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from cellmapper._docs import d
from cellmapper.logging import logger

if TYPE_CHECKING:
    from anndata import AnnData


def _get_category_colors(
    adata: "AnnData | None",
    label_key: str,
    categories: list[str],
) -> list[str]:
    """Get colors for categories from adata.uns, falling back to gray.

    Parameters
    ----------
    adata
        AnnData object to get colors from (must be AnnData, not a list).
    label_key
        Key in .obs storing the categorical annotation.
    categories
        List of category names to get colors for.

    Returns
    -------
    List of colors corresponding to each category.

    Raises
    ------
    TypeError
        If adata is not None and not an AnnData object.
    """
    if adata is not None and not hasattr(adata, "uns"):
        msg = f"Expected AnnData object, got {type(adata).__name__}."
        raise TypeError(msg)

    colors_key = f"{label_key}_colors"
    colors_dict: dict[str, str] = {}

    if adata is not None and colors_key in adata.uns:
        full_categories = adata.obs[label_key].cat.categories
        full_colors = adata.uns[colors_key]
        for i, cat in enumerate(full_categories):
            if i < len(full_colors):
                colors_dict[str(cat)] = full_colors[i]

    return [colors_dict.get(str(cat), "gray") for cat in categories]


def _get_text_color(background_color: str | tuple, threshold: float = 0.5) -> str:
    """Get contrasting text color (black or white) for a background color.

    Parameters
    ----------
    background_color
        Background color as hex string, named color, or RGB tuple.
    threshold
        Luminance threshold for switching between black and white.

    Returns
    -------
    "black" or "white" depending on background luminance.
    """
    import matplotlib.colors as mcolors

    try:
        rgb = mcolors.to_rgb(background_color)
        # Perceived luminance formula
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return "white" if luminance < threshold else "black"
    except ValueError:
        return "black"


def _draw_annotation_strips(
    ax: plt.Axes,
    row_colors: list[str],
    col_colors: list[str],
    xlabel_position: Literal["bottom", "top"] = "bottom",
    strip_frac: float = 0.02,
) -> None:
    """Draw colored annotation strips along heatmap axes.

    Parameters
    ----------
    ax
        Matplotlib axes containing the heatmap.
    row_colors
        Colors for each row (y-axis).
    col_colors
        Colors for each column (x-axis).
    xlabel_position
        Position of x-axis labels ("bottom" or "top").
    strip_frac
        Width of strips as fraction of axes size.
    """
    from matplotlib.patches import Rectangle

    # Pad tick labels to make room for strips
    ax.tick_params(axis="y", pad=15)
    ax.tick_params(axis="x", pad=15)

    # Use blended transforms for strips
    trans_left = ax.get_yaxis_transform()  # x=axes, y=data
    trans_x = ax.get_xaxis_transform()  # x=data, y=axes

    # Draw row strips (left side)
    for i, color in enumerate(row_colors):
        rect = Rectangle(
            (-strip_frac, i - 0.5),
            strip_frac,
            1,
            facecolor=color,
            edgecolor="none",
            clip_on=False,
            transform=trans_left,
        )
        ax.add_patch(rect)

    # Draw column strips (top or bottom)
    y_start = 1 if xlabel_position == "top" else -strip_frac
    for i, color in enumerate(col_colors):
        rect = Rectangle(
            (i - 0.5, y_start),
            1,
            strip_frac,
            facecolor=color,
            edgecolor="none",
            clip_on=False,
            transform=trans_x,
        )
        ax.add_patch(rect)


def _annotate_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str = ".2f",
    fontsize: float = 8,
) -> None:
    """Add value annotations to heatmap cells with contrast-aware text colors.

    Parameters
    ----------
    ax
        Matplotlib axes containing the heatmap.
    data
        2D array of values to annotate.
    cmap
        Colormap name used for the heatmap.
    vmin
        Minimum value for normalization.
    vmax
        Maximum value for normalization.
    fmt
        Format string for values.
    fontsize
        Font size for annotations.
    """
    import matplotlib.colors as mcolors

    colormap = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            # Get background color and choose contrasting text
            bg_color = colormap(norm(val))
            text_color = _get_text_color(bg_color)
            ax.text(
                j,
                i,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
            )


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
        self, label_key: str, prediction_postfix: str = "_pred", confidence_postfix: str = "_conf"
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
        pred_col = f"{label_key}{prediction_postfix}"
        conf_col = f"{label_key}{confidence_postfix}"

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
        y_pred = self.query.obs.loc[y_true.index, f"{label_key}{pred_postfix}"]
        confidence = self.query.obs.loc[y_true.index, f"{label_key}{conf_postfix}"]

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
        self,
        label_key: str,
        *,
        true_key: str | None = None,
        subset: np.ndarray | pd.Series | None = None,
        figsize: tuple[int, int] = (10, 8),
        cmap: str = "viridis",
        save: str | Path | None = None,
        ax: plt.Axes | None = None,
        show_annotation_colors: bool = True,
        xlabel_position: Literal["bottom", "top"] = "bottom",
        show_grid: bool = True,
        min_cells_true: int | None = None,
        min_cells_pred: int | None = None,
        show_yticklabels: bool = True,
        show_xticklabels: bool = True,
        normalize: Literal["true", "pred", "all"] | None = None,
        include_values: bool = True,
        values_format: str = ".2f",
        values_fontsize: float = 8,
        colorbar: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = "Confusion Matrix",
    ) -> plt.Axes:
        """Plot a confusion matrix heatmap comparing true vs predicted labels.

        Parameters
        ----------
        label_key
            Key in .obs storing predicted labels (from map_obs). The column
            ``f"{label_key}{prediction_postfix}"`` is used as the x-axis (predicted).
        true_key
            Key in .obs to use for the y-axis (true labels). If None, uses ``label_key``.
            This allows comparing arbitrary columns, e.g., source_time vs mapped_time.
        subset
            Boolean mask to select a subset of cells for the confusion matrix.
            Must have the same length as query.obs or be a pandas Series indexed by obs_names.
        figsize
            Size of the figure (width, height). Only used if ax is None.
        cmap
            Colormap to use for the heatmap.
        save
            Path to save the figure. If None, the figure is not saved.
        ax
            Matplotlib axes to plot on. If None, a new figure and axes are created.
        show_annotation_colors
            Whether to show colored bars along axes corresponding to category colors
            from ``adata.uns[f"{label_key}_colors"]``.
        xlabel_position
            Position of x-axis tick labels ("bottom" or "top").
        show_grid
            Whether to show gridlines on the heatmap.
        min_cells_true
            Minimum number of cells required for a true category to be included.
            If None, all true categories are shown.
        min_cells_pred
            Minimum number of cells required for a predicted category to be included.
            If None, all predicted categories are shown.
        show_yticklabels
            Whether to show y-axis tick labels.
        show_xticklabels
            Whether to show x-axis tick labels.
        normalize
            Normalization mode: "true" (row), "pred" (column), "all" (total), or None.
        include_values
            Whether to annotate cells with their values.
        values_format
            Format string for cell values (e.g., ".2f", ".0f", ".1%").
        values_fontsize
            Font size for cell value annotations.
        colorbar
            Whether to show a colorbar.
        vmin
            Minimum value for colormap normalization.
        vmax
            Maximum value for colormap normalization.
        title
            Title for the plot. Set to None to hide.

        Returns
        -------
        Matplotlib axes with the confusion matrix plot.
        """
        if self.prediction_postfix is None or self.confidence_postfix is None:
            raise ValueError("Label transfer has not been performed. Call map_obs() first.")

        # Extract true and predicted labels
        true_col = true_key if true_key is not None else label_key
        y_true = self.query.obs[true_col].copy()
        y_pred = self.query.obs[f"{label_key}{self.prediction_postfix}"].copy()

        # Drop NaNs
        valid_mask = y_true.notna() & y_pred.notna()
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        # Apply subset filter
        if subset is not None:
            if isinstance(subset, pd.Series):
                subset = subset.loc[y_true.index]
            else:
                subset = pd.Series(subset, index=self.query.obs_names).loc[y_true.index]
            y_true = y_true[subset]
            y_pred = y_pred[subset]

        # Convert to string for consistent handling
        y_true = y_true.astype(str)
        y_pred = y_pred.astype(str)

        # Create confusion matrix as DataFrame
        cm = pd.crosstab(y_true, y_pred, dropna=False)
        cm.index.name = "True"
        cm.columns.name = "Predicted"

        # Filter rows (true categories) by min_cells
        if min_cells_true is not None:
            row_counts = cm.sum(axis=1)
            cm = cm.loc[row_counts >= min_cells_true]

        # Filter columns (predicted categories) by min_cells
        if min_cells_pred is not None:
            col_counts = cm.sum(axis=0)
            cm = cm.loc[:, col_counts >= min_cells_pred]

        # Sort both axes alphabetically
        cm = cm.sort_index(axis=0).sort_index(axis=1)

        # Normalize if requested
        cm_display = cm.copy().astype(float)
        if normalize == "true":
            cm_display = cm_display.div(cm_display.sum(axis=1), axis=0)
        elif normalize == "pred":
            cm_display = cm_display.div(cm_display.sum(axis=0), axis=1)
        elif normalize == "all":
            cm_display = cm_display / cm_display.values.sum()

        # Handle NaN from division by zero
        cm_display = cm_display.fillna(0)

        # Set vmin/vmax defaults
        if vmin is None:
            vmin = 0 if normalize else cm_display.values.min()
        if vmax is None:
            vmax = 1 if normalize else cm_display.values.max()

        # Create figure/axes if not provided
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot heatmap
        im = ax.imshow(cm_display.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        # Set ticks
        ax.set_xticks(np.arange(len(cm_display.columns)))
        ax.set_yticks(np.arange(len(cm_display.index)))

        # Set tick labels
        if show_xticklabels:
            ax.set_xticklabels(cm_display.columns, rotation=90)
        else:
            ax.set_xticklabels([])

        if show_yticklabels:
            ax.set_yticklabels(cm_display.index)
        else:
            ax.set_yticklabels([])

        # Position x-axis labels
        if xlabel_position == "top":
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            if show_xticklabels:
                plt.setp(ax.get_xticklabels(), ha="center", va="bottom")

        # Axis labels
        if show_yticklabels:
            ax.set_ylabel("True label")
        if show_xticklabels:
            ax.set_xlabel("Predicted label")

        # Title
        if title:
            if xlabel_position == "top":
                ax.set_title(title, pad=20)
            else:
                ax.set_title(title)

        # Grid
        if not show_grid:
            ax.grid(False)
        else:
            ax.set_xticks(np.arange(-0.5, len(cm_display.columns), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(cm_display.index), 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", size=0)

        # Annotate values
        if include_values:
            _annotate_heatmap(
                ax,
                cm_display.values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                fmt=values_format,
                fontsize=values_fontsize,
            )

        # Colorbar
        if colorbar:
            ax.figure.colorbar(im, ax=ax, shrink=0.6)

        # Annotation color strips
        if show_annotation_colors:
            # Row colors (true labels) from query, column colors (predicted) from reference
            row_colors = _get_category_colors(self.query, label_key, list(cm_display.index))
            col_colors = _get_category_colors(self.reference, label_key, list(cm_display.columns))
            _draw_annotation_strips(ax, row_colors, col_colors, xlabel_position)

        if save:
            ax.figure.savefig(save, bbox_inches="tight")

        return ax

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
        minmax: bool = True,
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
        minmax
            Whether to apply min-max normalization to the scores.
        """
        if self.knn is None or self.knn.yx is None:
            raise ValueError("Neighbors must be computed before estimating presence scores.")

        conn = self.knn.yx.knn_graph_connectivities()
        reference_names = self.reference.obs_names

        # Always compute and post-process the overall score (all query cells)
        scores_all = np.array(conn.sum(axis=0)).flatten()
        df_all = pd.DataFrame({"all": scores_all}, index=reference_names)
        df_all_processed = self.process_presence_scores(df_all, log=log, percentile=percentile, minmax=minmax)
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
            df_groups_processed = self.process_presence_scores(df_groups, log=log, percentile=percentile, minmax=minmax)
            self.reference.obsm[key_added] = df_groups_processed

            logger.info(
                "Presence scores per group defined in `query.obs['%s']` computed and stored in `reference.obsm['%s']`",
                groupby,
                key_added,
            )

    @staticmethod
    def process_presence_scores(
        scores: pd.DataFrame,
        log: bool = False,
        percentile: tuple[float, float] = (1, 99),
        minmax: bool = True,
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
        minmax
            Whether to apply min-max normalization to the scores.

        Returns
        -------
        pd.DataFrame
            Post-processed presence scores, same shape as input.
        """
        # Log1p transformation
        if log:
            scores = np.log1p(scores)

        # Percentile clipping
        if percentile != (0, 100):
            low, high = percentile
            scores = scores.apply(lambda x: np.clip(x, np.percentile(x, low), np.percentile(x, high)), axis=0)

        # Min-max normalization
        if minmax:

            def minmax_norm(x):
                min_val, max_val = np.min(x), np.max(x)
                return (x - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(x)

            scores = scores.apply(minmax_norm, axis=0)

        return scores
