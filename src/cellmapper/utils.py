"""Utility functions for the CellMapper package."""

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from cellmapper.logging import logger


def create_imputed_anndata(
    expression_data: np.ndarray | csr_matrix | pd.DataFrame | AnnData,
    query_adata: AnnData,
    ref_adata: AnnData,
) -> AnnData:
    """
    Create an AnnData object for imputed expression data with validation and conversion.

    This function handles validation, conversion, and construction of an AnnData object
    from imputed expression data, taking observation metadata from the query and
    feature metadata from the reference.

    Parameters
    ----------
    expression_data
        The imputed expression data in one of these formats:
        - A numpy array with shape (n_query_cells, n_ref_genes)
        - A sparse matrix with shape (n_query_cells, n_ref_genes)
        - A pandas DataFrame with shape (n_query_cells, n_ref_genes)
        - An existing AnnData object with n_obs matching query_adata
    query_adata
        Query AnnData object providing observation metadata (obs, obsm).
    ref_adata
        Reference AnnData object providing feature metadata (var, varm).

    Returns
    -------
    AnnData
        An AnnData object containing the imputed expression with aligned metadata.

    Notes
    -----
    The returned AnnData object will have:
    - X: The imputed expression data
    - obs: Reference to query_adata.obs (not copied)
    - var: Reference to ref_adata.var (not copied)
    - obsm: Reference to query_adata.obsm (not copied)
    - varm: Reference to ref_adata.varm if available (not copied)
    - uns: Deep copy from query_adata.uns since it can contain complex objects
    """
    # Check for unsupported types first
    if not isinstance(expression_data, np.ndarray | csr_matrix | pd.DataFrame | AnnData):
        raise TypeError(
            f"Unsupported type for expression_data: {type(expression_data)}. "
            "Must be AnnData, numpy array, sparse matrix, or pandas DataFrame."
        )

    # Case 1: Handle existing AnnData object
    if isinstance(expression_data, AnnData):
        # Validate that the imputed data has the same number of observations as the query
        if expression_data.n_obs != query_adata.n_obs:
            raise ValueError(
                f"Imputed AnnData has {expression_data.n_obs} observations, but query has {query_adata.n_obs} observations. "
                "They must have the same number of observations."
            )

        # Check if the observations are aligned (same order)
        if not expression_data.obs_names.equals(query_adata.obs_names):
            logger.warning(
                "Observation names in imputed AnnData don't match query observation names. "
                "Make sure the cells are aligned correctly."
            )

        logger.info("Using existing AnnData object with %d genes as imputed data.", expression_data.n_vars)
        return expression_data

    # Case 2: Handle DataFrame
    if isinstance(expression_data, pd.DataFrame):
        # Check if DataFrame dimensions match expected dimensions
        if len(expression_data.index) != query_adata.n_obs:
            raise ValueError(
                f"DataFrame has {len(expression_data.index)} rows, but query has {query_adata.n_obs} observations. "
                "They must match."
            )

        if len(expression_data.columns) != ref_adata.n_vars:
            raise ValueError(
                f"DataFrame has {len(expression_data.columns)} columns, but reference has {ref_adata.n_vars} features. "
                "They must match."
            )

        # Convert to numpy array for consistency
        expression_data = expression_data.values

    # Case 3: Handle numpy arrays and sparse matrices
    # Validate shape
    expected_shape = (query_adata.n_obs, ref_adata.n_vars)
    if expression_data.shape != expected_shape:
        raise ValueError(
            f"Expression data shape mismatch: expected {expected_shape}, but got {expression_data.shape}. "
            f"Should be (n_query_cells, n_ref_genes)."
        )

    # Create a new AnnData object without copying data where possible
    imputed_adata = ad.AnnData(
        X=expression_data,
        obs=query_adata.obs,  # No copy - direct reference
        var=ref_adata.var,  # No copy - direct reference
        uns=query_adata.uns.copy(),  # Deep copy since uns can contain complex objects
        obsm=query_adata.obsm,  # No copy - direct reference
        varm=ref_adata.varm if hasattr(ref_adata, "varm") and ref_adata.varm is not None else None,  # No copy
    )

    logger.info(
        "Imputed expression matrix with shape %s converted to AnnData object.\n"
        "Observation metadata from query and feature metadata from reference were linked (not copied).",
        expression_data.shape,
    )

    return imputed_adata


def extract_neighbors_from_distances(distances_matrix: "csr_matrix") -> tuple[np.ndarray, np.ndarray]:
    """
    Extract neighbor indices and distances from a sparse distance matrix.

    Parameters
    ----------
    distances_matrix
        Sparse matrix of distances, typically from adata.obsp['distances']

    Returns
    -------
    tuple
        (indices, distances) in the format expected by NeighborsResults
    """
    # Check that the input is a sparse matrix
    if not issparse(distances_matrix):
        raise TypeError("Distances matrix must be sparse")

    n_cells = distances_matrix.shape[0]

    # Get the maximum number of neighbors per cell
    n_neighbors_per_cell = np.diff(distances_matrix.indptr)
    max_n_neighbors = n_neighbors_per_cell.max()

    # Pre-allocate arrays for indices and distances
    indices = np.zeros((n_cells, max_n_neighbors), dtype=np.int64)
    distances = np.ones((n_cells, max_n_neighbors), dtype=np.float64) * np.inf

    # Extract indices and distances for each cell
    for i in range(n_cells):
        # Get start and end indices for this cell in the sparse matrix
        start, end = distances_matrix.indptr[i], distances_matrix.indptr[i + 1]

        # Number of neighbors for this cell
        n_neighbors = end - start

        if n_neighbors > 0:
            # Get neighbor indices
            cell_indices = distances_matrix.indices[start:end]

            # Get distances
            cell_distances = distances_matrix.data[start:end]

            # Sort by distance if they aren't already sorted
            if not np.all(np.diff(cell_distances) >= 0):
                sort_idx = np.argsort(cell_distances)
                cell_indices = cell_indices[sort_idx]
                cell_distances = cell_distances[sort_idx]

            # Fill arrays
            indices[i, :n_neighbors] = cell_indices
            distances[i, :n_neighbors] = cell_distances

    return indices, distances
