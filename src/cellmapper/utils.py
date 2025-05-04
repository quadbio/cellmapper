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
    reference_adata: AnnData,
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
        - A numpy array with shape (n_query_cells, n_reference_genes)
        - A sparse matrix with shape (n_query_cells, n_reference_genes)
        - A pandas DataFrame with shape (n_query_cells, n_reference_genes)
        - An existing AnnData object with n_obs matching query_adata
    query_adata
        Query AnnData object providing observation metadata (obs, obsm).
    reference_adata
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
    - var: Reference to reference_adata.var (not copied)
    - obsm: Reference to query_adata.obsm (not copied)
    - varm: Reference to reference_adata.varm if available (not copied)
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

        if len(expression_data.columns) != reference_adata.n_vars:
            raise ValueError(
                f"DataFrame has {len(expression_data.columns)} columns, but reference has {reference_adata.n_vars} features. "
                "They must match."
            )

        # Convert to numpy array for consistency
        expression_data = expression_data.values

    # Case 3: Handle numpy arrays and sparse matrices
    # Validate shape
    expected_shape = (query_adata.n_obs, reference_adata.n_vars)
    if expression_data.shape != expected_shape:
        raise ValueError(
            f"Expression data shape mismatch: expected {expected_shape}, but got {expression_data.shape}. "
            f"Should be (n_query_cells, n_reference_genes)."
        )

    # Create a new AnnData object without copying data where possible
    imputed_adata = ad.AnnData(
        X=expression_data,
        obs=query_adata.obs,  # No copy - direct reference
        var=reference_adata.var,  # No copy - direct reference
        uns=query_adata.uns.copy(),  # Deep copy since uns can contain complex objects
        obsm=query_adata.obsm,  # No copy - direct reference
        varm=reference_adata.varm
        if hasattr(reference_adata, "varm") and reference_adata.varm is not None
        else None,  # No copy
    )

    logger.info(
        "Imputed expression matrix with shape %s converted to AnnData object.\n"
        "Observation metadata from query and feature metadata from reference were linked (not copied).",
        expression_data.shape,
    )

    return imputed_adata


def extract_neighbors_from_distances(
    distances_matrix: "csr_matrix", include_self: bool | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract neighbor indices and distances from a sparse distance matrix.

    Parameters
    ----------
    distances_matrix
        Sparse matrix of distances, typically from adata.obsp['distances']
    include_self
        If True, include self as a neighbor (even if not present in the distance matrix).
        If False, exclude self connections (even if present in the distance matrix).
        If None (default), preserve the original behavior of the distance matrix.

    Returns
    -------
    tuple
        (indices, distances) in the format expected by NeighborsResults
    """
    # Check that the input is a sparse matrix
    if not issparse(distances_matrix):
        raise TypeError("Distances matrix must be a sparse matrix")

    # Verify that the matrix is square
    if distances_matrix.shape[0] != distances_matrix.shape[1]:
        raise ValueError(f"Square distance matrix required (got {distances_matrix.shape})")

    n_cells = distances_matrix.shape[0]

    # Ensure the matrix is CSR format for efficient row-based operations
    distances_matrix = distances_matrix.tocsr()

    # First pass: determine the max number of neighbors after including/excluding self
    max_n_neighbors = 0
    for i in range(n_cells):
        start, end = distances_matrix.indptr[i], distances_matrix.indptr[i + 1]
        cell_indices = distances_matrix.indices[start:end]

        # Calculate how many neighbors this cell will have after applying include_self
        n_neighbors = len(cell_indices)
        if include_self is True and i not in cell_indices:
            n_neighbors += 1  # Will add self
        elif include_self is False and i in cell_indices:
            n_neighbors -= 1  # Will remove self

        max_n_neighbors = max(max_n_neighbors, n_neighbors)

    # Pre-allocate arrays for indices and distances with the correct size
    indices = np.full((n_cells, max_n_neighbors), -1, dtype=np.int64)
    distances = np.full((n_cells, max_n_neighbors), np.inf, dtype=np.float64)

    # Second pass: extract and process neighbor data
    for i in range(n_cells):
        # Get start and end indices for this cell in the sparse matrix
        start, end = distances_matrix.indptr[i], distances_matrix.indptr[i + 1]

        # Get neighbor indices and distances
        cell_indices = distances_matrix.indices[start:end]
        cell_distances = distances_matrix.data[start:end]

        # Filter self-connection if requested and present
        if include_self is False and i in cell_indices:
            # Find the index of self in the neighbors
            self_idx = np.where(cell_indices == i)[0]
            if len(self_idx) > 0:
                # Remove self from indices and distances
                mask = cell_indices != i
                cell_indices = cell_indices[mask]
                cell_distances = cell_distances[mask]
        # If include_self is True and self is not in the neighbors, add it
        elif include_self is True and i not in cell_indices:
            # Add self with distance 0
            cell_indices = np.append(cell_indices, i)
            cell_distances = np.append(cell_distances, 0.0)

        # Number of neighbors after potential filtering
        n_neighbors = len(cell_indices)

        if n_neighbors > 0:
            # Sort by distance if they aren't already sorted
            if not np.all(np.diff(cell_distances) >= 0):
                sort_idx = np.argsort(cell_distances)
                cell_indices = cell_indices[sort_idx]
                cell_distances = cell_distances[sort_idx]

            # Fill arrays
            indices[i, :n_neighbors] = cell_indices
            distances[i, :n_neighbors] = cell_distances

    return indices, distances
