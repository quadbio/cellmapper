"""Utility functions for the CellMapper package."""

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator, svds

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


def truncated_svd_cross_covariance(
    X: np.ndarray | csr_matrix,
    Y: np.ndarray | csr_matrix,
    n_components: int = 50,
    zero_center: bool = True,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute truncated SVD of the cross-covariance matrix X @ Y.T efficiently.

    This implementation avoids materializing the potentially large dense matrix X @ Y.T
    by using an implicit matrix-vector product approach, similar to scanpy's PCA
    implementation for sparse matrices. Mean-centering is also performed implicitly
    to preserve sparsity.

    Parameters
    ----------
    X
        First data matrix of shape (n_obs_x, n_vars). Rows correspond to
        cells/observations and columns to genes/variables.
    Y
        Second data matrix of shape (n_obs_y, n_vars). Must have the same number of
        variables as X.
    n_components
        Number of singular vectors to compute. Defaults to 50.
    zero_center
        If True (default), implicitly center the data before computing SVD.
        For sparse matrices, centering is done implicitly without densifying the matrix.
        For dense matrices, centering is done explicitly.
    random_state
        Random seed for reproducibility.

    Returns
    -------
    U
        Left singular vectors of shape (n_obs_x, n_components).
    s
        Singular values of shape (n_components,).
    Vt
        Right singular vectors of shape (n_components, n_obs_y).

    Notes
    -----
    This function parallels scanpy's approach to handling sparse matrices in PCA
    computation but is adapted for the cross-covariance case (X @ Y.T instead of X.T @ X).
    Both X and Y must be of the same type (both sparse or both dense).
    """
    # Check inputs
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"X and Y must have the same number of variables: X has {X.shape[1]}, Y has {Y.shape[1]}")

    # Check that X and Y are both sparse or both dense
    x_is_sparse = issparse(X)
    y_is_sparse = issparse(Y)

    if x_is_sparse != y_is_sparse:
        raise TypeError("X and Y must be of the same type: both sparse or both dense")

    # Ensure sparse matrices are in CSR format for efficiency
    if x_is_sparse:
        if not isinstance(X, csr_matrix):
            X = X.tocsr()
        if not isinstance(Y, csr_matrix):
            Y = Y.tocsr()

    # Compute means for implicit centering if needed
    if zero_center:
        # Shape: (n_vars,)
        if x_is_sparse:
            X_mean = np.asarray(X.mean(axis=0)).flatten()
            Y_mean = np.asarray(Y.mean(axis=0)).flatten()
        else:
            X_mean = np.mean(X, axis=0)
            Y_mean = np.mean(Y, axis=0)

    # For dense matrices with zero_center, explicitly center them
    if zero_center and not x_is_sparse:
        # X shape: (n_obs_x, n_vars)
        # X_mean shape: (n_vars,)
        X = X - X_mean
        Y = Y - Y_mean

        # Define a simple matrix multiplication for the operator
        def matvec(v):
            return X @ (Y.T @ v)

        def rmatvec(v):
            return Y @ (X.T @ v)

        matmat = matvec
        rmatmat = rmatvec

    # For sparse matrices with zero_center, use implicit centering
    elif zero_center and x_is_sparse:
        corr_1 = X @ Y_mean  # Shape: (n_obs_x,)
        corr_2 = Y @ X_mean  # Shape: (n_obs_y,)
        corr_3 = X_mean.T @ Y_mean  # Scalar value

        # Define matrix-vector multiplication operations with implicit centering
        def matvec(v):
            """Compute (X_c @ Y_c.T) @ v without materializing the full matrix."""
            # v shape: (n_obs_y,)
            v_sum = np.sum(v)
            return X @ (Y.T @ v) - corr_1 * v_sum - corr_2.T @ v + corr_3 * v_sum

        def rmatvec(v):
            """Compute (X_c @ Y_c.T).T @ v = Y_c @ X_c.T @ v without materializing the full matrix."""
            # v shape: (n_obs_x,)
            v_sum = np.sum(v)
            return Y @ (X.T @ v) - corr_1.T @ v - corr_2 * v_sum + corr_3 * v_sum

        # Define matrix-matrix multiplication operations for when v is a matrix
        def matmat(V):
            """Compute (X_c @ Y_c.T) @ V without materializing the full matrix."""
            # V shape: (n_obs_y, n_cols)

            # For each column, calculate the sum and apply corrections
            V_sum = V.sum(axis=0)  # Sum of each column
            return X @ (Y.T @ V) - np.outer(corr_1, V_sum) - corr_2.T @ V + corr_3 * V_sum

        def rmatmat(V):
            """Compute (X_c @ Y_c.T).T @ V without materializing the full matrix."""
            # V shape: (n_obs_x, n_cols)

            # For each column, calculate the sum and apply corrections
            V_sum = V.sum(axis=0)  # Sum of each column
            return Y @ (X.T @ V) - corr_1.T @ V - np.outer(corr_2, V_sum) + corr_3 * V_sum

    # For the case with no centering, direct matrix multiplication
    else:

        def matvec(v):
            return X @ (Y.T @ v)

        def rmatvec(v):
            return Y @ (X.T @ v)

        matmat = matvec
        rmatmat = rmatvec

    # Create LinearOperator representing the cross-covariance matrix without materializing it
    XYt_op = LinearOperator(
        shape=(X.shape[0], Y.shape[0]), matvec=matvec, rmatvec=rmatvec, matmat=matmat, rmatmat=rmatmat, dtype=np.float64
    )

    # Set random seed
    np.random.seed(random_state)
    random_init = np.random.rand(min(XYt_op.shape))

    # Compute truncated SVD using ARPACK
    u, s, vt = svds(XYt_op, k=n_components, v0=random_init)

    # Sort singular values in descending order
    idx = np.argsort(-s)
    u = u[:, idx]
    s = s[idx]
    vt = vt[idx, :]

    return u, s, vt
