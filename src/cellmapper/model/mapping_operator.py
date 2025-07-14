"""Mapping operator for applying powers of mapping matrices."""

from functools import cached_property
from typing import Literal

import numpy as np
from scipy.linalg import eig
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse
from scipy.sparse.linalg import eigs

from cellmapper.constants import PackageConstants
from cellmapper.logging import logger


class MappingOperator:
    """
    Operator for applying powers of mapping matrices with validation and normalization.

    This class provides two methods for computing matrix powers M^t:

    1. **Iterative method**: Computes M^t by repeated matrix multiplication (M @ M @ ... @ M).
       This is exact but can be slow for large t.

    2. **Spectral method**: Approximates M^t using eigendecomposition (M = V @ Λ @ V^(-1),
       so M^t ≈ V @ Λ^t @ V^(-1)). This can be much faster for large t but is approximate.

    **Approximation Quality Trade-offs:**

    The spectral method approximates the full matrix using only the largest eigenvalues/eigenvectors.
    The quality of this approximation depends on:

    - **More eigenvectors**: Better approximation of the full matrix, higher accuracy
    - **Fewer eigenvectors**: Faster computation, lower memory usage, but less accurate
    - **Larger t**: Approximation becomes more accurate because smaller eigenvalues (excluded
      from the approximation) decay exponentially as λ^t, making their contribution negligible
    - **Smaller t**: Excluded eigenvalues still contribute significantly, making approximation less accurate

    **Recommendations:**

    - Use `t=None` for single-step mapping (fastest, exact)
    - Use `method="iterative"` for small t (2-10 steps, exact but manageable cost)
    - Use `method="spectral"` for large t (>10 steps, approximate but much faster AND relatively more accurate)
    - Increase `n_eigenvectors` if spectral approximation quality is insufficient for your t values
    """

    def __init__(
        self,
        kernel_matrix: csr_matrix | coo_matrix | csc_matrix | np.ndarray,
        is_self_mapping: bool,
        expected_shape: tuple[int, int],
        n_eigenvectors: int = 50,
        eigen_solver: Literal["partial", "complete"] = "partial",
    ):
        """
        Initialize mapping operator with automatic validation and normalization.

        Parameters
        ----------
        kernel_matrix
            The unnormalized kernel matrix to validate and normalize
        is_self_mapping
            Whether this is self-mapping (square matrix) or cross-mapping
        expected_shape
            Expected shape (n_query_cells, n_reference_cells)
        n_eigenvectors
            Number of eigenvectors to compute for spectral decomposition.
            More eigenvectors = better approximation but slower computation.
            Automatically capped to ensure numerical stability.
        eigen_solver
            Eigendecomposition method for spectral approach:
            - "partial": Uses sparse eigendecomposition (scipy.sparse.linalg.eigs), faster
            - "complete": Uses complete eigendecomposition (scipy.linalg.eig), exact for testing
        """
        self.is_self_mapping = is_self_mapping
        self.expected_shape = expected_shape
        self.eigen_solver = eigen_solver

        # Ensure we don't compute too many eigenvectors for small matrices
        if eigen_solver == "complete":
            # For complete eigendecomposition, use all eigenvectors
            self.n_eigenvectors = expected_shape[0]
        elif eigen_solver == "partial":
            # For partial eigendecomposition, cap the number of eigenvectors
            max_eigenvectors = max(1, min(expected_shape[0] - 2, n_eigenvectors))
            self.n_eigenvectors = max_eigenvectors
        else:
            raise ValueError(f"Unknown eigen_solver: {eigen_solver}. Use 'partial' or 'complete'.")

        # Store matrix type information (set during validation)
        self.is_sparse: bool
        self.is_symmetric: bool | None

        # Validate and normalize the matrix
        self.mapping_matrix = self._validate_and_normalize_mapping_matrix(kernel_matrix)

    @property
    def matrix(self) -> csr_matrix | np.ndarray:
        """
        Get the underlying mapping matrix.

        Returns
        -------
        csr_matrix or np.ndarray
            The validated and normalized mapping matrix in original format
        """
        return self.mapping_matrix

    def _validate_and_normalize_mapping_matrix(
        self, kernel_matrix: csr_matrix | coo_matrix | csc_matrix | np.ndarray
    ) -> csr_matrix | np.ndarray:
        """
        Validate and normalize the mapping matrix.

        Parameters
        ----------
        kernel_matrix
            The kernel matrix to validate and normalize (sparse or dense)

        Returns
        -------
        Validated and row-normalized mapping matrix in original format
        """
        # Determine if input is sparse or dense
        self.is_sparse = issparse(kernel_matrix)

        # Validate the shape
        if kernel_matrix.shape != self.expected_shape:
            raise ValueError(
                f"Mapping matrix shape mismatch: expected {self.expected_shape}, but got {kernel_matrix.shape}."
            )

        # Validate self-mapping consistency
        n_rows, n_cols = kernel_matrix.shape
        if self.is_self_mapping and n_rows != n_cols:
            raise ValueError(f"Self-mapping requires square matrix, got shape {kernel_matrix.shape}")

        if not self.is_self_mapping and n_rows == n_cols:
            logger.warning(
                "Square matrix detected but is_self_mapping=False. "
                "Consider setting is_self_mapping=True for matrix powers."
            )

        # Check for symmetry before row-normalization (only for self-mapping)
        if self.is_self_mapping:
            if self.is_sparse:
                # Use sparse symmetry check - compute difference and check if all entries are zero
                diff = kernel_matrix - kernel_matrix.T
                self.is_symmetric = np.allclose(diff.data, 0, rtol=1e-10, atol=1e-12)
            else:
                # Dense matrix symmetry check
                dense_array = np.asarray(kernel_matrix)
                self.is_symmetric = np.allclose(dense_array, dense_array.T, rtol=1e-10, atol=1e-12)

            if self.is_symmetric:
                logger.debug("Input matrix is symmetric - will result in reversible Markov chain.")
            else:
                logger.warning(
                    "Input matrix is not symmetric - resulting Markov chain may not be reversible. "
                    "Consider using a symmetric adjacency matrix for better spectral properties."
                )
        else:
            # Non-self-mapping matrices cannot be checked for symmetry
            self.is_symmetric = None
            logger.debug("Non-self-mapping matrix - symmetry check skipped.")

        # Compute row sums (shared logic for sparse and dense)
        if self.is_sparse:
            row_sums = kernel_matrix.sum(axis=1).A1  # Convert to 1D array
        else:
            row_sums = np.asarray(kernel_matrix).sum(axis=1)

        # Check for zero rows and handle them
        if np.any(row_sums == 0):
            logger.warning("Some rows in the mapping matrix have a sum of zero. These rows will be left unchanged.")
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        # (Asymmetric) row-normalization
        if self.is_sparse:
            kernel_matrix = csr_matrix(kernel_matrix).multiply(1 / row_sums[:, None]).astype(np.float32)
        else:
            kernel_matrix = np.asarray(kernel_matrix) / row_sums[:, None].astype(np.float32)

        return kernel_matrix

    def _validate_power(self, t: int) -> None:
        """Validate that the requested power is feasible."""
        if t < 1:
            raise ValueError(f"Power t must be >= 1, got {t}")

        if t > 1 and not self.is_self_mapping:
            raise ValueError(f"Matrix powers t > 1 only supported for self-mapping mode, got t={t}")

    @cached_property
    def _eigendecomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute and cache eigendecomposition for self-mapping matrices.

        Returns
        -------
        eigenvalues, eigenvectors
            Real eigenvalues and corresponding eigenvectors for reversible Markov chain
        """
        if not self.is_self_mapping:
            raise RuntimeError("Eigendecomposition only available for self-mapping mode")

        if self.eigen_solver == "complete":
            logger.info("Computing complete eigendecomposition for matrix powers")
            # Convert to dense for complete eigendecomposition
            if issparse(self.mapping_matrix):
                dense_matrix = self.mapping_matrix.toarray()
            else:
                dense_matrix = np.asarray(self.mapping_matrix)

            eigenvalues, eigenvectors = eig(dense_matrix)  # type: ignore[assignment]

        else:
            logger.info(
                "Computing eigendecomposition with %d components for matrix powers",
                self.n_eigenvectors,
            )
            # Use partial eigendecomposition (original implementation)
            eigenvalues, eigenvectors = eigs(  # type: ignore[misc]
                self.mapping_matrix,
                k=self.n_eigenvectors,
                which="LM",  # Largest magnitude
                return_eigenvectors=True,
            )

        # Sort by eigenvalue magnitude (descending) for proper diffusion ordering
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Check for complex eigenvalues and fail explicitly
        if np.iscomplexobj(eigenvalues) and not np.allclose(np.imag(eigenvalues), 0):
            raise ValueError(
                "Complex eigenvalues detected. The mapping matrix may not be reversible. "
                "Consider using the 'iterative' method instead."
            )

        if np.iscomplexobj(eigenvectors) and not np.allclose(np.imag(eigenvectors), 0):
            raise ValueError(
                "Complex eigenvectors detected. The mapping matrix may not be reversible. "
                "Consider using the 'iterative' method instead."
            )

        # Convert to real arrays (safe since we checked for complex values)
        eigenvalues_real = np.real(eigenvalues) if np.iscomplexobj(eigenvalues) else eigenvalues
        eigenvectors_real = np.real(eigenvectors) if np.iscomplexobj(eigenvectors) else eigenvectors

        return eigenvalues_real, eigenvectors_real

    def _apply_iterative(self, reference_data, t: int):
        """Apply matrix power using iterative multiplication."""
        logger.debug("Using iterative multiplication for t=%d", t)
        result = reference_data.copy()
        for _ in range(t):
            result = self.mapping_matrix @ result
        return result

    def _apply_spectral(self, reference_data, t: int):
        """Apply matrix power using cached eigendecomposition."""
        logger.debug("Using spectral decomposition for t=%d", t)
        eigenvalues, eigenvectors = self._eigendecomposition

        # Project data onto eigenvector space
        projected = eigenvectors.T @ reference_data

        # Apply eigenvalue powers
        powered = (eigenvalues[:, np.newaxis] ** t) * projected

        # Project back to original space
        result = eigenvectors @ powered

        return result

    def apply(
        self,
        reference_data,  # Allow any array-like data type
        t: int | None = None,
        method: Literal["iterative", "spectral"] = "iterative",
    ):
        """
        Apply mapping matrix power: M^t @ reference_data.

        Parameters
        ----------
        reference_data
            Data to map (reference_cells x features). Can be dense or sparse arrays,
            pandas DataFrames, or any array-like structure.
        t
            Matrix power to apply. If None (default), uses direct multiplication (fastest).
            If t >= 1, allows method selection between iterative and spectral approaches.
        method
            Method for computing matrix powers. Options:
            - "iterative": Iterative matrix multiplication (exact but slow for large t)
            - "spectral": Eigendecomposition-based approximation (faster for large t,
              becomes more accurate as t increases due to exponential decay of small eigenvalues)

        Returns
        -------
        mapped_data
            Result of M^t @ reference_data (query_cells x features).
            Maintains sparsity of input data when possible.

        Notes
        -----
        The spectral method approximates the iterative method using eigendecomposition.
        See the class docstring for detailed trade-offs between accuracy and performance.
        """
        if t is None:
            # Direct multiplication - fastest path
            logger.debug("Using direct matrix multiplication (t=None)")
            return self.mapping_matrix @ reference_data

        # Validate the power (only for non-None values)
        self._validate_power(t)

        # Warn about performance for large matrix powers with iterative method
        if t > PackageConstants.SPECTRAL_METHOD_THRESHOLD and method == "iterative" and self.is_self_mapping:
            logger.warning(
                "Using iterative method for t=%d matrix powers may be slow for large datasets. "
                "Consider using method='spectral' for better performance.",
                t,
            )

        # Apply the chosen method for t >= 1
        if method == "iterative":
            return self._apply_iterative(reference_data, t)
        elif method == "spectral":
            return self._apply_spectral(reference_data, t)
        else:
            raise ValueError(f"Unknown method: {method}")

    def clear_cache(self) -> None:
        """Clear cached eigendecomposition to free memory."""
        if hasattr(self, "_eigendecomposition"):
            delattr(self, "_eigendecomposition")
            logger.debug("Cleared eigendecomposition cache")
