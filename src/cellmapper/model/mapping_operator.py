"""Mapping operator for applying powers of mapping matrices."""

from functools import cached_property
from typing import Literal

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse
from scipy.sparse.linalg import eigs

from cellmapper.logging import logger


class MappingOperator:
    """Operator for applying powers of mapping matrices with validation and normalization."""

    def __init__(
        self,
        mapping_matrix: csr_matrix | coo_matrix | csc_matrix | np.ndarray,
        is_self_mapping: bool,
        expected_shape: tuple[int, int],
        n_eigenvectors: int = 50,
    ):
        """
        Initialize mapping operator with automatic validation and normalization.

        Parameters
        ----------
        mapping_matrix
            The mapping matrix to validate and normalize
        is_self_mapping
            Whether this is self-mapping (square matrix) or cross-mapping
        expected_shape
            Expected shape (n_query_cells, n_reference_cells)
        n_eigenvectors
            Number of eigenvectors to compute for spectral decomposition
        """
        self.is_self_mapping = is_self_mapping
        self.expected_shape = expected_shape
        # Ensure we don't compute too many eigenvectors for small matrices
        max_eigenvectors = max(1, min(expected_shape[0] - 2, n_eigenvectors))
        self.n_eigenvectors = max_eigenvectors

        # Store matrix type information (set during validation)
        self.is_sparse: bool
        self.is_symmetric: bool | None

        # Validate and normalize the matrix
        self.mapping_matrix = self._validate_and_normalize_mapping_matrix(mapping_matrix)

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
        self, mapping_matrix: csr_matrix | coo_matrix | csc_matrix | np.ndarray
    ) -> csr_matrix | np.ndarray:
        """
        Validate and normalize the mapping matrix.

        Parameters
        ----------
        mapping_matrix
            The mapping matrix to validate and normalize (sparse or dense)

        Returns
        -------
        Validated and row-normalized mapping matrix in original format
        """
        # Determine if input is sparse or dense
        self.is_sparse = issparse(mapping_matrix)

        # Validate the shape
        if mapping_matrix.shape != self.expected_shape:
            raise ValueError(
                f"Mapping matrix shape mismatch: expected {self.expected_shape}, but got {mapping_matrix.shape}."
            )

        # Validate self-mapping consistency
        n_rows, n_cols = mapping_matrix.shape
        if self.is_self_mapping and n_rows != n_cols:
            raise ValueError(f"Self-mapping requires square matrix, got shape {mapping_matrix.shape}")

        if not self.is_self_mapping and n_rows == n_cols:
            logger.warning(
                "Square matrix detected but is_self_mapping=False. "
                "Consider setting is_self_mapping=True for matrix powers."
            )

        # Check for symmetry before row-normalization (only for self-mapping)
        if self.is_self_mapping:
            if self.is_sparse:
                # Use sparse symmetry check - compute difference and check if all entries are zero
                diff = mapping_matrix - mapping_matrix.T
                self.is_symmetric = np.allclose(diff.data, 0, rtol=1e-10, atol=1e-12)
            else:
                # Dense matrix symmetry check
                dense_array = np.asarray(mapping_matrix)
                self.is_symmetric = np.allclose(dense_array, dense_array.T, rtol=1e-10, atol=1e-12)

            if self.is_symmetric:
                logger.info(
                    "Input matrix is symmetric - will result in reversible Markov chain after row-normalization."
                )
            else:
                logger.warning(
                    "Input matrix is not symmetric - resulting Markov chain may not be reversible. "
                    "Consider using a symmetric adjacency matrix for better spectral properties."
                )
        else:
            # Non-self-mapping matrices cannot be checked for symmetry
            self.is_symmetric = None
            logger.info("Non-self-mapping matrix - symmetry check skipped.")

        # Compute row sums (shared logic for sparse and dense)
        if self.is_sparse:
            row_sums = mapping_matrix.sum(axis=1).A1  # Convert to 1D array
        else:
            row_sums = np.asarray(mapping_matrix).sum(axis=1)

        # Check for zero rows and handle them
        if np.any(row_sums == 0):
            logger.warning("Some rows in the mapping matrix have a sum of zero. These rows will be left unchanged.")
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        # Normalize if needed, otherwise keep original values
        if not np.allclose(row_sums, 1):
            logger.info("Row-normalizing the mapping matrix.")
            if self.is_sparse:
                mapping_matrix = csr_matrix(mapping_matrix).multiply(1 / row_sums[:, None])
            else:
                mapping_matrix = np.asarray(mapping_matrix) / row_sums[:, None]

        # Ensure proper format and dtype
        if self.is_sparse:
            return csr_matrix(mapping_matrix).astype(np.float32)
        else:
            return np.asarray(mapping_matrix).astype(np.float32)

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

        logger.info(
            "Computing eigendecomposition with %d components for matrix powers",
            self.n_eigenvectors,
        )

        # For row-stochastic matrices, we want the largest eigenvalues
        # Note: the mapping matrix is row-normalized, so largest eigenvalue should be 1
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
        t: int = 1,
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
            Matrix power to apply
        method
            Method for computing matrix powers. Options:
            - "iterative": Iterative matrix multiplication (default)
            - "spectral": Eigendecomposition-based (only for self-mapping)

        Returns
        -------
        mapped_data
            Result of M^t @ reference_data (query_cells x features).
            Maintains sparsity of input data when possible.
        """
        self._validate_power(t)

        if t == 1:
            # Fast path for t=1 regardless of method
            return self.mapping_matrix @ reference_data

        # Apply the chosen method
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
