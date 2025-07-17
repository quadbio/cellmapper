"""Mapping operator for applying powers of mapping matrices."""

from functools import cached_property
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse
from scipy.sparse.linalg import eigsh

from cellmapper._docs import d
from cellmapper.constants import PackageConstants
from cellmapper.logging import logger
from cellmapper.model.kernel import Kernel


class MappingOperator:
    """Operator for applying powers of mapping matrices with validation and normalization.

    This class provides two methods for computing matrix powers M^t:

    1. **Iterative method**: Computes M^t by repeated matrix multiplication (M @ M @ ... @ M).
       This is exact but can be slow for large t, inspired by MAGIC :cite:`van2018recovering`.

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

    **Sparsity:**

    - Iterative method preserves input sparsity when mapping matrix is sparse
    - Spectral method always produces dense output due to eigendecomposition operations
    - In practice, imputed matrices are typically dense regardless of method
    """

    def __init__(
        self,
        kernel_matrix: Kernel | csr_matrix | coo_matrix | csc_matrix | np.ndarray,
        is_self_mapping: bool | None = None,
        n_eigenvectors: int = 50,
        eigen_solver: Literal["partial", "complete"] = "partial",
    ):
        """Initialize mapping operator with automatic validation and normalization.

        Parameters
        ----------
        kernel_matrix
            The kernel matrix or Kernel object to use for mapping.
            If a Kernel object is provided, kernel_matrix and is_self_mapping
            are extracted automatically.
        is_self_mapping
            Whether this is self-mapping (square matrix) or cross-mapping.
            If None, will be inferred from the Kernel object or matrix shape.
        n_eigenvectors
            Number of eigenvectors to compute for spectral decomposition.
            More eigenvectors = better approximation but slower computation.
            Automatically capped to ensure numerical stability.
        eigen_solver
            Eigendecomposition method for spectral approach:
            - "partial": Uses sparse eigendecomposition (scipy.sparse.linalg.eigs), faster
            - "complete": Uses complete eigendecomposition (scipy.linalg.eig), exact for testing
        """
        # Extract matrix and metadata from Kernel object if provided

        if isinstance(kernel_matrix, Kernel):
            # This is a Kernel object
            kernel_obj = kernel_matrix
            actual_matrix = kernel_obj.kernel_matrix
            if actual_matrix is None:
                raise ValueError(
                    "Kernel object does not have a computed kernel matrix. Call compute_kernel_matrix() first."
                )

            # Extract is_self_mapping from Kernel if not provided
            if is_self_mapping is None:
                is_self_mapping = kernel_obj._is_self_mapping

            kernel_matrix = actual_matrix
        else:
            # This is a raw matrix
            actual_matrix = kernel_matrix

            # Infer is_self_mapping from matrix shape if not provided
            if is_self_mapping is None:
                n_rows, n_cols = actual_matrix.shape
                is_self_mapping = n_rows == n_cols
                logger.info("Inferred is_self_mapping=%s from matrix shape %s", is_self_mapping, actual_matrix.shape)

        self.is_self_mapping = is_self_mapping
        self.eigen_solver = eigen_solver

        # Ensure we don't compute too many eigenvectors for small matrices
        matrix_size = actual_matrix.shape[0]
        if eigen_solver == "complete":
            # For complete eigendecomposition, use all eigenvectors
            self.n_eigenvectors = matrix_size
        elif eigen_solver == "partial":
            # For partial eigendecomposition, cap the number of eigenvectors
            max_eigenvectors = max(1, min(matrix_size - 2, n_eigenvectors))
            self.n_eigenvectors = max_eigenvectors
        else:
            raise ValueError(f"Unknown eigen_solver: {eigen_solver}. Use 'partial' or 'complete'.")

        # Store matrix type information (set during validation)
        self.is_sparse: bool
        self.is_symmetric: bool | None
        self.row_degrees: np.ndarray  # Row sums of original kernel matrix

        # Validate and normalize the matrix
        self.mapping_matrix = self._validate_and_normalize_mapping_matrix(actual_matrix)

    @property
    def matrix(self) -> csr_matrix | np.ndarray:
        """Get the underlying mapping matrix.

        Returns
        -------
        The validated and normalized mapping matrix in original format
        """
        return self.mapping_matrix

    def _validate_and_normalize_mapping_matrix(
        self, kernel_matrix: csr_matrix | coo_matrix | csc_matrix | np.ndarray
    ) -> csr_matrix | np.ndarray:
        """Validate and normalize the mapping matrix.

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

            if not self.is_symmetric:
                logger.warning(
                    "Input matrix is not symmetric - spectral methods require symmetric matrices. "
                    "Consider using a symmetric adjacency matrix or the 'iterative' method."
                )
            else:
                logger.debug("Input matrix is symmetric - will result in reversible Markov chain.")
        else:
            # Non-self-mapping matrices cannot be checked for symmetry
            self.is_symmetric = None
            logger.debug("Non-self-mapping matrix - symmetry check skipped.")

        # Compute and store row degrees (row sums of original kernel matrix)
        if self.is_sparse:
            self.row_degrees = np.asarray(kernel_matrix.sum(axis=1)).flatten()
        else:
            self.row_degrees = np.asarray(kernel_matrix).sum(axis=1)

        # Check for zero rows and handle them
        if np.any(self.row_degrees == 0):
            logger.warning("Some rows in the mapping matrix have a sum of zero. These rows will be left unchanged.")

        # Create a copy of row_degrees for normalization to avoid division by zero
        row_sums = self.row_degrees.copy()
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        # (Asymmetric) row-normalization and ensure proper format and dtype
        if self.is_sparse:
            kernel_matrix = csr_matrix(kernel_matrix).multiply(1 / row_sums[:, None])
            return csr_matrix(kernel_matrix).astype(np.float32)
        else:
            kernel_matrix = np.asarray(kernel_matrix) / row_sums[:, None]
            return kernel_matrix.astype(np.float32)

    def _validate_power(self, t: int) -> None:
        """Validate that the requested power is feasible."""
        if t < 1:
            raise ValueError(f"Power t must be >= 1, got {t}")

        if t > 1 and not self.is_self_mapping:
            raise ValueError(f"Matrix powers t > 1 only supported for self-mapping mode, got t={t}")

    @cached_property
    def _eigendecomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute and cache eigendecomposition for self-mapping matrices.

        Uses symmetric eigendecomposition of T_sym = D^1/2 T_asym D^-1/2
        and converts eigenvectors back to T_asym space using u = D^-1/2 w.

        Returns
        -------
        eigenvalues, eigenvectors
            Real eigenvalues and corresponding eigenvectors for the asymmetric transition matrix T_asym
        """
        if not self.is_self_mapping:
            raise RuntimeError("Eigendecomposition only available for self-mapping mode")

        # Get the symmetric matrix for eigendecomposition
        symmetric_matrix = self._symmetric_mapping_matrix

        if self.eigen_solver == "complete":
            logger.info("Computing complete symmetric eigendecomposition for matrix powers")
            # Convert to dense for complete eigendecomposition
            if self.is_sparse:
                dense_matrix = symmetric_matrix.toarray()
            else:
                dense_matrix = np.asarray(symmetric_matrix)

            # Use symmetric eigendecomposition (guaranteed real eigenvalues)
            eigenvalues, eigenvectors_sym = eigh(dense_matrix)  # type: ignore[assignment]

        else:
            logger.info(
                "Computing symmetric eigendecomposition with %d components for matrix powers",
                self.n_eigenvectors,
            )
            # Use symmetric sparse eigendecomposition
            eigenvalues, eigenvectors_sym = eigsh(  # type: ignore[misc]
                symmetric_matrix,
                k=self.n_eigenvectors,
                which="LM",  # Largest magnitude
                return_eigenvectors=True,
            )

        # Sort by eigenvalue magnitude (descending) for proper diffusion ordering
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors_sym = eigenvectors_sym[:, idx]  # Convert symmetric eigenvectors to asymmetric space: u = D^-1/2 w
        _, inv_sqrt_degrees = self._compute_degree_transforms()

        # Apply D^-1/2 to each eigenvector
        eigenvectors_asym = eigenvectors_sym * inv_sqrt_degrees[:, np.newaxis]

        # All eigenvalues and eigenvectors are real due to symmetry
        return eigenvalues.astype(np.float32), eigenvectors_asym.astype(np.float32)

    @cached_property
    def _symmetric_mapping_matrix(self) -> csr_matrix | np.ndarray:
        """Generate symmetric mapping matrix on-the-fly for spectral methods.

        Uses the relationship: T_sym = D^1/2 T_asym D^-1/2
        where D is the diagonal matrix of row degrees.

        Returns
        -------
        Symmetric mapping matrix for eigendecomposition
        """
        if not self.is_self_mapping:
            raise RuntimeError("Symmetric mapping matrix only available for self-mapping mode")

        if not self.is_symmetric:
            raise ValueError(
                "Cannot generate symmetric mapping matrix from asymmetric kernel matrix. "
                "Provide a symmetric adjacency/kernel matrix."
            )

        # Compute D^1/2 and D^-1/2
        sqrt_degrees, inv_sqrt_degrees = self._compute_degree_transforms()

        # Compute T_sym = D^1/2 T_asym D^-1/2
        if self.is_sparse:
            # For sparse matrices, use diagonal matrices
            D_sqrt = sparse.diags(sqrt_degrees, format="csr")
            D_inv_sqrt = sparse.diags(inv_sqrt_degrees, format="csr")
            symmetric_matrix = D_sqrt @ self.mapping_matrix @ D_inv_sqrt
        else:
            # For dense matrices, use broadcasting
            symmetric_matrix = (sqrt_degrees[:, None] * self.mapping_matrix) * inv_sqrt_degrees[None, :]

        return symmetric_matrix

    def _compute_degree_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute D^1/2 and D^-1/2 from row degrees.

        Returns
        -------
        sqrt_degrees, inv_sqrt_degrees
            Square root of degrees and inverse square root of degrees
        """
        sqrt_degrees = np.sqrt(self.row_degrees)
        inv_sqrt_degrees = np.zeros_like(sqrt_degrees)

        # Handle zero degrees (avoid division by zero)
        nonzero_mask = sqrt_degrees > 0
        inv_sqrt_degrees[nonzero_mask] = 1.0 / sqrt_degrees[nonzero_mask]

        return sqrt_degrees, inv_sqrt_degrees

    def _apply_iterative(self, reference_data: np.ndarray | csr_matrix, t: int) -> np.ndarray | csr_matrix:
        """Apply matrix power using iterative multiplication.

        This is based on ideas from MAGIC :cite:`van2018recovering`.

        Parameters
        ----------
        reference_data
            Data to map (reference_cells x features)
        t
            Matrix power to apply (t >= 1)

        Returns
        -------
        Result of M^t @ reference_data
        """
        logger.debug("Using iterative multiplication for t=%d", t)
        result = reference_data.copy()
        for _ in range(t):
            result = self.mapping_matrix @ result
        return result

    def _apply_spectral(self, reference_data: np.ndarray | csr_matrix, t: int) -> np.ndarray | csr_matrix:
        """Apply matrix power using cached eigendecomposition.

        Parameters
        ----------
        reference_data
            Data to map (reference_cells x features)
        t
            Matrix power to apply (t >= 1)

        Returns
        -------
        Result of M^t @ reference_data using spectral approximation.
        Always returns dense array due to eigendecomposition operations.
        """
        logger.debug("Using spectral decomposition for t=%d", t)

        eigenvalues, eigenvectors = self._eigendecomposition

        # Project data onto eigenvector space
        projected = eigenvectors.T @ reference_data

        # Apply eigenvalue powers
        powered = (eigenvalues[:, np.newaxis] ** t) * projected

        # Project back to original space
        result = eigenvectors @ powered

        return result

    @d.dedent
    def apply(
        self,
        reference_data: np.ndarray | csr_matrix,
        t: int | None = None,
        diffusion_method: Literal["iterative", "spectral"] = "iterative",
    ) -> np.ndarray | csr_matrix:
        """Apply mapping matrix power: M^t @ reference_data.

        Parameters
        ----------
        reference_data
            Data to map (reference_cells x features). Can be dense or sparse arrays.
        %(t)s
        %(diffusion_method)s

        Returns
        -------
        Result of M^t @ reference_data (query_cells x features).
        Iterative method preserves sparsity when mapping matrix is sparse.
        Spectral method always returns dense arrays due to eigendecomposition operations.

        Notes
        -----
        The spectral method approximates the iterative method using eigendecomposition.
        See the class docstring for detailed trade-offs between accuracy and performance.

        **Sparsity Considerations:**

        - Iterative method preserves input sparsity when mapping matrix is sparse
        - Spectral method always produces dense output due to eigenvector operations
        - In practice, imputed matrices are typically dense regardless of method
        """
        if t is None:
            # Direct multiplication - fastest path
            logger.debug("Using direct matrix multiplication (t=None)")
            return self.mapping_matrix @ reference_data

        # Validate the power (only for non-None values)
        self._validate_power(t)

        # Warn about performance for large matrix powers with iterative method
        if t > PackageConstants.SPECTRAL_METHOD_THRESHOLD and diffusion_method == "iterative" and self.is_self_mapping:
            logger.warning(
                "Using iterative method for t=%d matrix powers may be slow for large datasets. "
                "Consider using method='spectral' for better performance.",
                t,
            )

        # Apply the chosen method for t >= 1
        if diffusion_method == "iterative":
            return self._apply_iterative(reference_data, t)
        elif diffusion_method == "spectral":
            return self._apply_spectral(reference_data, t)
        else:
            raise ValueError(f"Unknown diffusion method: {diffusion_method}")

    def clear_cache(self) -> None:
        """Clear cached eigendecomposition to free memory."""
        if hasattr(self, "_eigendecomposition"):
            delattr(self, "_eigendecomposition")
            logger.debug("Cleared eigendecomposition cache")

    def __repr__(self) -> str:
        """Return string representation of the MappingOperator.

        Returns
        -------
        String representation with key properties
        """
        # Basic properties
        shape_str = f"{self.mapping_matrix.shape[0]}×{self.mapping_matrix.shape[1]}"
        matrix_type = "sparse" if self.is_sparse else "dense"
        mapping_type = "self-mapping" if self.is_self_mapping else "cross-mapping"

        # Symmetry information
        if self.is_symmetric is None:
            symmetry_str = "N/A"
        elif self.is_symmetric:
            symmetry_str = "symmetric"
        else:
            symmetry_str = "asymmetric"

        # Matrix statistics
        if self.is_sparse:
            # Safe access to nnz attribute for sparse matrices
            nnz = getattr(self.mapping_matrix, "nnz", 0)
            sparsity = nnz / (self.mapping_matrix.shape[0] * self.mapping_matrix.shape[1])
            sparsity_str = f", sparsity: {sparsity:.1%}"
        else:
            sparsity_str = ""

        # Zero degree information
        zero_degrees = np.sum(self.row_degrees == 0)
        zero_str = f", zero-degree rows: {zero_degrees}" if zero_degrees > 0 else ""

        # Eigendecomposition status (only check if self-mapping to avoid errors)
        if self.is_self_mapping:
            # Check if the cached property has been computed without triggering it
            has_eigen = "_eigendecomposition" in self.__dict__
            eigen_str = f", eigendecomp: {'cached' if has_eigen else 'not computed'}"
        else:
            eigen_str = ""

        return (
            f"MappingOperator({shape_str}, {mapping_type}, {matrix_type}, "
            f"{symmetry_str}{sparsity_str}{zero_str}{eigen_str})"
        )

    def __str__(self) -> str:
        """Return user-friendly string representation of the MappingOperator.

        Returns
        -------
        User-friendly string representation
        """
        mapping_type = "Self-mapping" if self.is_self_mapping else "Cross-mapping"
        matrix_type = "sparse" if self.is_sparse else "dense"

        # Build description
        desc = f"{mapping_type} operator ({self.mapping_matrix.shape[0]}×{self.mapping_matrix.shape[1]}, {matrix_type}"

        if self.is_symmetric is not None:
            symmetry = "symmetric" if self.is_symmetric else "asymmetric"
            desc += f", {symmetry}"

        if self.is_sparse:
            nnz = getattr(self.mapping_matrix, "nnz", 0)
            total_elements = self.mapping_matrix.shape[0] * self.mapping_matrix.shape[1]
            sparsity = nnz / total_elements
            desc += f", {sparsity:.1%} filled"

        desc += ")"

        # Add warnings/info
        zero_degrees = np.sum(self.row_degrees == 0)
        if zero_degrees > 0:
            desc += f"\nWarning: {zero_degrees} rows have zero degree"

        if self.is_self_mapping and not self.is_symmetric:
            desc += "\nNote: Spectral methods require symmetric matrices"

        return desc
