from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from umap.umap_ import fuzzy_simplicial_set

from cellmapper.constants import SELF_MAPPING_ONLY_KERNELS
from cellmapper.logging import logger


@dataclass
class NeighborsResults:
    """Nearest neighbors results data store.

    Adapted from the scib-metrics package: https://github.com/YosefLab/scib-metrics.
    Extended to support non-square matrices and potentially varying number of neighbors per cell.
    This class stores the results of nearest neighbor searches and provides methods to compute
    adjacency matrices and connectivities using various kernels.

    Features:
    ---------
    - Multiple connectivity kernels (Gaussian, adaptive Gaussian, inverse distance, etc.)
    - Support for both square (self-mapping) and non-square (cross-mapping) matrices
    - Configurable self-edge inclusion for square matrices in connectivity computations
    - Symmetrization options for creating undirected graphs
    - Robust handling of variable neighbor counts and invalid entries


    Attributes
    ----------
    distances
        Array of distances to the nearest neighbors, excluding self-edges.
        For square matrices, self-edges are automatically removed during initialization
        to ensure consistent storage format across different k-NN algorithms.
    indices
        Array of indices of the nearest neighbors, excluding self-edges.
        For square matrices, self-edges are automatically removed during initialization
        to ensure consistent storage format across different k-NN algorithms.
    n_targets
        Number of target samples. If None, it is assumed to be the same as
        the number of samples in the indices array.

    Notes
    -----
    The `n_neighbors` property always refers to non-self neighbors (self-edges are automatically
    removed during initialization for consistent storage). When self-edges are included via
    connectivity methods with `self_edges=True`, the resulting arrays will have shape
    (n_samples, n_neighbors + 1), but the `n_neighbors` property remains unchanged.
    """

    distances: np.ndarray
    indices: np.ndarray
    n_targets: int | None = None

    def __post_init__(self):
        """
        Post-initialization logic for NeighborsResults.

        Ensures that `n_targets` is set correctly, validates array shapes,
        and removes self-edges from square matrices for consistent storage.
        """
        if self.indices.shape != self.distances.shape:
            raise ValueError("Indices and distances must have the same shape.")

        if self.n_targets is None:
            # Assume square adjacency matrix if `n_targets` is not provided
            self.n_targets = self.indices.shape[0]

        # Remove self-edges for consistent storage (square matrices only)
        if self.is_square:
            self._remove_self_edges_from_storage()

    @property
    def n_samples(self) -> int:
        """Number of samples (cells)."""
        return self.indices.shape[0]

    @property
    def n_neighbors(self) -> int:
        """Number of neighbors."""
        return self.indices.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the adjacency/graph matrices (n_samples, n_targets)."""
        return (self.n_samples, self.n_targets or self.n_samples)

    @property
    def is_square(self) -> bool:
        """Whether this represents a square matrix (self-mapping)."""
        return self.n_samples == (self.n_targets or self.n_samples)

    def _create_sparse_matrix_from_arrays(
        self, values: np.ndarray, indices: np.ndarray, valid_mask: np.ndarray, dtype=np.float64
    ) -> csr_matrix:
        """
        Helper method to create a sparse matrix from values and indices with filtering of invalid entries.

        Parameters
        ----------
        values
            Values for the sparse matrix, same shape as indices.
        indices
            Indices array (may include self-edges).
        valid_mask
            Boolean mask of valid entries, same shape as values and indices.
        dtype
            Data type for the matrix values.

        Returns
        -------
        csr_matrix
            Sparse matrix with only valid entries.
        """
        # Flatten all arrays
        flat_indices = indices.ravel()
        flat_values = values.ravel()
        valid_entries = valid_mask.ravel()

        # Extract valid data
        valid_indices = flat_indices[valid_entries]
        valid_values = flat_values[valid_entries]

        # Create row indices
        rows = np.repeat(np.arange(self.n_samples), indices.shape[1])
        rows = rows[valid_entries]

        # Create CSR matrix with only valid entries
        return csr_matrix((valid_values.astype(dtype), (rows, valid_indices)), shape=self.shape)

    @cached_property
    def knn_graph_distances(self, dtype=np.float64) -> csr_matrix:
        """
        Return the sparse weighted adjacency matrix of distances.

        Parameters
        ----------
        dtype
            Data type for the matrix values.

        Returns
        -------
        csr_matrix
            Sparse matrix of distances (shape: n_samples x n_targets).
        """
        # Get distances, indices, and valid mask without self-edges by default
        distances, indices, valid_mask = self._get_distances_and_indices(self_edges=False)

        # Create sparse matrix with distances as values
        return self._create_sparse_matrix_from_arrays(distances, indices, valid_mask, dtype=dtype)

    def knn_graph_connectivities(
        self,
        kernel: Literal[
            "gaussian", "adaptive_gaussian", "scarches", "random", "inverse_distance", "equal", "umap"
        ] = "gaussian",
        symmetrize: bool = False,
        self_edges: bool = False,
        dtype=np.float64,
        **kwargs,
    ) -> csr_matrix:
        """
        Compute connectivities using the specified kernel.

        Parameters
        ----------
        kernel
            Connectivity kernel to use. Supported: 'gaussian', 'adaptive_gaussian', 'scarches', 'random', 'inverse_distance', 'equal', 'umap'.
        symmetrize
            If True, create a symmetrize connectivity matrix where for each edge i→j,
            ensure j→i exists with the same weight. Only valid for square matrices.
        self_edges
            Control self-edges in the neighbor graph for square matrices:
            - True: Include self-edges in the connectivity computation
            - False: Exclude self-edges from the connectivity computation
        dtype
            Data type for the matrix values.
        **kwargs
            Additional keyword arguments for kernel computation.

        Returns
        -------
        csr_matrix
            Sparse matrix of connectivities (shape: n_samples x n_targets).
            If symmetrize=True, the matrix will satisfy W[i,j] = W[j,i] for all edges.
        """
        # Check if symmetrize is requested for non-square matrices
        if symmetrize and not self.is_square:
            raise ValueError("symmetrize connectivity matrices can only be created for self-mapping (square matrices)")

        # Check if self-mapping only kernel is used for non-square matrices
        if kernel in SELF_MAPPING_ONLY_KERNELS and not self.is_square:
            raise ValueError(f"Kernel '{kernel}' is only supported for self-mapping (square matrices)")

        # Warn about UMAP inherent symmetry
        if kernel == "umap" and not symmetrize:
            logger.info(
                "UMAP kernel inherently produces symmetric connectivities. "
                "Consider setting symmetrize=True for consistency with other kernels."
            )

        # Compute connectivities using the specified kernel (all kernels now return sparse matrices)
        if kernel == "umap":
            conn_matrix = self._compute_umap_kernel(self_edges, **kwargs)
        else:
            conn_matrix = self._compute_kernel_values(kernel, self_edges, dtype=dtype, **kwargs)

        # Apply symmetrization if requested
        if symmetrize:
            conn_matrix = self._symmetrize_matrix(conn_matrix)

        return conn_matrix

    def _compute_kernel_values(
        self,
        kernel: Literal["gaussian", "adaptive_gaussian", "scarches", "random", "inverse_distance", "equal"],
        self_edges: bool,
        dtype=np.float64,
        **kwargs,
    ) -> csr_matrix:
        """
        Helper method to compute kernel values based on distances.

        Parameters
        ----------
        kernel
            Kernel type to use for computing connectivities.
        self_edges
            Whether to include self-edges in the computation.
        dtype
            Data type for the matrix values.
        **kwargs
            Additional arguments for kernel computation.

        Returns
        -------
        csr_matrix
            Sparse matrix of connectivity values.
        """
        # Get distances, indices, and valid mask with appropriate self-edge handling
        distances, indices, valid_mask = self._get_distances_and_indices(self_edges)

        # Initialize empty connectivities array
        connectivities = np.zeros_like(distances)

        # Extract finite distances for parameter calculation
        finite_distances = distances[valid_mask]
        if len(finite_distances) == 0:
            raise ValueError("No finite distances found in the neighborhood graph")

        if kernel == "gaussian":
            # Calculate sigma using only finite distances
            sigma = np.mean(finite_distances)
            # Apply Gaussian kernel to valid entries
            connectivities[valid_mask] = np.exp(-(finite_distances**2) / (2 * sigma**2))

        elif kernel == "adaptive_gaussian":
            # Adaptive Gaussian kernel following Haghverdi et al. (2016) / scanpy implementation
            connectivities = self._compute_adaptive_gaussian_kernel(self_edges, **kwargs)

        elif kernel == "equal":
            # Set connectivities to 1 for valid entries
            connectivities[valid_mask] = 1.0

        elif kernel == "scarches":
            # Calculate sigma using only finite distances
            sigma = np.std(finite_distances)
            sigma = (2.0 / sigma) ** 2
            # Apply scArches kernel to valid entries
            connectivities[valid_mask] = np.exp(-finite_distances / sigma)

        elif kernel == "random":  # this is for testing purposes
            # Generate random connectivities for valid entries
            connectivities[valid_mask] = np.random.rand(np.sum(valid_mask))

        elif kernel == "inverse_distance":
            # Get epsilon parameter with default
            epsilon = kwargs.get("epsilon", 1e-8)
            # Apply inverse distance kernel to valid entries
            connectivities[valid_mask] = 1.0 / (finite_distances + epsilon)

        else:
            raise ValueError(
                f"Unknown kernel: {kernel}. Supported kernels are: 'gaussian', 'adaptive_gaussian', 'scarches', 'random', 'inverse_distance', 'equal'."
            )

        # Create and return sparse matrix
        return self._create_sparse_matrix_from_arrays(connectivities, indices, valid_mask, dtype=dtype)

    def _compute_umap_kernel(self, self_edges: bool, **kwargs) -> csr_matrix:
        """
        Compute UMAP fuzzy simplicial set connectivities following scanpy implementation.

        This calls umap-learn's fuzzy_simplicial_set function with a dummy data matrix
        to compute the connectivity weights based on the k-NN graph structure.

        Parameters
        ----------
        self_edges
            Whether to include self-edges in the computation.
        **kwargs
            Additional parameters for UMAP kernel:
            - set_op_mix_ratio: float, default 1.0
            - local_connectivity: float, default 1.0

        Returns
        -------
        csr_matrix
            Sparse connectivity matrix.
        """
        # Get distances, indices, and valid mask with appropriate self-edge handling
        distances, indices, _valid_mask = self._get_distances_and_indices(self_edges)

        # UMAP kernel requires true k-NN graphs (no padding with -1)
        if np.any(indices == -1):
            raise ValueError("UMAP kernel requires true k-NN graphs (all cells must have exactly k neighbors)")

        # Extract UMAP-specific parameters
        set_op_mix_ratio = kwargs.get("set_op_mix_ratio", 1.0)
        local_connectivity = kwargs.get("local_connectivity", 1.0)

        # Create dummy data matrix (scanpy approach)
        X = coo_matrix((self.n_samples, 1))

        # Call UMAP's fuzzy_simplicial_set
        result = fuzzy_simplicial_set(
            X,
            indices.shape[1],
            None,  # random_state
            None,  # metric
            knn_indices=indices,
            knn_dists=distances,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
        )

        # Extract the connectivity matrix (first element of tuple)
        connectivities_sparse = result[0]

        # Return as CSR matrix
        return connectivities_sparse.tocsr()

    def _compute_adaptive_gaussian_kernel(self, self_edges: bool, **kwargs) -> np.ndarray:
        """
        Compute adaptive Gaussian kernel weights following Haghverdi et al. (2016) / scanpy implementation.

        This implements the adaptive bandwidth scheme where each point gets its own sigma value
        based on the distance to its farthest neighbor, exactly as in scanpy's gauss kernel
        for dense distances (knn=True case).

        Parameters
        ----------
        self_edges
            Whether to include self-edges in the computation.
        **kwargs
            Additional parameters for the kernel (unused, for API compatibility).

        Returns
        -------
        np.ndarray
            Array of connectivity values with adaptive Gaussian weighting.
        """
        # Get distances, indices, and valid mask with appropriate self-edge handling
        distances, indices, valid_mask = self._get_distances_and_indices(self_edges)

        # Compute squared distances
        distances_sq = distances**2

        # Compute sigma using the farthest valid neighbor for each sample (scanpy approach)
        # This is equivalent to scanpy's: sigmas_sq = distances_sq[:, -1] / 4
        # but handles variable neighbor counts via valid_mask
        sigmas_sq = np.array(
            [
                np.max(distances_sq[i, valid_mask[i, :]]) / 4.0 if np.any(valid_mask[i, :]) else 1.0
                for i in range(self.n_samples)
            ]
        )
        sigmas = np.sqrt(sigmas_sq)

        # Initialize connectivities
        connectivities = np.zeros_like(distances)

        # Compute adaptive weights following scanpy's vectorized approach within each sample
        for i in range(self.n_samples):
            sample_mask = valid_mask[i, :]
            if not np.any(sample_mask):
                continue

            # Get neighbor indices and squared distances
            neighbor_indices = indices[i, sample_mask]
            sample_distances_sq = distances_sq[i, sample_mask]

            # Adaptive kernel computation (scanpy's formula)
            # W[i,j] = sqrt((2 * sigma_i * sigma_j) / (sigma_i^2 + sigma_j^2)) * exp(-d_ij^2 / (sigma_i^2 + sigma_j^2))
            sigma_i = sigmas[i]
            sigma_j = sigmas[neighbor_indices]

            num = 2 * sigma_i * sigma_j
            den = sigmas_sq[i] + sigmas_sq[neighbor_indices]

            # Compute weights (vectorized)
            weights = np.sqrt(num / den) * np.exp(-sample_distances_sq / den)
            connectivities[i, sample_mask] = weights

        return connectivities

    def boolean_adjacency(self, dtype=np.float64, self_edges: bool = False, symmetrize: bool = False) -> csr_matrix:
        """
        Construct a boolean adjacency matrix from neighbor indices.

        Parameters
        ----------
        dtype
            Data type for the matrix values.
        self_edges
            Control self-edges in the neighbor graph for square matrices:
            - True: Include self-edges in the adjacency computation
            - False: Exclude self-edges from the adjacency computation
        symmetrize
            If True, create a symmetrize adjacency matrix where for each edge i→j,
            ensure j→i exists with the same weight. Only valid for square matrices.

        Returns
        -------
        csr_matrix
            Boolean adjacency matrix (shape: n_samples x n_targets), with 1 for each neighbor relationship.
        """
        # Check if symmetrize is requested for non-square matrices
        if symmetrize and not self.is_square:
            raise ValueError("symmetrize adjacency matrices can only be created for square matrices")

        # Get distances, indices, and valid mask with appropriate self-edge handling
        _distances, indices, valid_mask = self._get_distances_and_indices(self_edges)

        # Create array of ones with same shape as indices
        ones = np.ones_like(indices, dtype=dtype)

        # Create sparse matrix with ones as values for valid entries
        adj_matrix = self._create_sparse_matrix_from_arrays(ones, indices, valid_mask, dtype=dtype)

        # Apply symmetrization if requested and matrix is square
        if symmetrize and self.is_square:
            adj_matrix = self._symmetrize_matrix(adj_matrix)

        return adj_matrix

    def _symmetrize_matrix(self, sparse_matrix: csr_matrix) -> csr_matrix:
        """
        Apply symmetrization to a sparse connectivity matrix.

        For each existing edge i→j, ensure that j→i exists with the same weight.
        This creates undirected graphs by copying edge weights rather than adding them.

        Only applies to square matrices.

        Parameters
        ----------
        sparse_matrix
            Input sparse connectivity matrix to symmetrize.

        Returns
        -------
        csr_matrix
            Symmetrized sparse matrix where W[i,j] = W[j,i] for all existing edges.
        """
        if not self.is_square:
            raise ValueError("Can only symmetrize square matrices")

        # Convert to LIL format for efficient item assignment
        matrix_lil = sparse_matrix.tolil()

        # Get all existing nonzero entries
        nonzero_indices = sparse_matrix.nonzero()
        rows, cols = nonzero_indices[0], nonzero_indices[1]

        # For each existing edge i→j, ensure j→i exists with the same weight
        for i, j in zip(rows, cols, strict=False):
            if matrix_lil[j, i] == 0:  # If reverse edge j→i doesn't exist
                matrix_lil[j, i] = matrix_lil[i, j]  # Copy forward edge weight

        # Convert back to CSR format for efficiency
        return matrix_lil.tocsr()

    def _remove_self_edges_from_storage(self):
        """Remove self-edges from stored distances and indices if present.

        Only applies to square matrices. Checks if the first column contains
        self-references (0, 1, 2, 3, ...) and removes it if found.
        """
        if not self.is_square:
            return  # Nothing to do for non-square matrices

        # Check if first column contains self-references (0, 1, 2, 3, ...)
        expected_self_indices = np.arange(self.n_samples)
        has_self_edges = np.array_equal(self.indices[:, 0], expected_self_indices)

        if has_self_edges:
            # Remove first column (self-edges) and keep remaining columns
            self.indices = self.indices[:, 1:]
            self.distances = self.distances[:, 1:]

    def _get_distances_and_indices(self, self_edges: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get distances and indices with optional self-edge handling, plus valid entries mask.

        Parameters
        ----------
        self_edges
            Control self-edge inclusion:
            - True: Add self-edges as first column with distance 0.0
            - False: Return stored arrays (without self-edges)

        Returns
        -------
        tuple
            Modified distances, indices, and valid_mask arrays based on self_edges parameter.
            When self_edges=True, arrays will have shape (n_samples, n_neighbors + 1).
            Valid mask identifies entries that are neither -1 indices nor infinite distances.
        """
        if self.is_square and self_edges:
            distances, indices = self._add_self_edges(self.distances, self.indices)
        else:
            distances, indices = self.distances, self.indices

        # Create valid entries mask
        valid_mask = (indices != -1) & np.isfinite(distances)

        return distances, indices, valid_mask

    def _add_self_edges(self, distances: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Add self-edges as first column with distance 0.0.

        Parameters
        ----------
        distances
            Distance array without self-edges
        indices
            Indices array without self-edges

        Returns
        -------
        tuple
            Arrays with self-edges prepended as first column.
            Shape increases from (n_samples, n_neighbors) to (n_samples, n_neighbors + 1).
        """
        # Create self-edge columns
        self_indices = np.arange(self.n_samples, dtype=indices.dtype).reshape(-1, 1)
        self_distances = np.zeros((self.n_samples, 1), dtype=distances.dtype)

        # Prepend self-edges to existing arrays (no truncation)
        new_indices = np.column_stack([self_indices, indices])
        new_distances = np.column_stack([self_distances, distances])

        return new_distances, new_indices
