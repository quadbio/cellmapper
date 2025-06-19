from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import sklearn.neighbors
from scipy.sparse import csr_matrix

from cellmapper.check import check_deps
from cellmapper.logging import logger
from cellmapper.utils import extract_neighbors_from_distances


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
    - Configurable self-edge handling for square matrices
    - Symmetrization options for creating undirected graphs
    - Robust handling of variable neighbor counts and invalid entries

    Self-Edge Handling:
    ------------------
    For square matrices, self-edges (diagonal entries) can be controlled via the `self_edges`
    parameter in connectivity methods:
    - `self_edges=True`: Include self-edges (diagonal entries set to 1)
    - `self_edges=False`: Exclude self-edges (diagonal set to zero)
    - `self_edges=None`: Leave as-is (preserve original neighbor graph structure)

    Attributes
    ----------
    distances
        Array of distances to the nearest neighbors.
    indices
        Array of indices of the nearest neighbors. Self should always
        be included here; however, some approximate algorithms may not return
        the self edge.
    n_targets
        Number of target samples. If None, it is assumed to be the same as
        the number of samples in the indices array.
    """

    distances: np.ndarray
    indices: np.ndarray
    n_targets: int | None = None

    def __post_init__(self):
        """
        Post-initialization logic for NeighborsResults.

        Ensures that `n_targets` is set correctly and validates the shape of
        `distances` and `indices`.
        """
        if self.indices.shape != self.distances.shape:
            raise ValueError("Indices and distances must have the same shape.")

        if self.n_targets is None:
            # Assume square adjacency matrix if `n_targets` is not provided
            self.n_targets = self.indices.shape[0]

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

    def _get_valid_entries_mask(self) -> np.ndarray:
        """
        Helper method to get a mask of valid entries (neither -1 indices nor infinite distances).

        Returns
        -------
        np.ndarray
            Boolean mask of valid entries with shape (n_samples, n_neighbors).
        """
        return (self.indices != -1) & np.isfinite(self.distances)

    def _create_sparse_matrix(self, values: np.ndarray, valid_mask: np.ndarray, dtype=np.float64) -> csr_matrix:
        """
        Helper method to create a sparse matrix from values with filtering of invalid entries.

        Parameters
        ----------
        values
            Values for the sparse matrix, same shape as self.indices/self.distances.
        valid_mask
            Boolean mask of valid entries, same shape as values.
        dtype
            Data type for the matrix values.

        Returns
        -------
        csr_matrix
            Sparse matrix with only valid entries.
        """
        # Flatten all arrays
        flat_indices = self.indices.ravel()
        flat_values = values.ravel()
        valid_entries = valid_mask.ravel()

        # Extract valid data
        valid_indices = flat_indices[valid_entries]
        valid_values = flat_values[valid_entries]

        # Create row indices
        rows = np.repeat(np.arange(self.n_samples), self.n_neighbors)
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
        # Get valid entries mask
        valid_mask = self._get_valid_entries_mask()

        # Create sparse matrix with distances as values
        return self._create_sparse_matrix(self.distances, valid_mask, dtype=dtype)

    def knn_graph_connectivities(
        self,
        kernel: Literal[
            "gaussian", "adaptive_gaussian", "scarches", "random", "inverse_distance", "equal"
        ] = "gaussian",
        symmetric: bool = False,
        self_edges: bool | None = False,
        dtype=np.float64,
        **kwargs,
    ) -> csr_matrix:
        """
        Compute connectivities using the specified kernel.

        Parameters
        ----------
        kernel
            Connectivity kernel to use. Supported: 'gaussian', 'adaptive_gaussian', 'scarches', 'random', 'inverse_distance'.
        symmetric
            If True, create a symmetric connectivity matrix where for each edge i→j,
            ensure j→i exists with the same weight. Only valid for square matrices.
        self_edges
            Control self-edges (diagonal entries) for square matrices:
            - True: Include self-edges (set diagonal entries to 1)
            - False: Exclude self-edges (set diagonal entries to 0)
            - None: Leave as-is (preserve original neighbor graph structure)
        dtype
            Data type for the matrix values.
        **kwargs
            Additional keyword arguments for kernel computation.

        Returns
        -------
        csr_matrix
            Sparse matrix of connectivities (shape: n_samples x n_targets).
            If symmetric=True, the matrix will satisfy W[i,j] = W[j,i] for all edges.
        """
        # Check if symmetric is requested for non-square matrices
        if symmetric and not self.is_square:
            raise ValueError("Symmetric connectivity matrices can only be created for self-mapping (square matrices)")

        # Get valid entries mask
        valid_mask = self._get_valid_entries_mask()

        # Calculate connectivities based on the kernel type
        connectivities = self._compute_kernel_values(kernel, valid_mask, **kwargs)

        # Create sparse matrix with calculated connectivities
        conn_matrix = self._create_sparse_matrix(connectivities, valid_mask, dtype=dtype)

        # Handle self-edges if specified and matrix is square
        if self_edges is not None and self.is_square:
            conn_matrix = self._set_self_edges(conn_matrix, self_edges)

        # Apply symmetrization if requested
        if symmetric:
            conn_matrix = self._symmetrize_matrix(conn_matrix)

        return conn_matrix

    def _compute_kernel_values(
        self,
        kernel: Literal["gaussian", "adaptive_gaussian", "scarches", "random", "inverse_distance", "equal"],
        valid_mask: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Helper method to compute kernel values based on distances.

        Parameters
        ----------
        kernel
            Kernel type to use for computing connectivities.
        valid_mask
            Boolean mask indicating valid entries.
        **kwargs
            Additional arguments for kernel computation.

        Returns
        -------
        np.ndarray
            Array of connectivity values with same shape as distances.
        """
        # Initialize empty connectivities array
        connectivities = np.zeros_like(self.distances)

        # Extract finite distances for parameter calculation
        finite_distances = self.distances[valid_mask]
        if len(finite_distances) == 0:
            raise ValueError("No finite distances found in the neighborhood graph")

        if kernel == "gaussian":
            # Calculate sigma using only finite distances
            sigma = np.mean(finite_distances)
            # Apply Gaussian kernel to valid entries
            connectivities[valid_mask] = np.exp(-(finite_distances**2) / (2 * sigma**2))

        elif kernel == "adaptive_gaussian":
            # Adaptive Gaussian kernel following Haghverdi et al. (2016) / scanpy implementation
            connectivities = self._compute_adaptive_gaussian_kernel(valid_mask, **kwargs)

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

        return connectivities

    def _compute_adaptive_gaussian_kernel(self, valid_mask: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute adaptive Gaussian kernel weights following Haghverdi et al. (2016) / scanpy implementation.

        This implements the adaptive bandwidth scheme where each point gets its own sigma value
        based on the distance to its farthest neighbor, exactly as in scanpy's gauss kernel
        for dense distances (knn=True case).

        Parameters
        ----------
        valid_mask
            Boolean mask indicating valid entries.
        **kwargs
            Additional parameters for the kernel (unused, for API compatibility).

        Returns
        -------
        np.ndarray
            Array of connectivity values with adaptive Gaussian weighting.
        """
        # Compute squared distances
        distances_sq = self.distances**2

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
        connectivities = np.zeros_like(self.distances)

        # Compute adaptive weights following scanpy's vectorized approach within each sample
        for i in range(self.n_samples):
            sample_mask = valid_mask[i, :]
            if not np.any(sample_mask):
                continue

            # Get neighbor indices and squared distances
            neighbor_indices = self.indices[i, sample_mask]
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

    def boolean_adjacency(
        self, dtype=np.float64, self_edges: bool | None = False, symmetric: bool = False
    ) -> csr_matrix:
        """
        Construct a boolean adjacency matrix from neighbor indices.

        Parameters
        ----------
        dtype
            Data type for the matrix values.
        self_edges
            Control self-edges (diagonal entries) for square matrices:
            - True: Include self-edges (set diagonal to 1)
            - False: Exclude self-edges (set diagonal to 0)
            - None: Leave as-is (preserve original neighbor graph structure)
        symmetric
            If True, create a symmetric adjacency matrix where for each edge i→j,
            ensure j→i exists with the same weight. Only valid for square matrices.

        Returns
        -------
        csr_matrix
            Boolean adjacency matrix (shape: n_samples x n_targets), with 1 for each neighbor relationship.
        """
        # Check if symmetric is requested for non-square matrices
        if symmetric and not self.is_square:
            raise ValueError("Symmetric adjacency matrices can only be created for square matrices")

        # Get valid entries mask (only check indices, not distances)
        valid_mask = self.indices != -1

        # Create array of ones with same shape as indices
        ones = np.ones_like(self.indices, dtype=dtype)

        # Create sparse matrix with ones as values for valid entries
        adj_matrix = self._create_sparse_matrix(ones, valid_mask, dtype=dtype)

        # Apply symmetrization if requested and matrix is square
        if symmetric and self.is_square:
            adj_matrix = self._symmetrize_matrix(adj_matrix)

        # Handle self-edges if specified and matrix is square
        if self_edges is not None and self.is_square:
            adj_matrix = self._set_self_edges(adj_matrix, self_edges)

        return adj_matrix

    def _set_self_edges(self, sparse_matrix: csr_matrix, self_edges: bool) -> csr_matrix:
        """
        Set or remove self-edges (diagonal entries) in a sparse connectivity matrix.

        Only applies to square matrices. Non-square matrices are returned unchanged
        since they don't have a meaningful diagonal.

        Parameters
        ----------
        sparse_matrix
            Input sparse connectivity matrix.
        self_edges
            If True, set diagonal entries to 1. If False, set diagonal entries to 0.

        Returns
        -------
        csr_matrix
            Matrix with diagonal entries set according to self_edges parameter.
            Non-square matrices are returned unchanged.
        """
        if not self.is_square:
            # Non-square matrices don't have a meaningful diagonal
            return sparse_matrix

        # Set diagonal entries
        result = sparse_matrix.copy()
        result.setdiag(1.0 if self_edges else 0.0)
        if not self_edges:
            result.eliminate_zeros()  # Remove explicit zeros when excluding self-edges
        return result

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


class Neighbors:
    """Class to compute and store nearest neighbors."""

    def __init__(self, xrep: np.ndarray, yrep: np.ndarray | None = None):
        """
        Initialize the Neighbors class.

        Parameters
        ----------
        xrep
            Representation of the reference dataset.
        yrep
            Representation of the query dataset. If None, self-mapping will be used.
        """
        self.xrep = xrep
        # Use xrep for self-mapping if yrep is None
        self.yrep = yrep if yrep is not None else xrep

        # Initialize neighbor result containers
        self.xx: NeighborsResults | None = None
        self.yy: NeighborsResults | None = None
        self.xy: NeighborsResults | None = None
        self.yx: NeighborsResults | None = None

        # Flag to track if this is a self-mapping case
        self._is_self_mapping = yrep is None

    @classmethod
    def from_distances(cls, distances_matrix: csr_matrix, self_edges: bool | None = None) -> "Neighbors":
        """
        Create a Neighbors object from a pre-computed distances matrix.

        Parameters
        ----------
        distances_matrix
            Sparse distance matrix, typically from adata.obsp['distances']
        self_edges
            If True, include self as a neighbor (cells are their own neighbors).
            If False, exclude self connections, even if present in the distance matrix.
            If None (default), preserve the original behavior of the distance matrix.

        Returns
        -------
        Neighbors
            A new Neighbors object with pre-computed neighbor information
        """
        # Extract indices and distances from the sparse matrix
        indices, distances = extract_neighbors_from_distances(distances_matrix, include_self=self_edges)

        # Create a minimal Neighbors object for self-mapping
        n_cells = distances_matrix.shape[0]
        placeholder_rep = np.zeros((n_cells, 1))
        neighbors = cls(xrep=placeholder_rep)

        # Create a NeighborsResults object with the extracted data
        neighbors_result = NeighborsResults(distances=distances, indices=indices)

        # For self-mapping, all neighbor objects should be the same
        neighbors.xx = neighbors_result
        neighbors.yy = neighbors_result
        neighbors.xy = neighbors_result
        neighbors.yx = neighbors_result

        # Mark as self-mapping
        neighbors._is_self_mapping = True

        logger.info("Created Neighbors object from distances matrix with %d cells", n_cells)

        return neighbors

    def compute_neighbors(
        self,
        n_neighbors: int = 30,
        method: Literal["sklearn", "pynndescent", "rapids", "faiss"] = "sklearn",
        metric: str = "euclidean",
        random_state: int = 0,
        only_yx: bool = False,
    ):
        """
        Compute nearest neighbors using either sklearn or rapids.

        Parameters
        ----------
         n_neighbors
            Number of nearest neighbors.
        method
            Method to use for computing neighbors.
        metric
            Distance metric to use for nearest neighbors.
        random_state
            Random state for reproducibility.
        only_yx
            If True, only compute the xy neighbors.

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``xx``: Nearest neighbors results for reference to reference.
        - ``yy``: Nearest neighbors results for query to query.
        - ``xy``: Nearest neighbors results for reference to query.
        - ``yx``: Nearest neighbors results for query to reference.
        """
        if method in ["rapids", "sklearn", "pynndescent", "faiss"]:
            logger.info("Using %s to compute %d neighbors.", method, n_neighbors)

            if method == "rapids":
                check_deps("cuml")
                import cuml as cm

                check_deps("cupy")
                import cupy as cp

                xrep_gpu = cp.asarray(self.xrep)
                yrep_gpu = cp.asarray(self.yrep)

                xnn = cm.neighbors.NearestNeighbors(n_neighbors=n_neighbors, output_type="numpy", metric=metric).fit(
                    xrep_gpu
                )

                if only_yx:
                    self.yx = NeighborsResults(*xnn.kneighbors(yrep_gpu), n_targets=self.xrep.shape[0])
                    return

                ynn = cm.neighbors.NearestNeighbors(n_neighbors=n_neighbors, output_type="numpy", metric=metric).fit(
                    yrep_gpu
                )

                x_results = xnn.kneighbors(xrep_gpu)
                y_results = ynn.kneighbors(yrep_gpu)
                xy_results = ynn.kneighbors(xrep_gpu)
                yx_results = xnn.kneighbors(yrep_gpu)

            elif method == "faiss":
                check_deps("faiss")
                import faiss

                res = faiss.StandardGpuResources()
                xnn = faiss.IndexFlatL2(self.xrep.shape[1])
                xnn_gpu = faiss.index_cpu_to_gpu(res, 0, xnn)
                xnn_gpu.add(self.xrep)

                if only_yx:
                    self.yx = NeighborsResults(*xnn_gpu.search(self.yrep, n_neighbors), n_targets=self.xrep.shape[0])
                    return

                ynn = faiss.IndexFlatL2(self.yrep.shape[1])
                ynn_gpu = faiss.index_cpu_to_gpu(res, 0, ynn)
                ynn_gpu.add(self.yrep)

                x_results = xnn_gpu.search(self.xrep, n_neighbors)
                y_results = ynn_gpu.search(self.yrep, n_neighbors)
                xy_results = ynn_gpu.search(self.xrep, n_neighbors)
                yx_results = xnn_gpu.search(self.yrep, n_neighbors)

            elif method == "sklearn":
                xnn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(self.xrep)

                if only_yx:
                    self.yx = NeighborsResults(*xnn.kneighbors(self.yrep), n_targets=self.xrep.shape[0])
                    return

                ynn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(self.yrep)

                x_results = xnn.kneighbors(self.xrep)
                y_results = ynn.kneighbors(self.yrep)
                xy_results = ynn.kneighbors(self.xrep)
                yx_results = xnn.kneighbors(self.yrep)

            elif method == "pynndescent":
                check_deps("pynndescent")
                from pynndescent import NNDescent

                xnn = NNDescent(self.xrep, metric=metric, n_jobs=-1, random_state=random_state)

                if only_yx:
                    self.yx = NeighborsResults(*xnn.query(self.yrep, k=n_neighbors)[::-1], n_targets=self.xrep.shape[0])
                    return

                ynn = NNDescent(self.yrep, metric=metric, n_jobs=-1, random_state=random_state)

                x_results = xnn.query(self.xrep, k=n_neighbors)[::-1]
                y_results = ynn.query(self.yrep, k=n_neighbors)[::-1]
                xy_results = ynn.query(self.xrep, k=n_neighbors)[::-1]
                yx_results = xnn.query(self.yrep, k=n_neighbors)[::-1]

            self.xx = NeighborsResults(*x_results, n_targets=None)
            self.yy = NeighborsResults(*y_results, n_targets=None)
            self.xy = NeighborsResults(*xy_results, n_targets=self.yrep.shape[0])
            self.yx = NeighborsResults(*yx_results, n_targets=self.xrep.shape[0])

        else:
            raise ValueError(f"Unknown method: {method}. Supported methods are 'sklearn', 'pynndescent', and 'rapids'.")

    def get_adjacency_matrices(
        self, symmetric: bool = False, self_edges: bool | None = False
    ) -> tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
        """
        Compute unweighted adjacency matrices for all k-NN graphs.

        Parameters
        ----------
        symmetric
            If True, make self-terms (xx, yy) symmetric. Cross-terms (xy, yx) are not affected.
        self_edges
            Control self-edges for self-terms (xx, yy). Cross-terms (xy, yx) are not affected.
            - True: Include self-edges (set diagonal entries to 1)
            - False: Exclude self-edges (set diagonal entries to 0)
            - None: Leave as-is (preserve original neighbor graph structure)

        Returns
        -------
        tuple
            Unweighted adjacency matrices (xx, yy, xy, yx).

        Notes
        -----
        The symmetric and self_edges parameters only apply to self-terms (xx, yy) since
        these represent within-dataset neighborhoods. Cross-terms (xy, yx) represent
        between-dataset relationships where symmetry and self-edges are not meaningful.
        """
        if self.xx is None or self.yy is None or self.xy is None or self.yx is None:
            raise ValueError("Neighbors must be computed before accessing adjacency matrices.")

        # Apply parameters to self-terms (within-dataset neighborhoods)
        xx_adj = self.xx.boolean_adjacency(self_edges=self_edges, symmetric=symmetric)
        yy_adj = self.yy.boolean_adjacency(self_edges=self_edges, symmetric=symmetric)

        # Cross-terms use default parameters (no symmetry/self-edges modification)
        xy_adj = self.xy.boolean_adjacency()
        yx_adj = self.yx.boolean_adjacency()

        return xx_adj, yy_adj, xy_adj, yx_adj

    def __repr__(self):
        """Return a string representation of the Neighbors object."""
        return (
            f"Neighbors(xrep_shape={self.xrep.shape}, yrep_shape={self.yrep.shape}, "
            f"xx={self.xx is not None}, yy={self.yy is not None}, "
            f"xy={self.xy is not None}, yx={self.yx is not None}, "
            f"self_mapping={self._is_self_mapping})"
        )
