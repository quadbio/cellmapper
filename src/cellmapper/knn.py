from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import sklearn.neighbors
from pynndescent import NNDescent
from scipy.sparse import csr_matrix

from cellmapper.logging import logger

from .check import check_deps
from .utils import extract_neighbors_from_distances


@dataclass
class NeighborsResults:
    """Nearest neighbors results data store.

    Adapted from the scib-metrics package: https://github.com/YosefLab/scib-metrics. Extended to non-square matrices and potentially
    varying number of neighbors per cell. This class is used to store the results of nearest neighbor searches, including
    distances and indices of the nearest neighbors. It also provides methods to compute adjacency matrices and connectivities based on the nearest neighbors.

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
        kernel: Literal["gaussian", "scarches", "random", "inverse_distance"] = "gaussian",
        dtype=np.float64,
        **kwargs,
    ) -> csr_matrix:
        """
        Compute connectivities using the specified kernel.

        Parameters
        ----------
        kernel
            Connectivity kernel to use. Supported: 'gaussian', 'scarches', 'random', 'inverse_distance'.
        dtype
            Data type for the matrix values.
        **kwargs
            Additional keyword arguments for kernel computation.

        Returns
        -------
        csr_matrix
            Sparse matrix of connectivities (shape: n_samples x n_targets).
        """
        # Get valid entries mask
        valid_mask = self._get_valid_entries_mask()

        # Calculate connectivities based on the kernel type
        connectivities = self._compute_kernel_values(kernel, valid_mask, **kwargs)

        # Create sparse matrix with calculated connectivities
        return self._create_sparse_matrix(connectivities, valid_mask, dtype=dtype)

    def _compute_kernel_values(
        self, kernel: Literal["gaussian", "scarches", "random", "inverse_distance"], valid_mask: np.ndarray, **kwargs
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
                f"Unknown kernel: {kernel}. Supported kernels are: 'gaussian', 'scarches', 'random', 'inverse_distance'."
            )

        return connectivities

    def boolean_adjacency(self, dtype=np.float64, set_diag: bool | None = None) -> csr_matrix:
        """
        Construct a boolean adjacency matrix from neighbor indices.

        Parameters
        ----------
        dtype
            Data type for the matrix values.
        set_diag
            If True, set the diagonal to 1. If False, set the diagonal to 0.
            If None, do not modify the diagonal - whether it's 0 or 1 depends on the neighbor algorithm.
            This parameter can only be used with square matrices.

        Returns
        -------
        csr_matrix
            Boolean adjacency matrix (shape: n_samples x n_targets), with 1 for each neighbor relationship.
        """
        # Get valid entries mask (only check indices, not distances)
        valid_mask = self.indices != -1

        # Create array of ones with same shape as indices
        ones = np.ones_like(self.indices, dtype=dtype)

        # Create sparse matrix with ones as values for valid entries
        adj_matrix = self._create_sparse_matrix(ones, valid_mask, dtype=dtype)

        # Handle the diagonal based on set_diag parameter
        if set_diag is not None:
            # Check that we have a square matrix
            if self.shape[0] != self.shape[1]:
                raise ValueError(
                    "The set_diag parameter can only be used with square matrices "
                    f"(got shape {self.shape[0]} x {self.shape[1]})."
                )
            # Use setdiag to efficiently set diagonal elements
            adj_matrix.setdiag(1.0 if set_diag else 0.0)

        return adj_matrix


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
    def from_distances(cls, distances_matrix: "csr_matrix") -> "Neighbors":
        """
        Create a Neighbors object from a pre-computed distances matrix.

        Parameters
        ----------
        distances_matrix
            Sparse distance matrix, typically from adata.obsp['distances']

        Returns
        -------
        Neighbors
            A new Neighbors object with pre-computed neighbor information
        """
        # Extract indices and distances from the sparse matrix
        indices, distances = extract_neighbors_from_distances(distances_matrix)

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

    def get_adjacency_matrices(self) -> tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
        """
        Compute unweighted adjacency matrices for all k-NN graphs.

        Returns
        -------
        tuple
            Unweighted adjacency matrices (xx, yy, xy, yx).
        """
        if self.xx is None or self.yy is None or self.xy is None or self.yx is None:
            raise ValueError("Neighbors must be computed before accessing adjacency matrices.")
        xx_adj = self.xx.boolean_adjacency()
        yy_adj = self.yy.boolean_adjacency()
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
