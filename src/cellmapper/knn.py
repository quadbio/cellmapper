from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import sklearn.neighbors
from pynndescent import NNDescent
from scipy.sparse import csr_matrix

from cellmapper.logging import logger

from .check import check_deps


@dataclass
class NeighborsResults:
    """Nearest neighbors results data store.

    Adapted from the scib-metrics package: https://github.com/YosefLab/scib-metrics

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
        rowptr = np.arange(0, self.n_samples * self.n_neighbors + 1, self.n_neighbors)
        return csr_matrix((self.distances.ravel().astype(dtype), self.indices.ravel(), rowptr), shape=self.shape)

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
        if kernel == "gaussian":
            sigma = np.mean(self.distances)
            connectivities = np.exp(-(self.distances**2) / (2 * sigma**2))
        elif kernel == "scarches":
            sigma = np.std(self.distances)
            sigma = (2.0 / sigma) ** 2
            connectivities = np.exp(-self.distances / sigma)
        elif kernel == "random":  # this is for testing purposes
            connectivities = np.random.rand(*self.distances.shape)  # Random values for connectivities
        elif kernel == "inverse_distance":
            epsilon = kwargs.get("epsilon", 1e-8)
            connectivities = 1.0 / (self.distances + epsilon)
        else:
            raise ValueError(f"Unknown kernel: {kernel}. Supported kernels are 'gaussian' and 'scarches'.")
        rowptr = np.arange(0, self.n_samples * self.n_neighbors + 1, self.n_neighbors)
        return csr_matrix((connectivities.ravel().astype(dtype), self.indices.ravel(), rowptr), shape=self.shape)

    def boolean_adjacency(self, dtype=np.float64) -> csr_matrix:
        """
        Construct a boolean adjacency matrix from neighbor indices using the same rowptr mechanism as other graph methods.

        Parameters
        ----------
        dtype
            Data type for the matrix values.

        Returns
        -------
        csr_matrix
            Boolean adjacency matrix (shape: n_samples x n_targets), with 1 for each neighbor relationship.
        """
        rowptr = np.arange(0, self.n_samples * self.n_neighbors + 1, self.n_neighbors)
        data = np.ones(self.indices.size, dtype=dtype)
        return csr_matrix((data, self.indices.ravel(), rowptr), shape=self.shape)


class Neighbors:
    """Class to compute and store nearest neighbors."""

    def __init__(self, xrep: np.ndarray, yrep: np.ndarray):
        """
        Initialize the Neighbors class.

        Parameters
        ----------
        xrep
            Representation of the reference dataset.
        yrep
            Representation of the query dataset.
        """
        self.xrep = xrep
        self.yrep = yrep

        self.xx: NeighborsResults | None = None
        self.yy: NeighborsResults | None = None
        self.xy: NeighborsResults | None = None
        self.yx: NeighborsResults | None = None

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
        """
        if method in ["rapids", "sklearn", "pynndescent", "faiss"]:
            logger.info("Using %s to compute %d neighbors.", method, n_neighbors)

            if method == "rapids":
                check_deps("rapids")
                import cuml as cm
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
            f"xy={self.xy is not None}, yx={self.yx is not None})"
        )
