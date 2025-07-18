from typing import Literal, cast

import numpy as np
from scipy.sparse import csr_matrix

from cellmapper._docs import d
from cellmapper.constants import PackageConstants
from cellmapper.logging import logger
from cellmapper.model._knn_backend import get_backend
from cellmapper.model.neighbors import Neighbors
from cellmapper.utils import extract_neighbors_from_distances


class Kernel:
    """Class to compute and store nearest neighbors."""

    def __init__(self, xrep: np.ndarray, yrep: np.ndarray | None = None, is_self_mapping: bool | None = None):
        """
        Initialize the Neighbors class.

        Parameters
        ----------
        xrep
            Representation of the reference dataset.
        yrep
            Representation of the query dataset. If None, self-mapping will be used.
        is_self_mapping
            Explicitly specify if this is a self-mapping case. If None, will be inferred
            from whether yrep is None.
        """
        self.xrep = xrep
        # Use xrep for self-mapping if yrep is None
        self.yrep = yrep if yrep is not None else xrep

        # Initialize neighbor result containers
        self.xx: Neighbors | None = None
        self.yy: Neighbors | None = None
        self.xy: Neighbors | None = None
        self.yx: Neighbors | None = None

        # Initialize kernel matrix storage
        self.kernel_matrix: csr_matrix | None = None
        self.kernel_method: str | None = None
        self.is_symmetric: bool | None = None

        # Flag to track if this is a self-mapping case
        # Use explicit parameter if provided, otherwise infer from yrep
        if is_self_mapping is not None:
            self._is_self_mapping = is_self_mapping
        else:
            self._is_self_mapping = yrep is None

        # Store only_yx flag set during compute_neighbors
        self.only_yx: bool | None = None

    @classmethod
    def from_distances(cls, distances_matrix: csr_matrix, remove_last_neighbor: bool = False) -> "Kernel":
        """
        Create a Neighbors object from a pre-computed distances matrix.

        Parameters
        ----------
        distances_matrix
            Sparse distance matrix, typically from adata.obsp['distances']
        remove_last_neighbor
            If True, removes the last neighbor from the distances matrix.
            This is useful for direct comparisons with scanpy, which uses a
            different convention for neighbor counts.

        Returns
        -------
        Neighbors
            A new Neighbors object with pre-computed neighbor information.
            Self-edge handling is performed automatically by NeighborsResults during initialization.
        """
        assert distances_matrix is not None, "distances_matrix must be provided"
        # Extract indices and distances from the sparse matrix
        indices, distances = extract_neighbors_from_distances(distances_matrix)

        if remove_last_neighbor:
            # Remove the last neighbor (last column) from indices and distances
            indices = indices[:, :-1]
            distances = distances[:, :-1]

            logger.info("Removed last neighbor from distances matrix for compatibility with scanpy conventions.")

        # Create a minimal Neighbors object for self-mapping
        n_cells = distances_matrix.shape[0]  # type: ignore
        placeholder_rep = np.zeros((n_cells, 1))
        neighbors = cls(xrep=placeholder_rep)

        # Create a NeighborsResults object with the extracted data
        neighbors_result = Neighbors(distances=distances, indices=indices)

        # For self-mapping, all neighbor objects should be the same
        neighbors.xx = neighbors_result
        neighbors.yy = neighbors_result
        neighbors.xy = neighbors_result
        neighbors.yx = neighbors_result

        # Mark as self-mapping
        neighbors._is_self_mapping = True

        logger.info("Created Neighbors object from distances matrix with %d cells", n_cells)

        return neighbors

    @d.dedent
    def compute_neighbors(
        self,
        n_neighbors: int = 30,
        knn_method: Literal["sklearn", "pynndescent", "rapids", "faiss-cpu", "faiss-gpu"] = "sklearn",
        knn_dist_metric: str = "euclidean",
        random_state: int = 0,
        only_yx: bool = False,
        **kwargs,
    ):
        """
        Compute nearest neighbors using either sklearn or rapids.

        Parameters
        ----------
         n_neighbors
            Number of nearest neighbors.
        %(knn_method)s
        %(knn_dist_metric)s
        random_state
            Random state for reproducibility.
        %(only_yx)s
        **kwargs
            Additional keyword arguments to pass to the underlying k-NN algorithm.
            These are method-specific and will be passed directly to the algorithm's
            constructor or fitting method.

            For pynndescent, scanpy-style defaults are applied:
            - n_jobs: -1 (use all CPU cores)
            - n_trees: min(64, 5 + round(n_samples^0.5 / 20.0)) (per dataset)
            - n_iters: max(5, round(log2(n_samples))) (per dataset)

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

        In self-mapping mode, all four matrices will reference the same NeighborsResults
        object for memory efficiency.
        """
        # Optimize for self-mapping: only compute yx and reuse for all matrices
        if self._is_self_mapping:
            only_yx = True
            logger.info(
                "Self-mapping mode detected. Computing only yx neighbors for efficiency "
                "(all neighbor matrices will contain the same information)."
            )

        # Store only_yx as instance attribute for use in compute_kernel_matrix
        self.only_yx = only_yx

        # issue a warning if using sklearn with large datasets
        if knn_method == "sklearn" and (
            self.xrep.shape[0] > PackageConstants.SKLEARN_WARNING_CUTOFF
            or self.yrep.shape[0] > PackageConstants.SKLEARN_WARNING_CUTOFF
        ):
            logger.warning(
                "Using sklearn for neighbor search with large dataset (%d cells). "
                "Consider using approximate k-NN search (e.g. pynndescent) or GPU acceleration (e.g. faiss or rapids)",
                self.xrep.shape[0],
            )

        # use strategy pattern to reduce duplication
        logger.info("Using %s to compute %d neighbors.", knn_method, n_neighbors)
        backend_x = get_backend(
            knn_method, n_neighbors=n_neighbors, metric=knn_dist_metric, random_state=random_state, **kwargs
        )
        backend_x.fit(self.xrep)

        if only_yx:
            dists, idx = backend_x.query(self.yrep, k=n_neighbors)
            self.yx = Neighbors(distances=dists, indices=idx, n_targets=self.xrep.shape[0])
            return

        backend_y = get_backend(
            knn_method, n_neighbors=n_neighbors, metric=knn_dist_metric, random_state=random_state, **kwargs
        )
        backend_y.fit(self.yrep)

        x_d, x_i = backend_x.query(self.xrep, k=n_neighbors)
        y_d, y_i = backend_y.query(self.yrep, k=n_neighbors)
        xy_d, xy_i = backend_y.query(self.xrep, k=n_neighbors)
        yx_d, yx_i = backend_x.query(self.yrep, k=n_neighbors)

        self.xx = Neighbors(distances=x_d, indices=x_i, n_targets=None)
        self.yy = Neighbors(distances=y_d, indices=y_i, n_targets=None)
        self.xy = Neighbors(distances=xy_d, indices=xy_i, n_targets=self.yrep.shape[0])
        self.yx = Neighbors(distances=yx_d, indices=yx_i, n_targets=self.xrep.shape[0])

    @d.dedent
    def compute_kernel_matrix(
        self,
        kernel_method: Literal[
            "jaccard",
            "gauss",
            "scarches",
            "inverse_distance",
            "random",
            "hnoca",
            "equal",
            "umap",
        ],
        symmetrize: bool = False,
        symmetrize_method: Literal["max", "mean"] = "max",
        self_edges: bool = False,
        **kwargs,
    ) -> None:
        """
        Compute the kernel matrix using the specified method.

        Parameters
        ----------
        %(kernel_method)s
        %(symmetrize)s
        symmetrize_method
            Method for symmetrization when symmetrize=True:
            - "max": Take element-wise maximum between matrix and transpose (preserves strongest connections)
            - "mean": Take element-wise average between matrix and transpose (smooths connections)
        %(self_edges)s
        **kwargs
            Additional keyword arguments for kernel computation.

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``kernel_matrix``: The computed kernel matrix.
        - ``kernel_method``: The method used to compute the kernel.
        - ``is_symmetric``: Whether the resulting matrix is symmetric (self-mapping only).

        The method uses the `only_yx` attribute set during `compute_neighbors` to determine
        which neighbors were computed and validate method compatibility.
        """
        if kernel_method in PackageConstants.JACCARD_BASED_KERNELS:
            # In cross-mapping mode, we need all four adjacency matrices
            if self.only_yx and not self._is_self_mapping:
                raise ValueError(
                    "Jaccard and HNOCA methods require both x and y neighbors to be computed in cross-mapping mode. Set only_yx=False."
                )  # Get adjacency matrices (self_edges=True for jaccard/hnoca computation)
            xx, yy, xy, yx = self.get_adjacency_matrices(self_edges=True)

            # Get number of neighbors for normalization
            assert self.yx is not None, "yx neighbors must be computed"
            n_neighbors = self.yx.n_neighbors

            # Compute kernel matrix
            kernel_matrix = (yx @ xx.T) + (yy @ xy.T)

            if kernel_method == "jaccard":
                kernel_matrix.data /= 4 * n_neighbors - kernel_matrix.data
            elif kernel_method == "hnoca":
                kernel_matrix.data /= 2 * n_neighbors - kernel_matrix.data
                kernel_matrix.data = kernel_matrix.data**2

        elif kernel_method in PackageConstants.CONNECTIVITY_BASED_KERNELS:
            # Validate self-mapping-only kernels
            if kernel_method in PackageConstants.SELF_MAPPING_ONLY_KERNELS and not self._is_self_mapping:
                raise ValueError(f"Method '{kernel_method}' is only supported for self-mapping mode.")

            # Type cast to satisfy the type checker
            kernel_method = cast(
                Literal["gauss", "scarches", "inverse_distance", "random", "equal", "umap"],
                kernel_method,
            )

            # Use yx neighbors to compute kernel
            assert self.yx is not None, "yx neighbors must be computed"
            kernel_matrix = self.yx.knn_graph_connectivities(kernel=kernel_method, self_edges=self_edges, **kwargs)
        else:
            raise NotImplementedError(f"Method '{kernel_method}' is not implemented.")

        # Apply symmetrization if requested and matrix is square
        if symmetrize and self._is_self_mapping:
            kernel_matrix = self._symmetrize_matrix(kernel_matrix, method=symmetrize_method)
        elif symmetrize and not self._is_self_mapping:
            raise ValueError("Symmetrization is only supported for self-mapping (square matrices).")

        # Store the computed kernel matrix and metadata
        self.kernel_matrix = kernel_matrix
        self.kernel_method = kernel_method

        # Check if the resulting matrix is symmetric (for self-mapping only)
        if self._is_self_mapping:
            self.is_symmetric = self._check_symmetry(kernel_matrix)
        else:
            self.is_symmetric = None

    def _symmetrize_matrix(self, sparse_matrix: csr_matrix, method: str = "max") -> csr_matrix:
        """
        Apply symmetrization to a sparse kernel matrix.

        Parameters
        ----------
        sparse_matrix
            Input sparse kernel matrix to symmetrize.
        method
            Method for symmetrization:
            - "max": Take element-wise maximum between matrix and transpose.
              For each position (i,j), W_sym[i,j] = max(W[i,j], W[j,i]).
            - "mean": Take element-wise average between matrix and transpose.
              For each position (i,j), W_sym[i,j] = (W[i,j] + W[j,i]) / 2.

        Returns
        -------
        csr_matrix
            Symmetrized sparse matrix where W[i,j] = W[j,i] for all positions.

        Notes
        -----
        Only applies to square matrices (self-mapping).

        The "max" method preserves the strongest connections and ensures that
        existing edges are not weakened. The "mean" method averages the weights
        and can provide smoother transitions but may weaken strong connections.
        """
        if not self._is_self_mapping:
            raise ValueError("Can only symmetrize square matrices (self-mapping)")

        if method == "max":
            # Take element-wise maximum between matrix and its transpose
            return sparse_matrix.maximum(sparse_matrix.T)
        elif method == "mean":
            # Take element-wise average: (M + M^T) / 2
            return (sparse_matrix + sparse_matrix.T) / 2
        else:
            raise ValueError(f"Unknown symmetrization method: {method}. Use 'max' or 'mean'.")

    @d.dedent
    def get_adjacency_matrices(self, self_edges: bool = True) -> tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
        """
        Compute unweighted adjacency matrices for all k-NN graphs.

        Parameters
        ----------
        %(self_edges)s

        Returns
        -------
        tuple
            Unweighted adjacency matrices (xx, yy, xy, yx).

        Notes
        -----
        The self_edges parameter only applies to self-terms (xx, yy) since
        these represent within-dataset neighborhoods. Cross-terms (xy, yx) represent
        between-dataset relationships where self-edges are not meaningful.
        """
        if self._is_self_mapping:
            if self.yx is None:
                raise ValueError("Neighbors must be computed before accessing adjacency matrices.")
            yx_adj = self.yx.boolean_adjacency(self_edges=self_edges)
            xx_adj = yx_adj
            yy_adj = yx_adj
            xy_adj = yx_adj
        else:
            if self.xx is None or self.yy is None or self.xy is None or self.yx is None:
                raise ValueError("Neighbors must be computed before accessing adjacency matrices.")

            # self-terms (within-dataset neighborhoods)
            xx_adj = self.xx.boolean_adjacency(self_edges=self_edges)
            yy_adj = self.yy.boolean_adjacency(self_edges=self_edges)

            # Cross-terms (between-dataset neighborhoods)
            xy_adj = self.xy.boolean_adjacency(self_edges=False)
            yx_adj = self.yx.boolean_adjacency(self_edges=False)

        return xx_adj, yy_adj, xy_adj, yx_adj

    def _check_symmetry(self, sparse_matrix: csr_matrix) -> bool:
        """
        Check if a sparse matrix is symmetric.

        Parameters
        ----------
        sparse_matrix
            Input sparse matrix to check for symmetry.

        Returns
        -------
        bool
            True if the matrix is symmetric, False otherwise.
        """
        # For sparse matrices, check if the difference from transpose has any nonzero entries
        diff = sparse_matrix - sparse_matrix.T
        return diff.nnz == 0

    def __repr__(self):
        """Return a string representation of the Kernel object."""
        # Basic info
        info_parts = [
            f"xrep_shape={self.xrep.shape}",
            f"yrep_shape={self.yrep.shape}",
            f"self_mapping={self._is_self_mapping}",
        ]

        # Neighbors info
        neighbors_computed = [
            f"xx={self.xx is not None}",
            f"yy={self.yy is not None}",
            f"xy={self.xy is not None}",
            f"yx={self.yx is not None}",
        ]

        # Kernel matrix info
        if self.kernel_matrix is not None:
            # Calculate sparsity percentage
            total_elements = self.kernel_matrix.shape[0] * self.kernel_matrix.shape[1]
            sparsity = self.kernel_matrix.nnz / total_elements

            kernel_info = [
                f"kernel='{self.kernel_method}'",
                f"matrix_shape={self.kernel_matrix.shape}",
                f"sparsity={sparsity:.1%}",
            ]
            if self.is_symmetric is not None:
                kernel_info.append(f"symmetric={self.is_symmetric}")
        else:
            kernel_info = ["kernel=None"]

        all_info = info_parts + neighbors_computed + kernel_info
        return f"Kernel({', '.join(all_info)})"
