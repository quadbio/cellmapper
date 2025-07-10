from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix

from cellmapper.logging import logger
from cellmapper.model._knn_backend import get_backend
from cellmapper.model.neighbors_results import NeighborsResults
from cellmapper.utils import extract_neighbors_from_distances


class Neighbors:
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
        self.xx: NeighborsResults | None = None
        self.yy: NeighborsResults | None = None
        self.xy: NeighborsResults | None = None
        self.yx: NeighborsResults | None = None

        # Flag to track if this is a self-mapping case
        # Use explicit parameter if provided, otherwise infer from yrep
        if is_self_mapping is not None:
            self._is_self_mapping = is_self_mapping
        else:
            self._is_self_mapping = yrep is None

    @classmethod
    def from_distances(cls, distances_matrix: csr_matrix, remove_last_neighbor: bool = False) -> "Neighbors":
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
        **kwargs,
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
            If True, only compute the xy neighbors. In self-mapping mode, this is
            automatically set to True for efficiency since all neighbor matrices
            contain the same information.
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

        Examples
        --------
        >>> neighbors = Neighbors(xrep, yrep)
        >>> # sklearn with custom parameters
        >>> neighbors.compute_neighbors(method="sklearn", algorithm="ball_tree", leaf_size=20)
        >>> # pynndescent with custom parameters (inherits scanpy-style defaults)
        >>> neighbors.compute_neighbors(method="pynndescent", n_trees=32, verbose=True)
        """
        # Optimize for self-mapping: only compute yx and reuse for all matrices
        if self._is_self_mapping:
            only_yx = True
            logger.info(
                "Self-mapping mode detected. Computing only yx neighbors for efficiency "
                "(all neighbor matrices will contain the same information)."
            )

        # use strategy pattern to reduce duplication
        logger.info("Using %s to compute %d neighbors.", method, n_neighbors)
        backend_x = get_backend(method, n_neighbors=n_neighbors, metric=metric, random_state=random_state, **kwargs)
        backend_x.fit(self.xrep)

        if only_yx:
            dists, idx = backend_x.query(self.yrep, k=n_neighbors)
            self.yx = NeighborsResults(distances=dists, indices=idx, n_targets=self.xrep.shape[0])
            return

        backend_y = get_backend(method, n_neighbors=n_neighbors, metric=metric, random_state=random_state, **kwargs)
        backend_y.fit(self.yrep)

        x_d, x_i = backend_x.query(self.xrep, k=n_neighbors)
        y_d, y_i = backend_y.query(self.yrep, k=n_neighbors)
        xy_d, xy_i = backend_y.query(self.xrep, k=n_neighbors)
        yx_d, yx_i = backend_x.query(self.yrep, k=n_neighbors)

        self.xx = NeighborsResults(distances=x_d, indices=x_i, n_targets=None)
        self.yy = NeighborsResults(distances=y_d, indices=y_i, n_targets=None)
        self.xy = NeighborsResults(distances=xy_d, indices=xy_i, n_targets=self.yrep.shape[0])
        self.yx = NeighborsResults(distances=yx_d, indices=yx_i, n_targets=self.xrep.shape[0])

    def get_adjacency_matrices(
        self, symmetrize: bool = False, self_edges: bool = True
    ) -> tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
        """
        Compute unweighted adjacency matrices for all k-NN graphs.

        Parameters
        ----------
        symmetrize
            If True, make self-terms (xx, yy) symmetrize. Cross-terms (xy, yx) are not affected.
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
        The symmetrize and self_edges parameters only apply to self-terms (xx, yy) since
        these represent within-dataset neighborhoods. Cross-terms (xy, yx) represent
        between-dataset relationships where symmetry and self-edges are not meaningful.
        """
        if self._is_self_mapping:
            if self.yx is None:
                raise ValueError("Neighbors must be computed before accessing adjacency matrices.")
            yx_adj = self.yx.boolean_adjacency(self_edges=self_edges, symmetrize=symmetrize)
            xx_adj = yx_adj
            yy_adj = yx_adj
            xy_adj = yx_adj
        else:
            if self.xx is None or self.yy is None or self.xy is None or self.yx is None:
                raise ValueError("Neighbors must be computed before accessing adjacency matrices.")

            # self-terms (within-dataset neighborhoods)
            xx_adj = self.xx.boolean_adjacency(self_edges=self_edges, symmetrize=symmetrize)
            yy_adj = self.yy.boolean_adjacency(self_edges=self_edges, symmetrize=symmetrize)

            # Cross-terms (between-dataset neighborhoods)
            xy_adj = self.xy.boolean_adjacency(self_edges=False, symmetrize=False)
            yx_adj = self.yx.boolean_adjacency(self_edges=False, symmetrize=False)

        return xx_adj, yy_adj, xy_adj, yx_adj

    def __repr__(self):
        """Return a string representation of the Neighbors object."""
        return (
            f"Neighbors(xrep_shape={self.xrep.shape}, yrep_shape={self.yrep.shape}, "
            f"xx={self.xx is not None}, yy={self.yy is not None}, "
            f"xy={self.xy is not None}, yx={self.yx is not None}, "
            f"self_mapping={self._is_self_mapping})"
        )
