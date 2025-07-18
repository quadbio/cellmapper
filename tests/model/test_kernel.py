import numpy as np
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix

from cellmapper.model._knn_backend import _BACKENDS, _FaissCpuBackend, get_backend
from cellmapper.model.kernel import Kernel


def assert_adjacency_equal(neigh1, neigh2, attrs=("xx", "yy", "xy", "yx")):
    for attr in attrs:
        adj1 = getattr(neigh1, attr).boolean_adjacency().toarray()
        adj2 = getattr(neigh2, attr).boolean_adjacency().toarray()
        assert np.allclose(adj1, adj2), f"Adjacency matrices ({attr}) differ between methods"


class TestKernel:
    @pytest.mark.parametrize("only_yx", [False, True])
    def test_kernel_sklearn_vs_pynndescent(self, small_data, only_yx):
        pytest.importorskip("pynndescent")
        x, y = small_data
        n_neighbors = 3
        # sklearn
        neigh_skl = Kernel(x, y)
        neigh_skl.compute_neighbors(n_neighbors=n_neighbors, knn_method="sklearn", only_yx=only_yx)
        # pynndescent
        neigh_pynn = Kernel(x, y)
        neigh_pynn.compute_neighbors(n_neighbors=n_neighbors, knn_method="pynndescent", only_yx=only_yx)
        if only_yx:
            with pytest.raises(ValueError):
                neigh_skl.get_adjacency_matrices()
            with pytest.raises(ValueError):
                neigh_pynn.get_adjacency_matrices()
        else:
            assert_adjacency_equal(neigh_skl, neigh_pynn)
        # Always compare connectivities (yx)
        conn_skl = neigh_skl.yx.knn_graph_connectivities()
        conn_pynn = neigh_pynn.yx.knn_graph_connectivities()
        assert np.allclose(conn_skl.toarray(), conn_pynn.toarray(), atol=1e-6)

    def test_kernel_faiss_backend_availability(self):
        """Test that faiss-cpu backend is available and can be imported."""
        pytest.importorskip("faiss")

        assert "faiss-cpu" in _BACKENDS
        assert "faiss-gpu" in _BACKENDS

        # This should not raise an exception
        backend = get_backend(knn_method="faiss-cpu", n_neighbors=3, metric="euclidean")
        assert backend is not None

        # Verify it's the right type
        assert isinstance(backend, _FaissCpuBackend)

    def test_kernel_repr(self, small_data):
        x, y = small_data
        neigh = Kernel(x, y)
        r = repr(neigh)
        assert "Kernel(" in r and "xrep_shape" in r and "yrep_shape" in r
        neigh.compute_neighbors(n_neighbors=2, knn_method="sklearn")
        r2 = repr(neigh)
        assert "xx=True" in r2 and "yy=True" in r2 and "xy=True" in r2 and "yx=True" in r2

    def test_from_distances_factory_method(self, small_data):
        """Test the from_distances factory method for creating Neighbors from a distance matrix."""
        # Create a basic distance matrix
        n_samples = 5
        distances_data = np.zeros((n_samples, n_samples))
        # Fill with distance values (using Euclidean distance for simplicity)
        x, _ = small_data
        for i in range(n_samples):
            for j in range(n_samples):
                distances_data[i, j] = np.linalg.norm(x[i] - x[j])
        distances = csr_matrix(distances_data)

        # Create Neighbors object using factory method
        neighbors = Kernel.from_distances(distances)

        # Verify the object is created correctly
        assert neighbors is not None
        assert neighbors._is_self_mapping
        assert neighbors.xx is not None
        assert neighbors.yy is not None
        assert neighbors.xy is not None
        assert neighbors.yx is not None

        # All neighbor matrices should be identical for self-mapping
        assert neighbors.xx is neighbors.yy
        assert neighbors.xx is neighbors.xy
        assert neighbors.xx is neighbors.yx

        # Test with the neighbors results
        nn_results = neighbors.xx
        assert nn_results.n_samples == n_samples
        assert (
            nn_results.n_neighbors + 1 == n_samples
        )  # the distance to self is exactly 0, so it won't be included as a neighbor

        # Get adjacency matrix and verify it reflects the original distances
        adj_matrix = nn_results.knn_graph_distances
        assert adj_matrix.shape == (n_samples, n_samples)
        # Check that diagonal is zero (self-distance)
        assert np.allclose(adj_matrix.diagonal(), 0)

    @pytest.mark.parametrize("n_neighbors", [2, 3, 5])
    def test_from_distances_varying_neighbors(self, n_neighbors):
        """Test from_distances with distance matrices containing varying numbers of neighbors."""
        n_samples = 10
        # Create distance matrix with only n_neighbors per cell
        rows, cols, data = [], [], []

        for i in range(n_samples):
            # Add diagonal element (self)
            rows.append(i)
            cols.append(i)
            data.append(0.0)

            # Add n_neighbors-1 other neighbors with increasing distances
            for j in range(1, n_neighbors):
                neighbor_idx = (i + j) % n_samples  # Simple circular pattern
                rows.append(i)
                cols.append(neighbor_idx)
                data.append(float(j))  # Distance increases with j

        distances = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

        # Create Neighbors object
        neighbors = Kernel.from_distances(distances)

        # Check the number of neighbors
        assert neighbors.xx.n_samples == n_samples
        # After initialization, self-edges are removed, so n_neighbors = n_neighbors - 1
        assert neighbors.xx.n_neighbors == n_neighbors - 1

        # Verify the connectivities
        connectivities = neighbors.xx.knn_graph_connectivities(self_edges=True)
        assert connectivities.shape == (n_samples, n_samples)
        assert np.count_nonzero(connectivities.toarray()[0]) == n_neighbors

    def test_from_distances_with_spatial_structure(self, adata_spatial):
        """Test from_distances with a realistic spatial structure."""

        # Compute standard scanpy neighbors
        sc.pp.neighbors(adata_spatial, n_neighbors=10, use_rep="X_pca")

        # Extract the distance matrix
        distances = adata_spatial.obsp["distances"]

        # Create Neighbors object from distances
        neighbors = Kernel.from_distances(distances)

        # Verify basic properties
        assert neighbors is not None
        assert neighbors.xx.n_samples == adata_spatial.n_obs

        # Compare kNN graphs
        knn_graph = neighbors.xx.knn_graph_connectivities()
        original_conn = adata_spatial.obsp["connectivities"]

        # The graphs might not be identical, but should have similar properties
        assert knn_graph.shape == original_conn.shape
        assert np.isclose(knn_graph.sum(), original_conn.sum(), rtol=0.1)

    @pytest.mark.parametrize("kernel", ["gauss", "scarches", "inverse_distance"])
    def test_from_distances_different_kernels(self, kernel):
        """Test different kernels with distances from Neighbors.from_distances."""
        n_samples = 10
        # Create a simple distance matrix
        distances_data = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    distances_data[i, j] = abs(i - j)  # Simple metric: difference in indices

        distances = csr_matrix(distances_data)
        print(distances)

        # Create Neighbors object
        neighbors = Kernel.from_distances(distances)

        # Compute connectivities with different kernels
        connectivities = neighbors.xx.knn_graph_connectivities(kernel=kernel, self_edges=True)

        # Basic checks
        assert connectivities.shape == (n_samples, n_samples)
        # All kernels should produce positive values
        assert np.all(connectivities.data > 0)
        # Diagonal should be large values (except for random kernel)
        if kernel != "random":
            diag_values = connectivities.diagonal()
            # Diagonal elements should typically be the largest for each row
            for i in range(n_samples):
                row = connectivities.getrow(i).toarray().flatten()
                assert diag_values[i] >= np.max(row) or np.isclose(diag_values[i], np.max(row))
