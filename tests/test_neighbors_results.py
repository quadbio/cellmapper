import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cellmapper.knn import NeighborsResults


class TestNeighborsResults:
    def test_neighborsresults_init_shape(self, sample_distances, sample_indices):
        # Should not raise
        nr = NeighborsResults(distances=sample_distances, indices=sample_indices)
        assert nr.n_samples == 3
        assert nr.n_neighbors == 2
        assert nr.shape == (3, 3)

    def test_neighborsresults_invalid_shape(self, sample_distances):
        # indices shape mismatch should raise
        bad_indices = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        with pytest.raises(ValueError):
            NeighborsResults(distances=sample_distances, indices=bad_indices)

    def test_knn_graph_distances(self, sample_distances, sample_indices):
        nr = NeighborsResults(distances=sample_distances, indices=sample_indices)
        mat = nr.knn_graph_distances
        assert isinstance(mat, csr_matrix)
        assert mat.shape == (3, 3)
        # Check that diagonal is zero (self-distance)
        assert np.allclose(mat.diagonal(), 0)

    @pytest.mark.parametrize("kernel", ["gaussian", "scarches", "random", "inverse_distance"])
    def test_knn_graph_connectivities_kernels(self, sample_distances, sample_indices, kernel):
        nr = NeighborsResults(distances=sample_distances, indices=sample_indices)
        mat = nr.knn_graph_connectivities(kernel=kernel)
        assert isinstance(mat, csr_matrix)
        assert mat.shape == (3, 3)
        # All values should be > 0 for all kernels except 'random', which can be 0
        if kernel != "random":
            assert np.all(mat.data > 0)

    def test_boolean_adjacency(self, sample_distances, sample_indices):
        nr = NeighborsResults(distances=sample_distances, indices=sample_indices)
        mat = nr.boolean_adjacency()
        assert isinstance(mat, csr_matrix)
        assert mat.shape == (3, 3)
        # All nonzero values should be 1
        assert np.all(mat.data == 1)
