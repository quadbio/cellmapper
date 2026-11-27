import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cellmapper.model.neighbors import Neighbors


class TestNeighbors:
    def test_neighbors_init_shape(self, sample_distances, sample_indices):
        # Should not raise
        nr = Neighbors(distances=sample_distances, indices=sample_indices)
        assert nr.n_samples == 3
        # After initialization, self-edges are removed for square matrices, so n_neighbors = 1
        assert nr.n_neighbors == 1
        assert nr.shape == (3, 3)

    def test_neighbors_invalid_shape(self, sample_distances):
        # indices shape mismatch should raise
        bad_indices = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        with pytest.raises(ValueError):
            Neighbors(distances=sample_distances, indices=bad_indices)

    def test_knn_graph_distances(self, sample_distances, sample_indices):
        nr = Neighbors(distances=sample_distances, indices=sample_indices)
        mat = nr.knn_graph_distances
        assert isinstance(mat, csr_matrix)
        assert mat.shape == (3, 3)
        # Check that diagonal is zero (self-distance)
        assert np.allclose(mat.diagonal(), 0)

    @pytest.mark.parametrize("kernel", ["gauss", "scarches", "random", "inverse_distance"])
    def test_knn_graph_connectivities_kernels(self, sample_distances, sample_indices, kernel):
        nr = Neighbors(distances=sample_distances, indices=sample_indices)
        mat = nr.knn_graph_connectivities(kernel=kernel)
        assert isinstance(mat, csr_matrix)
        assert mat.shape == (3, 3)
        # All values should be > 0 for all kernels except 'random', which can be 0
        if kernel != "random":
            assert np.all(mat.data > 0)

    def test_boolean_adjacency(self, sample_distances, sample_indices):
        nr = Neighbors(distances=sample_distances, indices=sample_indices)
        mat = nr.boolean_adjacency()
        assert isinstance(mat, csr_matrix)
        assert mat.shape == (3, 3)
        # All nonzero values should be 1
        assert np.all(mat.data == 1)
