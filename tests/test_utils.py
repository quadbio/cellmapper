import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cellmapper.utils import extract_neighbors_from_distances


class TestExtractNeighborsFromDistances:
    """Tests for the extract_neighbors_from_distances function."""

    def test_basic_extraction(self):
        """Test basic extraction from a simple distance matrix."""
        # Create a simple distance matrix
        distances_data = np.array(
            [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 4.0, 5.0], [2.0, 4.0, 0.0, 6.0], [3.0, 5.0, 6.0, 0.0]]
        )
        distances = csr_matrix(distances_data)

        # Extract neighbors
        indices, nbr_distances = extract_neighbors_from_distances(distances)

        # Check shapes
        assert indices.shape == (4, 4)
        assert nbr_distances.shape == (4, 4)

        # Check values - neighbors should be sorted by distance
        np.testing.assert_array_equal(indices[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(indices[1], [1, 0, 2, 3])
        np.testing.assert_array_almost_equal(nbr_distances[0], [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(nbr_distances[1], [0.0, 1.0, 4.0, 5.0])

    def test_sparse_matrix(self):
        """Test extraction from a sparse distance matrix."""
        # Create a sparse matrix with only some connections
        row_idx = [0, 0, 1, 1, 2, 2, 3, 3]
        col_idx = [0, 1, 1, 0, 2, 0, 3, 2]
        values = [0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 6.0]
        distances = csr_matrix((values, (row_idx, col_idx)), shape=(4, 4))

        # Extract neighbors
        indices, nbr_distances = extract_neighbors_from_distances(distances)

        # Check shapes
        assert indices.shape == (4, 2)  # Only 2 neighbors per cell
        assert nbr_distances.shape == (4, 2)

        # Check values
        np.testing.assert_array_equal(indices[0], [0, 1])
        np.testing.assert_array_equal(indices[2], [2, 0])
        np.testing.assert_array_almost_equal(nbr_distances[0], [0.0, 1.0])
        np.testing.assert_array_almost_equal(nbr_distances[2], [0.0, 2.0])

    def test_invalid_input(self):
        """Test with invalid inputs."""
        # Test with non-square matrix
        distances = csr_matrix((3, 4))
        with pytest.raises(ValueError, match="Square distance matrix"):
            extract_neighbors_from_distances(distances)

        # Test with non-sparse matrix
        with pytest.raises(TypeError, match="must be a sparse matrix"):
            extract_neighbors_from_distances(np.zeros((3, 3)))

    def test_empty_matrix(self):
        """Test with an empty matrix."""
        distances = csr_matrix((0, 0))
        indices, nbr_distances = extract_neighbors_from_distances(distances)
        assert indices.size == 0
        assert nbr_distances.size == 0

    def test_infinite_values(self):
        """Test with a matrix containing infinite values."""
        # Create a distance matrix with some infinite values
        distances_data = np.array(
            [[0.0, 1.0, np.inf, 3.0], [1.0, 0.0, 4.0, np.inf], [np.inf, 4.0, 0.0, 6.0], [3.0, np.inf, 6.0, 0.0]]
        )
        distances = csr_matrix(distances_data)

        # Extract neighbors
        indices, nbr_distances = extract_neighbors_from_distances(distances)

        # Check shapes - should exclude infinite values
        assert indices.shape == (4, 3)  # One less neighbor per cell
        assert nbr_distances.shape == (4, 3)

        # Check that infinite values are excluded
        for i in range(4):
            assert np.all(np.isfinite(nbr_distances[i]))

    def test_include_self_parameter(self):
        """Test the include_self parameter to control self-connections."""
        # Create a distance matrix with self-connections (diagonal = 0)
        distances_data = np.array(
            [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 4.0, 5.0], [2.0, 4.0, 0.0, 6.0], [3.0, 5.0, 6.0, 0.0]]
        )
        distances = csr_matrix(distances_data)

        # Test with include_self=True (default)
        indices_with_self, distances_with_self = extract_neighbors_from_distances(distances, include_self=True)

        # Test with include_self=False
        indices_without_self, distances_without_self = extract_neighbors_from_distances(distances, include_self=False)

        # With include_self=True, diagonal elements should be included
        # (self should be the first neighbor because distance is 0)
        for i in range(4):
            assert indices_with_self[i, 0] == i
            assert distances_with_self[i, 0] == 0.0

        # With include_self=False, diagonal elements should be excluded
        for i in range(4):
            # The self-index should not be in the neighbors
            assert i not in indices_without_self[i]
            # Check that no zero distances are present
            assert np.all(distances_without_self[i] > 0)
