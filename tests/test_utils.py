import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cellmapper.utils import extract_neighbors_from_distances, truncated_svd_cross_covariance


class TestExtractNeighborsFromDistances:
    """Tests for the extract_neighbors_from_distances function."""

    @pytest.mark.parametrize("include_self", [None, True, False])
    def test_basic_extraction(self, include_self):
        """Test basic extraction from a simple distance matrix."""
        # Create a simple distance matrix
        distances_data = np.array(
            [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 4.0, 5.0], [2.0, 4.0, 0.0, 6.0], [3.0, 5.0, 6.0, 0.0]]
        )
        distances = csr_matrix(distances_data)

        # Extract neighbors with specified include_self parameter
        indices, nbr_distances = extract_neighbors_from_distances(distances, include_self=include_self)

        # Determine expected shapes and content based on include_self
        if include_self is False or include_self is None:
            # Without self, each row has 3 neighbors
            expected_shape = (4, 3)
            # First neighbors should be the closest non-self neighbors
            expected_first_neighbors = [1, 0, 0, 0]
        else:
            # With self (or default behavior), each row has 4 neighbors
            expected_shape = (4, 4)
            # First neighbors should be self (distance 0)
            expected_first_neighbors = [0, 1, 2, 3]

        # Check shapes
        assert indices.shape == expected_shape
        assert nbr_distances.shape == expected_shape

        # Check first neighbors for each cell
        for i, expected in enumerate(expected_first_neighbors):
            assert indices[i, 0] == expected, f"Cell {i}'s first neighbor should be {expected}"

        # Check all distances are sorted
        for i in range(4):
            assert np.all(np.diff(nbr_distances[i, :]) >= 0), f"Distances for cell {i} should be sorted"

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
        """Test that infinite values are preserved in the neighbor matrix."""
        # Create a distance matrix with some infinite values
        distances_data = np.array(
            [[0.0, 1.0, np.inf, 3.0], [1.0, 0.0, 4.0, np.inf], [np.inf, 4.0, 0.0, 6.0], [3.0, np.inf, 6.0, 0.0]]
        )
        distances = csr_matrix(distances_data)

        # Extract neighbors with include_self=True
        indices, nbr_distances = extract_neighbors_from_distances(distances, include_self=True)

        # Check for a specific cell with infinite distance
        row0_neighbors = indices[0]
        assert 2 in row0_neighbors, "Neighbor with infinite distance should be included"

        # Find where the infinite value is in the results
        idx = np.where(row0_neighbors == 2)[0][0]

        # Verify the distance is infinite
        assert np.isinf(nbr_distances[0, idx]), "Distance should be infinite"

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


class TestTruncatedSVDCrossCovariance:
    """Tests for the truncated_svd_cross_covariance function."""

    @pytest.mark.parametrize(
        "sparse,zero_center,implicit",
        [
            (False, True, True),
            (False, True, False),
            (False, False, True),
            (False, False, False),
            (True, True, True),
            (True, True, False),
            (True, False, True),
            (True, False, False),
        ],
    )
    def test_dense_vs_sparse_and_centering(self, sparse, zero_center, implicit):
        np.random.seed(42)
        n_obs_x, n_obs_y, n_vars, n_comps = 20, 18, 10, 5
        X = np.random.randn(n_obs_x, n_vars)
        Y = np.random.randn(n_obs_y, n_vars)
        if sparse:
            X = csr_matrix(X)
            Y = csr_matrix(Y)
        U, s, Vt = truncated_svd_cross_covariance(
            X, Y, n_comps=n_comps, zero_center=zero_center, implicit=implicit, random_state=42
        )
        # Check shapes
        assert U.shape == (n_obs_x, n_comps)
        assert s.shape == (n_comps,)
        assert Vt.shape == (n_comps, n_obs_y)
        # Check singular values are sorted
        assert np.all(np.diff(s) <= 0)

    def test_consistency_dense_sparse(self):
        np.random.seed(0)
        n_obs_x, n_obs_y, n_vars, n_comps = 15, 12, 8, 4
        X = np.random.randn(n_obs_x, n_vars)
        Y = np.random.randn(n_obs_y, n_vars)
        # Dense, implicit centering
        U1, s1, Vt1 = truncated_svd_cross_covariance(
            X, Y, n_comps=n_comps, zero_center=True, implicit=True, random_state=0
        )
        # Sparse, implicit centering
        Xs = csr_matrix(X)
        Ys = csr_matrix(Y)
        U2, s2, Vt2 = truncated_svd_cross_covariance(
            Xs, Ys, n_comps=n_comps, zero_center=True, implicit=True, random_state=0
        )
        # Compare singular values (allowing some tolerance)
        np.testing.assert_allclose(np.sort(s1), np.sort(s2), rtol=1e-2, atol=1e-2)

    def test_error_on_shape_mismatch(self):
        X = np.random.randn(10, 5)
        Y = np.random.randn(8, 4)
        with pytest.raises(ValueError, match="same number of variables"):
            truncated_svd_cross_covariance(X, Y)

    def test_error_on_type_mismatch(self):
        X = np.random.randn(10, 5)
        Y = csr_matrix(np.random.randn(8, 5))
        with pytest.raises(TypeError, match="same type"):
            truncated_svd_cross_covariance(X, Y)
