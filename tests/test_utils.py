import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cellmapper.utils import extract_neighbors_from_distances, truncated_svd_cross_covariance


class TestExtractNeighborsFromDistances:
    """Tests for the extract_neighbors_from_distances function."""

    def test_basic_extraction(self):
        """Test basic extraction from a simple distance matrix."""
        # Create a simple distance matrix
        distances_data = np.array(
            [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 4.0, 5.0], [2.0, 4.0, 0.0, 6.0], [3.0, 5.0, 6.0, 0.0]]
        )
        distances = csr_matrix(distances_data)

        # Extract neighbors - now simply extracts as-is from the sparse matrix
        indices, nbr_distances = extract_neighbors_from_distances(distances)

        # Sparse matrix automatically removes zeros (self-loops), so we expect 3 neighbors per cell
        expected_shape = (4, 3)  # 3 non-self neighbors for each cell

        assert indices.shape == expected_shape
        assert nbr_distances.shape == expected_shape

        # Check that all original non-zero values are preserved
        for i in range(4):
            for j in range(4):
                if distances_data[i, j] != 0:
                    assert j in indices[i]
                    idx = np.where(indices[i] == j)[0][0]
                    assert nbr_distances[i, idx] == distances_data[i, j]

        # Check all distances are sorted
        for i in range(4):
            assert np.all(np.diff(nbr_distances[i, :]) >= 0), f"Distances for cell {i} should be sorted"

    def test_sparse_matrix(self):
        """Test extraction from a sparse distance matrix with varying neighbor counts."""
        # Create a sparse matrix with only some connections (no self-loops)
        row_idx = [0, 0, 1, 1, 2, 2, 3, 3]
        col_idx = [1, 2, 0, 2, 0, 3, 1, 2]
        values = [1.0, 2.0, 1.0, 4.0, 2.0, 6.0, 4.0, 6.0]
        distances = csr_matrix((values, (row_idx, col_idx)), shape=(4, 4))

        # Extract neighbors
        indices, nbr_distances = extract_neighbors_from_distances(distances)

        # Check shapes - each cell has 2 neighbors
        assert indices.shape == (4, 2)
        assert nbr_distances.shape == (4, 2)

        # Check values for specific cells
        np.testing.assert_array_equal(indices[0], [1, 2])  # Cell 0 connects to cells 1,2
        np.testing.assert_array_equal(indices[2], [0, 3])  # Cell 2 connects to cells 0,3
        np.testing.assert_array_almost_equal(nbr_distances[0], [1.0, 2.0])
        np.testing.assert_array_almost_equal(nbr_distances[2], [2.0, 6.0])

    def test_invalid_input(self):
        """Test with invalid inputs."""
        # Test with non-square matrix
        distances = csr_matrix((3, 4))
        with pytest.raises(ValueError, match="Square distance matrix"):
            extract_neighbors_from_distances(distances)

        # Test with non-sparse matrix
        with pytest.raises(TypeError, match="must be a sparse matrix"):
            extract_neighbors_from_distances(np.zeros((3, 3)))  # type: ignore

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

        # Extract neighbors
        indices, nbr_distances = extract_neighbors_from_distances(distances)

        # Check for a specific cell with infinite distance
        row0_neighbors = indices[0]
        assert 2 in row0_neighbors, "Neighbor with infinite distance should be included"

        # Find where the infinite value is in the results
        idx = np.where(row0_neighbors == 2)[0][0]

        # Verify the distance is infinite
        assert np.isinf(nbr_distances[0, idx]), "Distance should be infinite"


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
