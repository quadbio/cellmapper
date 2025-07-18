"""Test edge cases for MappingOperator class."""

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from cellmapper.model.kernel import Kernel
from cellmapper.model.mapping_operator import MappingOperator


class TestMappingOperatorInitialization:
    """Test MappingOperator initialization with various inputs."""

    @pytest.mark.parametrize("is_sparse", [False, True])
    def test_init_with_raw_matrices(self, small_symmetric_matrix, sparse_symmetric_matrix, is_sparse):
        """Test initialization with raw numpy and sparse matrices."""
        matrix = sparse_symmetric_matrix if is_sparse else small_symmetric_matrix
        op = MappingOperator(matrix, is_self_mapping=True)

        assert op.is_self_mapping is True
        assert op.is_sparse is is_sparse
        assert op.is_symmetric is True
        assert op.mapping_matrix.shape == matrix.shape
        if is_sparse:
            assert isinstance(op.mapping_matrix, csr_matrix)
        else:
            assert op.mapping_matrix.dtype == np.float32

    @pytest.mark.parametrize("sparse_format", [csc_matrix, coo_matrix])
    def test_sparse_format_conversion(self, sparse_symmetric_matrix, sparse_format):
        """Test that different sparse formats are converted to CSR."""
        matrix = sparse_format(sparse_symmetric_matrix)
        op = MappingOperator(matrix, is_self_mapping=True)
        assert isinstance(op.mapping_matrix, csr_matrix)

    @pytest.mark.parametrize(
        "eigen_solver,expected_n_eigs",
        [
            ("partial", 1),  # Capped for small matrix
            ("complete", 3),  # Full matrix size
        ],
    )
    def test_eigen_solver_options(self, small_symmetric_matrix, eigen_solver, expected_n_eigs):
        """Test different eigendecomposition methods."""
        op = MappingOperator(small_symmetric_matrix, is_self_mapping=True, eigen_solver=eigen_solver)
        assert op.eigen_solver == eigen_solver
        assert op.n_eigenvectors == expected_n_eigs

    def test_inference_of_mapping_type(self, small_symmetric_matrix, rectangular_matrix):
        """Test automatic inference of is_self_mapping from matrix shape."""
        # Square matrix should infer self-mapping
        op_square = MappingOperator(small_symmetric_matrix)
        assert op_square.is_self_mapping is True

        # Rectangular matrix should infer cross-mapping
        op_rect = MappingOperator(rectangular_matrix)
        assert op_rect.is_self_mapping is False

    def test_kernel_object_without_matrix(self):
        """Test initialization with Kernel object that lacks computed matrix."""
        x = np.random.rand(10, 5)
        kernel = Kernel(x, x)  # No compute_kernel_matrix() called

        with pytest.raises(ValueError, match="Kernel object does not have a computed kernel matrix"):
            MappingOperator(kernel)


class TestMappingOperatorErrors:
    """Test error conditions in MappingOperator."""

    @pytest.mark.parametrize(
        "invalid_input,expected_error,match_text",
        [
            ("invalid_solver", ValueError, "Unknown eigen_solver"),
            ("invalid_diffusion", ValueError, "Unknown diffusion method"),
            ("invalid_power", ValueError, "Power t must be >= 1"),
            ("cross_mapping_power", ValueError, "Matrix powers t > 1 only supported for self-mapping"),
            ("self_mapping_rect", ValueError, "Self-mapping requires square matrix"),
        ],
    )
    def test_various_errors(
        self,
        small_symmetric_matrix,
        rectangular_matrix,
        test_data_for_mapping,
        invalid_input,
        expected_error,
        match_text,
    ):
        """Test various error conditions."""
        if invalid_input == "invalid_solver":
            with pytest.raises(expected_error, match=match_text):
                MappingOperator(small_symmetric_matrix, is_self_mapping=True, eigen_solver="invalid")

        elif invalid_input == "invalid_diffusion":
            op = MappingOperator(small_symmetric_matrix, is_self_mapping=True)
            with pytest.raises(expected_error, match=match_text):
                op.apply(test_data_for_mapping, t=2, diffusion_method="invalid")

        elif invalid_input == "invalid_power":
            op = MappingOperator(small_symmetric_matrix, is_self_mapping=True)
            with pytest.raises(expected_error, match=match_text):
                op.apply(test_data_for_mapping, t=0)

        elif invalid_input == "cross_mapping_power":
            op = MappingOperator(rectangular_matrix, is_self_mapping=False)
            with pytest.raises(expected_error, match=match_text):
                op.apply(test_data_for_mapping[:3], t=2)

        elif invalid_input == "self_mapping_rect":
            with pytest.raises(expected_error, match=match_text):
                MappingOperator(rectangular_matrix, is_self_mapping=True)

    def test_asymmetric_matrix_spectral_methods(self, small_asymmetric_matrix):
        """Test that asymmetric matrices cannot use spectral methods."""
        op = MappingOperator(small_asymmetric_matrix, is_self_mapping=True)

        with pytest.raises(ValueError, match="Cannot generate symmetric mapping matrix"):
            _ = op._symmetric_mapping_matrix


class TestMappingOperatorDenseOperations:
    """Test operations with dense matrices (usually sparse in practice)."""

    @pytest.mark.parametrize("diffusion_method", ["iterative", "spectral"])
    @pytest.mark.parametrize("t", [None, 2, 5])
    def test_dense_mapping_methods(self, small_symmetric_matrix, test_data_for_mapping, diffusion_method, t):
        """Test different mapping methods with dense matrices."""
        op = MappingOperator(small_symmetric_matrix, is_self_mapping=True)

        if t is None:
            result = op.apply(test_data_for_mapping, t=t)
        else:
            result = op.apply(test_data_for_mapping, t=t, diffusion_method=diffusion_method)

        assert isinstance(result, np.ndarray)
        assert result.shape == test_data_for_mapping.shape

    def test_dense_vs_sparse_consistency(self, small_symmetric_matrix, test_data_for_mapping):
        """Test that dense and sparse versions give similar results."""
        # Dense version
        op_dense = MappingOperator(small_symmetric_matrix, is_self_mapping=True)
        result_dense = op_dense.apply(test_data_for_mapping, t=2, diffusion_method="iterative")

        # Sparse version
        sparse_matrix = csr_matrix(small_symmetric_matrix)
        op_sparse = MappingOperator(sparse_matrix, is_self_mapping=True)
        result_sparse = op_sparse.apply(test_data_for_mapping, t=2, diffusion_method="iterative")

        # Convert sparse result to dense if needed
        if hasattr(result_sparse, "toarray"):
            result_sparse = result_sparse.toarray()

        assert np.allclose(result_dense, result_sparse, rtol=1e-5)


class TestMappingOperatorCaching:
    """Test eigendecomposition caching functionality."""

    def test_eigendecomposition_caching_and_clearing(self, small_symmetric_matrix):
        """Test that eigendecomposition is cached and can be cleared."""
        op = MappingOperator(small_symmetric_matrix, is_self_mapping=True)

        # First access should compute eigendecomposition
        eigenvals1, eigenvecs1 = op._eigendecomposition

        # Second access should use cached version
        eigenvals2, eigenvecs2 = op._eigendecomposition
        assert eigenvals1 is eigenvals2  # Same object reference
        assert eigenvecs1 is eigenvecs2

        # Clear cache
        op.clear_cache()
        assert "_eigendecomposition" not in op.__dict__

        # Should recompute after clearing
        eigenvals3, eigenvecs3 = op._eigendecomposition
        assert eigenvals3 is not eigenvals1  # Different object after recomputation


class TestMappingOperatorStringRepresentations:
    """Test __repr__ and __str__ methods."""

    @pytest.mark.parametrize(
        "matrix_type,expected_terms",
        [
            ("dense_symmetric", ["MappingOperator", "3×3", "self-mapping", "dense", "symmetric"]),
            ("sparse_symmetric", ["sparse", "sparsity:"]),
            ("rectangular", ["cross-mapping", "3×4", "N/A"]),
        ],
    )
    def test_repr_different_matrices(
        self, matrix_type, expected_terms, small_symmetric_matrix, sparse_symmetric_matrix, rectangular_matrix
    ):
        """Test __repr__ with different matrix types."""
        if matrix_type == "dense_symmetric":
            op = MappingOperator(small_symmetric_matrix, is_self_mapping=True)
        elif matrix_type == "sparse_symmetric":
            op = MappingOperator(sparse_symmetric_matrix, is_self_mapping=True)
        else:  # rectangular
            op = MappingOperator(rectangular_matrix, is_self_mapping=False)

        repr_str = repr(op)
        for term in expected_terms:
            assert term in repr_str

    def test_str_basic_features(self, sparse_symmetric_matrix):
        """Test __str__ includes basic features."""
        # Test sparse fill percentage
        op_sparse = MappingOperator(sparse_symmetric_matrix, is_self_mapping=True)
        str_repr_sparse = str(op_sparse)
        assert "filled" in str_repr_sparse

    def test_eigendecomposition_status_in_repr(self, small_symmetric_matrix):
        """Test __repr__ shows eigendecomposition status."""
        op = MappingOperator(small_symmetric_matrix, is_self_mapping=True)

        # Before computing eigendecomposition
        assert "eigendecomp: not computed" in repr(op)

        # After computing eigendecomposition
        _ = op._eigendecomposition
        assert "eigendecomp: cached" in repr(op)


class TestMappingOperatorEdgeCases:
    """Test edge cases and numerical stability."""

    @pytest.mark.parametrize(
        "matrix_case",
        [
            "very_small",  # 1x1 matrix
            "identity",  # Identity matrix
            "numerical_precision",  # Very small values
        ],
    )
    def test_edge_case_matrices(self, matrix_case):
        """Test various edge case matrices."""
        if matrix_case == "very_small":
            matrix = np.array([[1.0]])
            op = MappingOperator(matrix, is_self_mapping=True)
            assert op.mapping_matrix.shape == (1, 1)
            assert op.n_eigenvectors == 1

        elif matrix_case == "identity":
            matrix = np.eye(3)
            op = MappingOperator(matrix, is_self_mapping=True)
            assert op.is_symmetric is True
            # Test mapping preserves structure
            test_data = np.array([[1, 2], [3, 4], [5, 6]])
            result = op.apply(test_data, t=None)
            assert result.shape == test_data.shape

        elif matrix_case == "numerical_precision":
            matrix = np.array([[1e-10, 1e-8, 1e-9], [1e-8, 1e-10, 1e-7], [1e-9, 1e-7, 1e-10]])
            op = MappingOperator(matrix, is_self_mapping=True)
            assert op.mapping_matrix.shape == matrix.shape

    def test_matrix_property_normalization(self, small_symmetric_matrix, sparse_symmetric_matrix):
        """Test that matrix property returns normalized matrices."""
        # Dense matrix
        op_dense = MappingOperator(small_symmetric_matrix, is_self_mapping=True)
        matrix_dense = op_dense.matrix
        row_sums_dense = matrix_dense.sum(axis=1)
        assert np.allclose(row_sums_dense, 1.0, rtol=1e-6)

        # Sparse matrix
        op_sparse = MappingOperator(sparse_symmetric_matrix, is_self_mapping=True)
        matrix_sparse = op_sparse.matrix
        assert isinstance(matrix_sparse, csr_matrix)
        row_sums_sparse = np.array(matrix_sparse.sum(axis=1)).flatten()
        assert np.allclose(row_sums_sparse, 1.0, rtol=1e-6)
