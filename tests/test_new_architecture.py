"""Test suite for the new BaseMapper, ObsMapper, and VarMapper architecture."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from cellmapper.model.obs_mapper import ObsMapper
from cellmapper.model.var_mapper import VarMapper


class TestObsMapper:
    """Tests for ObsMapper functionality."""

    @pytest.fixture
    def obs_mapper_self(self, adata_pbmc3k):
        """ObsMapper for self-mapping."""
        return ObsMapper(adata_pbmc3k)

    @pytest.fixture
    def obs_mapper_query_ref(self, query_reference_adata):
        """ObsMapper for query-to-reference mapping."""
        query, reference = query_reference_adata
        return ObsMapper(reference, query=query)

    def test_obs_mapper_initialization_self(self, obs_mapper_self, adata_pbmc3k):
        """Test ObsMapper initialization for self-mapping."""
        assert obs_mapper_self.reference is adata_pbmc3k
        assert obs_mapper_self.query is adata_pbmc3k
        assert obs_mapper_self._is_self_mapping is True

    def test_obs_mapper_initialization_query_ref(self, obs_mapper_query_ref, query_reference_adata):
        """Test ObsMapper initialization for query-to-reference mapping."""
        query, reference = query_reference_adata
        assert obs_mapper_query_ref.reference is reference
        assert obs_mapper_query_ref.query is query
        assert obs_mapper_query_ref._is_self_mapping is False

    def test_compute_neighbors(self, obs_mapper_self):
        """Test compute_neighbors method."""
        obs_mapper_self.compute_neighbors(n_neighbors=15, use_rep="X_pca", method="sklearn")

        # Check that neighbors are computed
        assert obs_mapper_self.neighbors is not None
        assert hasattr(obs_mapper_self.neighbors, "query_indices")
        assert hasattr(obs_mapper_self.neighbors, "query_distances")

    def test_compute_mapping_matrix(self, obs_mapper_self):
        """Test compute_mapping_matrix method."""
        obs_mapper_self.compute_neighbors(n_neighbors=15, use_rep="X_pca", method="sklearn")
        obs_mapper_self.compute_mapping_matrix(method="gaussian")

        # Check mapping matrix properties
        assert obs_mapper_self.mapping_matrix is not None
        assert issparse(obs_mapper_self.mapping_matrix)
        assert obs_mapper_self.mapping_matrix.shape == (obs_mapper_self.query.n_obs, obs_mapper_self.reference.n_obs)

    def test_map_obs_categorical(self, obs_mapper_self):
        """Test mapping categorical observations."""
        obs_mapper_self.compute_neighbors(n_neighbors=15, use_rep="X_pca", method="sklearn")
        obs_mapper_self.compute_mapping_matrix(method="gaussian")

        # Map categorical observation
        obs_mapper_self.map_obs(key="leiden", prediction_key="leiden_pred")

        # Check results
        assert "leiden_pred" in obs_mapper_self.query.obs.columns
        assert obs_mapper_self.query.obs["leiden_pred"].dtype.name == "category"

    def test_map_obs_numerical(self, obs_mapper_self):
        """Test mapping numerical observations."""
        obs_mapper_self.compute_neighbors(n_neighbors=15, use_rep="X_pca", method="sklearn")
        obs_mapper_self.compute_mapping_matrix(method="gaussian")

        # Map numerical observation
        obs_mapper_self.map_obs(key="dpt_pseudotime", prediction_key="dpt_pseudotime_pred")

        # Check results
        assert "dpt_pseudotime_pred" in obs_mapper_self.query.obs.columns
        assert np.issubdtype(obs_mapper_self.query.obs["dpt_pseudotime_pred"].dtype, np.number)

    def test_map_obsm_embeddings(self, obs_mapper_self):
        """Test mapping obsm embeddings."""
        obs_mapper_self.compute_neighbors(n_neighbors=15, use_rep="X_pca", method="sklearn")
        obs_mapper_self.compute_mapping_matrix(method="gaussian")

        # Map embeddings
        obs_mapper_self.map_obsm(key="X_pca", prediction_key="X_pca_pred")

        # Check results
        assert "X_pca_pred" in obs_mapper_self.query.obsm.keys()
        assert obs_mapper_self.query.obsm["X_pca_pred"].shape[0] == obs_mapper_self.query.n_obs
        assert obs_mapper_self.query.obsm["X_pca_pred"].shape[1] == obs_mapper_self.reference.obsm["X_pca"].shape[1]

    def test_query_reference_mapping(self, obs_mapper_query_ref):
        """Test query-to-reference mapping functionality."""
        obs_mapper_query_ref.compute_neighbors(n_neighbors=30, use_rep="X_pca", method="sklearn")
        obs_mapper_query_ref.compute_mapping_matrix(method="gaussian")

        # Map categorical observation
        obs_mapper_query_ref.map_obs(key="leiden", prediction_key="leiden_pred")

        # Check results
        assert "leiden_pred" in obs_mapper_query_ref.query.obs.columns
        assert obs_mapper_query_ref.query.obs["leiden_pred"].dtype.name == "category"

        # Check that query and reference are different
        assert obs_mapper_query_ref.query.n_obs != obs_mapper_query_ref.reference.n_obs


class TestVarMapper:
    """Tests for VarMapper functionality."""

    @pytest.fixture
    def var_mapper_self(self, adata_pbmc3k):
        """VarMapper for self-mapping."""
        return VarMapper(adata_pbmc3k)

    @pytest.fixture
    def var_mapper_query_ref(self, query_reference_adata):
        """VarMapper for query-to-reference mapping with gene overlap."""
        query, reference = query_reference_adata
        return VarMapper(reference, query=query)

    def test_var_mapper_initialization_self(self, var_mapper_self, adata_pbmc3k):
        """Test VarMapper initialization for self-mapping."""
        assert var_mapper_self.reference is adata_pbmc3k
        assert var_mapper_self.query is adata_pbmc3k
        assert var_mapper_self._is_self_mapping is True

    def test_var_mapper_initialization_query_ref(self, var_mapper_query_ref, query_reference_adata):
        """Test VarMapper initialization for query-to-reference mapping."""
        query, reference = query_reference_adata
        assert var_mapper_query_ref.reference is reference
        assert var_mapper_query_ref.query is query
        assert var_mapper_query_ref._is_self_mapping is False

    def test_compute_neighbors_genes(self, var_mapper_self):
        """Test compute_neighbors method for gene similarity."""
        var_mapper_self.compute_neighbors(n_neighbors=15, use_rep="X", method="sklearn")

        # Check that neighbors are computed
        assert var_mapper_self.neighbors is not None
        assert hasattr(var_mapper_self.neighbors, "query_indices")
        assert hasattr(var_mapper_self.neighbors, "query_distances")

    def test_compute_mapping_matrix_vars(self, var_mapper_self):
        """Test compute_mapping_matrix method for variables."""
        var_mapper_self.compute_neighbors(n_neighbors=15, use_rep="X", method="sklearn")
        var_mapper_self.compute_mapping_matrix(method="gaussian")

        # Check mapping matrix properties
        assert var_mapper_self.mapping_matrix is not None
        assert issparse(var_mapper_self.mapping_matrix)
        assert var_mapper_self.mapping_matrix.shape == (var_mapper_self.query.n_vars, var_mapper_self.reference.n_vars)

    def test_map_var_categorical(self, var_mapper_self):
        """Test mapping categorical variable annotations."""
        # Add some categorical variable annotation for testing
        var_mapper_self.reference.var["test_category"] = pd.Categorical(
            np.random.choice(["A", "B", "C"], size=var_mapper_self.reference.n_vars)
        )

        var_mapper_self.compute_neighbors(n_neighbors=15, use_rep="X", method="sklearn")
        var_mapper_self.compute_mapping_matrix(method="gaussian")

        # Map categorical variable annotation
        var_mapper_self.map_var(key="test_category", prediction_key="test_category_pred")

        # Check results
        assert "test_category_pred" in var_mapper_self.query.var.columns
        assert var_mapper_self.query.var["test_category_pred"].dtype.name == "category"

    def test_map_var_numerical(self, var_mapper_self):
        """Test mapping numerical variable annotations."""
        # Add some numerical variable annotation for testing
        var_mapper_self.reference.var["test_score"] = np.random.random(var_mapper_self.reference.n_vars)

        var_mapper_self.compute_neighbors(n_neighbors=15, use_rep="X", method="sklearn")
        var_mapper_self.compute_mapping_matrix(method="gaussian")

        # Map numerical variable annotation
        var_mapper_self.map_var(key="test_score", prediction_key="test_score_pred")

        # Check results
        assert "test_score_pred" in var_mapper_self.query.var.columns
        assert np.issubdtype(var_mapper_self.query.var["test_score_pred"].dtype, np.number)

    def test_map_varm_embeddings(self, var_mapper_self):
        """Test mapping varm embeddings."""
        # Add some variable embeddings for testing
        var_mapper_self.reference.varm["test_embedding"] = np.random.random((var_mapper_self.reference.n_vars, 10))

        var_mapper_self.compute_neighbors(n_neighbors=15, use_rep="X", method="sklearn")
        var_mapper_self.compute_mapping_matrix(method="gaussian")

        # Map variable embeddings
        var_mapper_self.map_varm(key="test_embedding", prediction_key="test_embedding_pred")

        # Check results
        assert "test_embedding_pred" in var_mapper_self.query.varm.keys()
        assert var_mapper_self.query.varm["test_embedding_pred"].shape[0] == var_mapper_self.query.n_vars
        assert var_mapper_self.query.varm["test_embedding_pred"].shape[1] == 10

    def test_query_reference_var_mapping(self, var_mapper_query_ref):
        """Test query-to-reference variable mapping with gene overlap."""
        # Add categorical variable annotation to reference
        var_mapper_query_ref.reference.var["gene_type"] = pd.Categorical(
            np.random.choice(["protein_coding", "lncRNA", "miRNA"], size=var_mapper_query_ref.reference.n_vars)
        )

        var_mapper_query_ref.compute_neighbors(n_neighbors=30, use_rep="X", method="sklearn")
        var_mapper_query_ref.compute_mapping_matrix(method="gaussian")

        # Map variable annotation
        var_mapper_query_ref.map_var(key="gene_type", prediction_key="gene_type_pred")

        # Check results
        assert "gene_type_pred" in var_mapper_query_ref.query.var.columns
        assert var_mapper_query_ref.query.var["gene_type_pred"].dtype.name == "category"


class TestBaseMapperIntegration:
    """Integration tests for BaseMapper functionality across ObsMapper and VarMapper."""

    def test_mapping_matrix_normalization(self, adata_pbmc3k):
        """Test that mapping matrices are properly normalized."""
        obs_mapper = ObsMapper(adata_pbmc3k)
        obs_mapper.compute_neighbors(n_neighbors=15, use_rep="X_pca", method="sklearn")
        obs_mapper.compute_mapping_matrix(method="gaussian")

        # Check row normalization (each row should sum to approximately 1)
        row_sums = np.array(obs_mapper.mapping_matrix.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_neighbors_consistency(self, adata_pbmc3k):
        """Test that neighbors computation is consistent between mappers."""
        obs_mapper = ObsMapper(adata_pbmc3k)
        var_mapper = VarMapper(adata_pbmc3k)

        # Compute neighbors for both
        obs_mapper.compute_neighbors(n_neighbors=15, use_rep="X_pca", method="sklearn")
        var_mapper.compute_neighbors(n_neighbors=15, use_rep="X", method="sklearn")

        # Both should have neighbors computed
        assert obs_mapper.neighbors is not None
        assert var_mapper.neighbors is not None

        # Shapes should be appropriate for their dimensions
        assert obs_mapper.neighbors.query_indices.shape[0] == adata_pbmc3k.n_obs
        assert var_mapper.neighbors.query_indices.shape[0] == adata_pbmc3k.n_vars

    def test_error_handling_missing_key(self, adata_pbmc3k):
        """Test error handling for missing keys."""
        obs_mapper = ObsMapper(adata_pbmc3k)
        obs_mapper.compute_neighbors(n_neighbors=15, use_rep="X_pca", method="sklearn")
        obs_mapper.compute_mapping_matrix(method="gaussian")

        # Should raise error for missing key
        with pytest.raises(KeyError):
            obs_mapper.map_obs(key="nonexistent_key", prediction_key="pred")

    def test_error_handling_missing_neighbors(self, adata_pbmc3k):
        """Test error handling when neighbors are not computed."""
        obs_mapper = ObsMapper(adata_pbmc3k)

        # Should raise error when trying to compute mapping matrix without neighbors
        with pytest.raises(ValueError, match="Neighbors must be computed"):
            obs_mapper.compute_mapping_matrix(method="gaussian")

    def test_error_handling_missing_mapping_matrix(self, adata_pbmc3k):
        """Test error handling when mapping matrix is not computed."""
        obs_mapper = ObsMapper(adata_pbmc3k)
        obs_mapper.compute_neighbors(n_neighbors=15, use_rep="X_pca", method="sklearn")

        # Should raise error when trying to map without mapping matrix
        with pytest.raises(ValueError, match="Mapping matrix must be computed"):
            obs_mapper.map_obs(key="leiden", prediction_key="leiden_pred")


class TestCompatibilityWithOldAPI:
    """Tests to ensure compatibility with the old CellMapper API patterns."""

    def test_obs_mapper_old_style_usage(self, query_reference_adata):
        """Test that ObsMapper can be used in old CellMapper style."""
        query, reference = query_reference_adata

        # Create mapper similar to old CellMapper usage
        mapper = ObsMapper(reference, query=query)

        # Use the old-style workflow
        mapper.compute_neighbors(n_neighbors=30, use_rep="X_pca", method="sklearn")
        mapper.compute_mapping_matrix(method="gaussian")
        mapper.map_obs(key="leiden")  # Should use default prediction_key

        # Check results
        assert "leiden_pred" in mapper.query.obs.columns

    def test_var_mapper_old_style_usage(self, query_reference_adata):
        """Test that VarMapper can be used in old CellMapper style."""
        query, reference = query_reference_adata

        # Add variable annotation for testing
        reference.var["gene_biotype"] = pd.Categorical(
            np.random.choice(["protein_coding", "lncRNA"], size=reference.n_vars)
        )

        # Create mapper similar to old CellMapper usage
        mapper = VarMapper(reference, query=query)

        # Use the old-style workflow
        mapper.compute_neighbors(n_neighbors=30, use_rep="X", method="sklearn")
        mapper.compute_mapping_matrix(method="gaussian")
        mapper.map_var(key="gene_biotype")  # Should use default prediction_key

        # Check results
        assert "gene_biotype_pred" in mapper.query.var.columns
