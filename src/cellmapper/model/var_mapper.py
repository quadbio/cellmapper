"""Variable-dimension mapper implementation."""

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

from cellmapper.logging import logger
from cellmapper.model.base_mapper import BaseMapper
from cellmapper.model.embedding import EmbeddingMixin


class VarMapper(BaseMapper, EmbeddingMixin):
    """
    Mapper for variable-dimension data (genes/features).

    This class handles mapping along the variable dimension, transferring
    data from reference variables to query variables. This enables gene-level
    mapping and analysis across datasets with different feature spaces.
    """

    def _get_dimension_name(self) -> str:
        """Get the name of the dimension being mapped."""
        return "variables"

    def _get_query_size(self) -> int:
        """Get the size of the query dataset in the variable dimension."""
        return self.query.n_vars

    def _get_reference_size(self) -> int:
        """Get the size of the reference dataset in the variable dimension."""
        return self.reference.n_vars

    def _get_target_index(self) -> pd.Index:
        """Get the index for the target variable dimension."""
        return self.query.var_names

    def _apply_mapping_to_matrix(self, matrix: np.ndarray | csr_matrix) -> np.ndarray | csr_matrix:
        """Apply the mapping matrix to a data matrix for variable mapping."""
        # For variable mapping, we map FROM reference variables TO query variables
        # mapping_matrix is (query_vars, reference_vars)
        # Input matrix is (reference_vars, features)
        # We need: mapping_matrix @ matrix to get (query_vars, features)
        if self.mapping_matrix is None:
            raise ValueError("Mapping matrix is None")
        return self.mapping_matrix @ matrix

    def _store_mapping_result(
        self,
        target_container: dict | AnnData,
        target_key: str,
        postfix: str,
        data: pd.Series | np.ndarray,
        data_type: str,
    ) -> None:
        """Store the mapping result in query.var or query.varm."""
        output_key = f"{target_key}_{postfix}"

        if isinstance(target_container, AnnData):
            if data_type in ["categorical", "numerical", "confidence"]:
                target_container.var[output_key] = data
            elif data_type == "embedding":
                target_container.varm[output_key] = np.asarray(data)
        else:
            # Dictionary-like container
            target_container[output_key] = data

    def _compute_fallback_representation_self(self, key_added: str, n_comps: int | None, fallback_kwargs: dict) -> None:
        """Compute fallback representation for self-mapping."""
        # For variables, we compute PCA on the transposed expression matrix
        query_T = self.query.T.copy()
        sc.tl.pca(query_T, n_comps=n_comps, key_added=key_added, **fallback_kwargs)
        self.query.varm[key_added] = query_T.obsm[key_added]

    def _compute_fallback_representation_joint(
        self, method: str, key_added: str, n_comps: int | None, fallback_kwargs: dict
    ) -> None:
        """Compute joint fallback representation for cross-mapping."""
        if method == "joint_pca":
            # Create transposed copies for variable-level analysis
            query_T = self.query.T.copy()
            reference_T = self.reference.T.copy()

            # Use inherited EmbeddingMixin functionality
            # Temporarily assign transposed data
            original_query = self.query
            original_reference = self.reference

            self.query = query_T
            self.reference = reference_T

            # Compute joint PCA
            self.compute_joint_pca(n_comps=n_comps, key_added=key_added, **fallback_kwargs)

            # Store results in original objects' varm
            original_query.varm[key_added] = query_T.obsm[key_added]
            original_reference.varm[key_added] = reference_T.obsm[key_added]

            # Restore original references
            self.query = original_query
            self.reference = original_reference
        else:
            raise ValueError(
                f"Unknown fallback_representation: {method}. "
                "For variable mapping, only 'joint_pca' is currently supported."
            )

    def _extract_representations(self, use_rep: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract the representations for neighbor computation."""
        if use_rep == "X":
            # For variables, we use the transposed expression matrix
            # Handle both dense and sparse matrices properly
            ref_X = self.reference.X
            query_X = self.query.X

            # Convert sparse matrices to dense first, then transpose
            if hasattr(ref_X, "toarray"):  # sparse matrix
                xrep = ref_X.toarray().T
            else:
                xrep = np.asarray(ref_X).T

            if hasattr(query_X, "toarray"):  # sparse matrix
                yrep = query_X.toarray().T
            else:
                yrep = np.asarray(query_X).T
        else:
            # Use precomputed variable representations
            xrep = np.asarray(self.reference.varm[use_rep])
            yrep = np.asarray(self.query.varm[use_rep])
        return xrep, yrep

    def map_var(
        self,
        key: str,
        prediction_key: str | None = None,
        prediction_postfix: str = "pred",
        confidence_postfix: str = "conf",
    ) -> None:
        """
        Map variable data from reference to query variables.

        Parameters
        ----------
        key
            Key in reference.var to be transferred to query.var.
        prediction_key
            Custom key for prediction results. If provided, overrides prediction_postfix.
        prediction_postfix
            Postfix for prediction results.
        confidence_postfix
            Postfix for confidence scores (categorical data only).
        """
        if key not in self.reference.var.columns:
            raise KeyError(f"Key '{key}' not found in reference.var")

        reference_data = self.reference.var[key]

        # Use prediction_key if provided, otherwise use default postfix
        if prediction_key is not None:
            prediction_postfix = (
                prediction_key.replace(f"{key}_", "") if f"{key}_" in prediction_key else prediction_key
            )

        self._map_data(reference_data, prediction_postfix, confidence_postfix, self.query, key)

    def map_varm(self, key: str, prediction_key: str | None = None, prediction_postfix: str = "pred") -> None:
        """
        Map variable embeddings from reference to query variables.

        Parameters
        ----------
        key
            Key in reference.varm storing variable embeddings to transfer.
        prediction_key
            Custom key for output. If provided, overrides prediction_postfix.
        prediction_postfix
            Postfix for output key in query.varm.
        """
        if key not in self.reference.varm.keys():
            raise KeyError(f"Key '{key}' not found in reference.varm")

        logger.info("Mapping variable embeddings for key '%s'.", key)

        reference_embeddings = np.asarray(self.reference.varm[key])
        query_embeddings = self._map_matrix(reference_embeddings)

        # Use prediction_key if provided, otherwise create output key with postfix
        if prediction_key is not None:
            output_key = prediction_key
        else:
            output_key = f"{key}_{prediction_postfix}"

        self.query.varm[output_key] = query_embeddings
        logger.info("Variable embeddings mapped and stored in query.varm['%s'].", output_key)
