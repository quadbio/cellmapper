"""Observation-dimension mapper implementation."""

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

from cellmapper.logging import logger
from cellmapper.model.base_mapper import BaseMapper
from cellmapper.model.embedding import EmbeddingMixin
from cellmapper.model.evaluate import EvaluationMixin
from cellmapper.utils import create_imputed_anndata


class ObsMapper(EvaluationMixin, EmbeddingMixin, BaseMapper):
    """
    Mapper for observation-dimension data (cells).

    This class handles mapping along the observation dimension, transferring
    data from reference cells to query cells. This is the traditional mapping
    direction used in the original CellMapper.
    """

    def _get_dimension_name(self) -> str:
        """Get the name of the dimension being mapped."""
        return "cells"

    def _get_query_size(self) -> int:
        """Get the size of the query dataset in the observation dimension."""
        return self.query.n_obs

    def _get_reference_size(self) -> int:
        """Get the size of the reference dataset in the observation dimension."""
        return self.reference.n_obs

    def _get_expected_mapping_matrix_shape(self) -> tuple[int, int]:
        """Get the expected shape of the mapping matrix for observation mapping."""
        return (self.query.n_obs, self.reference.n_obs)

    def _get_target_index(self) -> pd.Index:
        """Get the index for the target observation dimension."""
        return self.query.obs_names

    def _apply_mapping_to_matrix(self, matrix: np.ndarray | csr_matrix) -> np.ndarray | csr_matrix:
        """Apply the mapping matrix to a data matrix for observation mapping."""
        return self.mapping_matrix @ matrix

    def _store_mapping_result(
        self,
        target_container: dict | AnnData,
        target_key: str,
        postfix: str,
        data: pd.Series | np.ndarray,
        data_type: str,
    ) -> None:
        """Store the mapping result in query.obs or query.obsm."""
        output_key = f"{target_key}_{postfix}"

        if isinstance(target_container, AnnData):
            if data_type in ["categorical", "numerical", "confidence"]:
                target_container.obs[output_key] = data
            elif data_type == "embedding":
                target_container.obsm[output_key] = np.asarray(data)
        else:
            # Dictionary-like container
            target_container[output_key] = data

    def _compute_fallback_representation_self(self, key_added: str, n_comps: int | None, fallback_kwargs: dict) -> None:
        """Compute fallback representation for self-mapping."""
        sc.tl.pca(self.query, n_comps=n_comps, key_added=key_added, **fallback_kwargs)

    def _compute_fallback_representation_joint(
        self, method: str, key_added: str, n_comps: int | None, fallback_kwargs: dict
    ) -> None:
        """Compute joint fallback representation for cross-mapping."""
        if method == "fast_cca":
            self.compute_fast_cca(n_comps=n_comps, key_added=key_added, **fallback_kwargs)
        elif method == "joint_pca":
            self.compute_joint_pca(n_comps=n_comps, key_added=key_added, **fallback_kwargs)
        else:
            raise ValueError(f"Unknown fallback_representation: {method}")

    def _extract_representations(self, use_rep: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract the representations for neighbor computation."""
        if use_rep == "X":
            xrep = np.asarray(self.reference.X)
            yrep = np.asarray(self.query.X)
        else:
            xrep = np.asarray(self.reference.obsm[use_rep])
            yrep = np.asarray(self.query.obsm[use_rep])
        return xrep, yrep

    def map_obs(
        self,
        key: str,
        prediction_key: str | None = None,
        prediction_postfix: str = "pred",
        confidence_postfix: str = "conf",
    ) -> None:
        """
        Map observation data from reference to query cells.

        Parameters
        ----------
        key
            Key in reference.obs to be transferred to query.obs.
        prediction_key
            Custom key for prediction results. If provided, overrides prediction_postfix.
        prediction_postfix
            Postfix for prediction results.
        confidence_postfix
            Postfix for confidence scores (categorical data only).
        """
        if key not in self.reference.obs.columns:
            raise KeyError(f"Key '{key}' not found in reference.obs")

        reference_data = self.reference.obs[key]

        # Use prediction_key if provided, otherwise use default postfix
        if prediction_key is not None:
            prediction_postfix = (
                prediction_key.replace(f"{key}_", "") if f"{key}_" in prediction_key else prediction_key
            )

        self._map_data(reference_data, prediction_postfix, confidence_postfix, self.query, key)

    def map_obsm(self, key: str, prediction_key: str | None = None, prediction_postfix: str = "pred") -> None:
        """
        Map embeddings from reference to query cells.

        Parameters
        ----------
        key
            Key in reference.obsm storing embeddings to transfer.
        prediction_key
            Custom key for output. If provided, overrides prediction_postfix.
        prediction_postfix
            Postfix for output key in query.obsm.
        """
        if key not in self.reference.obsm.keys():
            raise KeyError(f"Key '{key}' not found in reference.obsm")

        reference_data = self.reference.obsm[key]

        # Use prediction_key if provided, otherwise use default postfix
        if prediction_key is not None:
            prediction_postfix = (
                prediction_key.replace(f"{key}_", "") if f"{key}_" in prediction_key else prediction_key
            )

        self._map_data(reference_data, prediction_postfix, None, self.query, key)
        if key not in self.reference.obsm.keys():
            raise KeyError(f"Key '{key}' not found in reference.obsm")

        logger.info("Mapping embeddings for key '%s'.", key)

        reference_embeddings = np.asarray(self.reference.obsm[key])
        query_embeddings = self._map_matrix(reference_embeddings)

        output_key = f"{key}_{prediction_postfix}"
        self.query.obsm[output_key] = query_embeddings

        logger.info("Embeddings mapped and stored in query.obsm['%s'].", output_key)

    def map_layers(self, key: str) -> AnnData:
        """
        Map expression values from reference to query cells.

        Parameters
        ----------
        key
            Key in reference.layers to transfer. Use "X" for reference.X.

        Returns
        -------
        AnnData object with mapped expression data.
        """
        if key != "X" and key not in self.reference.layers.keys():
            raise KeyError(f"Key '{key}' not found in reference.layers")

        logger.info("Mapping expression data for key '%s'.", key)

        reference_layer = self.reference.X if key == "X" else self.reference.layers[key]
        query_layer = self._map_matrix(np.asarray(reference_layer))

        # Create imputed AnnData
        query_imputed = create_imputed_anndata(
            expression_data=query_layer, query_adata=self.query, reference_adata=self.reference
        )

        message = f"Expression for layer '{key}' mapped."
        if not self._is_self_mapping:
            message += f" Feature space matches reference (n_vars={self.reference.n_vars})."
        logger.info(message)

        return query_imputed
