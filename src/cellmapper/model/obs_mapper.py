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

    def _compute_fallback_representation_self(self, n_comps: int | None, fallback_kwargs: dict) -> str:
        """Compute fallback representation for self-mapping."""
        key_added = fallback_kwargs.pop("key_added", "X_pca")
        sc.tl.pca(self.query, n_comps=n_comps, key_added=key_added, **fallback_kwargs)

        return key_added

    def _compute_fallback_representation_joint(self, method: str, n_comps: int | None, fallback_kwargs: dict) -> str:
        """Compute joint fallback representation for cross-mapping."""
        if method == "fast_cca":
            key_added = fallback_kwargs.pop("key_added", "X_cca")
            self.compute_fast_cca(n_comps=n_comps, key_added=key_added, **fallback_kwargs)
        elif method == "joint_pca":
            key_added = fallback_kwargs.pop("key_added", "X_pca")
            self.compute_joint_pca(n_comps=n_comps, key_added=key_added, **fallback_kwargs)
        else:
            raise ValueError(f"Unknown fallback_representation: {method}")

        return key_added

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

        # Store postfixes for evaluation methods
        self.prediction_postfix = prediction_postfix
        self.confidence_postfix = confidence_postfix

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

        logger.info("Mapping embeddings for key '%s'.", key)

        reference_embeddings = np.asarray(self.reference.obsm[key])
        query_embeddings = self._map_matrix(reference_embeddings)

        # Use prediction_key if provided, otherwise create output key with postfix
        if prediction_key is not None:
            output_key = prediction_key
        else:
            output_key = f"{key}_{prediction_postfix}"

        self.query.obsm[output_key] = query_embeddings
        logger.info("Embeddings mapped and stored in query.obsm['%s'].", output_key)

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

        # Handle sparse matrices properly by checking if it's sparse
        from scipy.sparse import issparse

        if issparse(reference_layer):
            reference_matrix = reference_layer.toarray()  # type: ignore
        else:
            reference_matrix = np.asarray(reference_layer)

        query_layer = self._map_matrix(reference_matrix)

        # Create imputed AnnData
        query_imputed = create_imputed_anndata(
            expression_data=query_layer, query_adata=self.query, reference_adata=self.reference
        )

        # Store as property for evaluation
        self.query_imputed = query_imputed

        message = f"Expression for layer '{key}' mapped."
        if not self._is_self_mapping:
            message += f" Feature space matches reference (n_vars={self.reference.n_vars})."
        logger.info(message)

        return query_imputed

    def map(
        self,
        obs_keys: str | list[str] | None = None,
        obsm_keys: str | list[str] | None = None,
        layer_key: str | None = None,
        prediction_postfix: str = "pred",
        knn_method: str = "sklearn",
        mapping_method: str = "gaussian",
        use_rep: str | None = "X_pca",
        n_neighbors: int = 30,
        **kwargs,
    ) -> None:
        """
        Map observations, embeddings, and expression layers in one call.

        This method combines neighbor computation, mapping matrix computation,
        and the actual mapping operations.

        Parameters
        ----------
        obs_keys
            Observation keys to map.
        obsm_keys
            Embedding keys to map.
        layer_key
            Layer key to map.
        prediction_postfix
            Postfix for predictions.
        knn_method
            Method for computing neighbors.
        mapping_method
            Method for computing mapping matrix.
        use_rep
            Representation to use for neighbors.
        n_neighbors
            Number of neighbors.
        **kwargs
            Additional arguments passed to compute_neighbors.
        """
        # Compute neighbors if not already computed
        if self.knn is None:
            self.compute_neighbors(n_neighbors=n_neighbors, use_rep=use_rep, method=knn_method, **kwargs)

        # Compute mapping matrix if not already computed
        if self.mapping_matrix is None:
            self.compute_mapping_matrix(method=mapping_method)

        # Map observations
        if obs_keys is not None:
            if isinstance(obs_keys, str):
                obs_keys = [obs_keys]
            for obs_key in obs_keys:
                self.map_obs(key=obs_key, prediction_postfix=prediction_postfix)

        # Map embeddings
        if obsm_keys is not None:
            if isinstance(obsm_keys, str):
                obsm_keys = [obsm_keys]
            for obsm_key in obsm_keys:
                self.map_obsm(key=obsm_key, prediction_postfix=prediction_postfix)

        # Map layers
        if layer_key is not None:
            self.map_layers(key=layer_key)

    @property
    def query_imputed(self) -> AnnData | None:
        """Get imputed query data."""
        return getattr(self, "_query_imputed", None)

    @query_imputed.setter
    def query_imputed(self, value: AnnData | np.ndarray | csr_matrix | pd.DataFrame | None) -> None:
        """Set imputed query data."""
        if value is None:
            self._query_imputed = None
        elif isinstance(value, AnnData):
            self._query_imputed = value
        else:
            # Convert to AnnData using utility function
            self._query_imputed = create_imputed_anndata(
                expression_data=value, query_adata=self.query, reference_adata=self.reference
            )

    def load_precomputed_distances(self, distances_key: str = "distances", include_self: bool | None = None) -> None:
        """
        Load precomputed distances from the AnnData object.

        This method is only available in self-mapping mode.

        Parameters
        ----------
        distances_key
            Key in adata.obsp where the precomputed distances are stored.
        include_self
            If True, include self as a neighbor (even if not present in the distance matrix).
            If False, exclude self connections (even if present in the distance matrix).
            If None (default), preserve the original behavior of the distance matrix.

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``knn``: Neighbors object constructed from the precomputed distances.
        """
        if not self._is_self_mapping:
            raise ValueError("load_precomputed_distances is only available in self-mapping mode.")

        # Access the precomputed distances
        distances_matrix = self.query.obsp[distances_key]

        # Import here to avoid circular imports
        from cellmapper.model.knn import Neighbors

        # Create a neighbors object using the factory method
        self.knn = Neighbors.from_distances(distances_matrix, include_self=include_self)

        logger.info(
            "Loaded precomputed distances from '%s' with %d cells and %d neighbors per cell.",
            distances_key,
            distances_matrix.shape[0],
            self.knn.xx.n_neighbors,
        )
