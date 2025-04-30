"""k-NN based mapping of labels, embeddings, and expression values."""

import gc
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from sklearn.preprocessing import OneHotEncoder

from cellmapper.evaluate import CellMapperEvaluationMixin
from cellmapper.logging import logger

from .knn import Neighbors


class CellMapper(CellMapperEvaluationMixin):
    """Mapping of labels, embeddings, and expression values between reference and query datasets."""

    def __init__(self, ref: AnnData, query: AnnData) -> None:
        """
        Initialize the CellMapper class.

        Parameters
        ----------
        ref
            Reference dataset.
        query
            Query dataset.
        """
        self.ref = ref
        self.query = query

        # Initialize attributes
        self.knn: Neighbors | None = None
        self._mapping_matrix: csr_matrix | None = None
        self.n_neighbors: int | None = None
        self.label_transfer_metrics: dict[str, Any] | None = None
        self.label_transfer_report: pd.DataFrame | None = None
        self.prediction_postfix: str | None = None
        self.confidence_postfix: str | None = None
        self.only_yx: bool | None = None
        self.query_imputed: AnnData | None = None
        self.expression_transfer_metrics: dict[str, Any] | None = None

    def __repr__(self):
        """Return a concise string representation of the CellMapper object."""
        ref_summary = f"AnnData(n_obs={self.ref.n_obs:,}, n_vars={self.ref.n_vars:,})"
        query_summary = f"AnnData(n_obs={self.query.n_obs:,}, n_vars={self.query.n_vars:,})"
        return (
            f"CellMapper(ref={ref_summary}, query={query_summary}, "
            f"n_neighbors={self.n_neighbors}, "
            f"prediction_postfix={self.prediction_postfix}, "
            f"confidence_postfix={self.confidence_postfix})"
        )

    @property
    def mapping_matrix(self):
        """
        Get the mapping matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The mapping matrix.
        """
        return self._mapping_matrix

    @mapping_matrix.setter
    def mapping_matrix(self, value):
        """
        Set and validate the mapping matrix.

        Parameters
        ----------
        value
            The new mapping matrix to set.
        """
        if value is not None:
            # Validate and normalize the matrix
            self._mapping_matrix = self._validate_and_normalize_mapping_matrix(value)
        else:
            self._mapping_matrix = None

    def _validate_and_normalize_mapping_matrix(
        self, mapping_matrix: csr_matrix | coo_matrix | csc_matrix
    ) -> csr_matrix:
        """
        Validate and normalize the mapping matrix.

        Parameters
        ----------
        mapping_matrix
            The mapping matrix to validate and normalize.
        n_ref_cells
            Number of reference cells (rows).
        n_query_cells
            Number of query cells (columns).

        Returns
        -------
        Validated and row-normalized mapping matrix.
        """
        # Validate the shape
        if mapping_matrix.shape != (self.query.n_obs, self.ref.n_obs):
            raise ValueError(
                f"Mapping matrix shape mismatch: expected ({self.query.n_obs}, {self.ref.n_obs}), "
                f"but got {mapping_matrix.shape}."
            )

        # Row normalize the matrix
        row_sums = mapping_matrix.sum(axis=1).A1  # Convert to 1D array
        if np.any(row_sums == 0):
            logger.warning("Some rows in the mapping matrix have a sum of zero. These rows will be left unchanged.")
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        if not np.allclose(row_sums, 1):
            logger.info("Row-normalizing the mapping matrix.")
        mapping_matrix = mapping_matrix.multiply(1 / row_sums[:, None])

        # Ensure the matrix is in CSR format
        mapping_matrix = mapping_matrix.tocsr().astype(np.float32)

        return mapping_matrix

    def compute_joint_pca(self, n_components: int = 50, key_added: str = "pca_joint", **kwargs) -> None:
        """
        Compute a joint PCA on the normalized .X matrices of ref and query, using only overlapping genes.

        Parameters
        ----------
        n_components
            Number of principal components to compute.
        key_added
            Key under which to store the joint PCA embeddings in `.obsm` of both ref and query AnnData objects.
        **kwargs
            Additional keyword arguments to pass to scanpy's `pp.pca` function.

        Notes
        -----
        This method performs an inner join on genes (variables) between the reference and query AnnData objects,
        concatenates the normalized expression matrices, and computes a joint PCA using Scanpy. The resulting
        PCA embeddings are stored in `.obsm[key_added]` for both objects. This is a fallback and not recommended
        for most use cases. Please provide a biologically meaningful representation if possible.
        """
        logger.warning(
            "No representation provided (use_rep=None). "
            "Falling back to joint PCA on normalized .X of both datasets using only overlapping genes. "
            "This is NOT recommended for most use cases! Please provide a biologically meaningful representation."
        )
        # Concatenate with inner join on genes
        joint = ad.concat([self.ref, self.query], join="inner", label="batch", keys=["ref", "query"])

        # Compute PCA using scanpy
        sc.pp.pca(joint, n_comps=n_components, **kwargs)

        # Assign PCA embeddings back to each object using the batch key
        self.ref.obsm[key_added] = joint.obsm["X_pca"][joint.obs["batch"] == "ref"]
        self.query.obsm[key_added] = joint.obsm["X_pca"][joint.obs["batch"] == "query"]
        logger.info(
            "Joint PCA computed and stored as '%s' in both ref.obsm and query.obsm. "
            "Proceeding to use this as the representation for neighbor search.",
            key_added,
        )

    def compute_neighbors(
        self,
        n_neighbors: int = 30,
        use_rep: str | None = None,
        method: Literal["sklearn", "pynndescent", "rapids", "faiss"] = "sklearn",
        metric: str = "euclidean",
        only_yx: bool = False,
        joint_pca_key: str = "pca_joint",
        n_pca_components: int = 50,
        pca_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Compute nearest neighbors between reference and query datasets.

        Parameters
        ----------
        n_neighbors
            Number of nearest neighbors.
        use_rep
            Data representation based on which to find nearest neighbors. If None, a joint PCA will be computed.
        method
            Method to use for computing neighbors. "sklearn" and "pynndescent" run on CPU, "rapids" and "faiss" run on GPU. Note that all but "pynndescent" perform exact neighbor search. With GPU acceleration, "faiss" is usually fastest and more memory efficient than "rapids".
        metric
            Distance metric to use for nearest neighbors.
        only_yx
            If True, only compute the xy neighbors. This is faster, but not suitable for Jaccard or HNOCA methods.
        joint_pca_key
            Key under which to store the joint PCA embeddings if use_rep is None.
        n_pca_components
            Number of principal components to compute for joint PCA if use_rep is None.
        pca_kwargs
            Additional keyword arguments to pass to scanpy's `pp.pca` function if use_rep is None.

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``knn``: Nearest neighbors object.
        - ``n_neighbors``: Number of nearest neighbors.
        - ``only_yx``: Whether only yx neighbors were computed.
        """
        self.n_neighbors = n_neighbors
        self.only_yx = only_yx
        if use_rep is None:
            if pca_kwargs is None:
                pca_kwargs = {}
            self.compute_joint_pca(n_components=n_pca_components, key_added=joint_pca_key, **pca_kwargs)
            use_rep = joint_pca_key
        if use_rep == "X":
            xrep = self.ref.X
            yrep = self.query.X
        else:
            xrep = self.ref.obsm[use_rep]
            yrep = self.query.obsm[use_rep]
        self.knn = Neighbors(np.ascontiguousarray(xrep), np.ascontiguousarray(yrep))
        self.knn.compute_neighbors(n_neighbors=n_neighbors, method=method, metric=metric, only_yx=only_yx)

    def compute_mappping_matrix(
        self,
        method: Literal["jaccard", "gaussian", "scarches", "inverse_distance", "random", "hnoca"] = "gaussian",
    ) -> None:
        """
        Compute the mapping matrix for label transfer.

        Parameters
        ----------
        method
            Method to use for computing the mapping matrix. Options include:

            - "jaccard": Jaccard similarity. Inspired by GLUE :cite:`cao2022multi`
            - "gaussian": Gaussian kernel with adaptive bandwidth. Loosely inspired by MAGIC :cite:`van2018recovering`
            - "scarches": scArches kernel. Inspired by scArches :cite:`lotfollahi2022mapping`
            - "inverse_distance": Inverse distance kernel.
            - "random": Random kernel, useful for testing.
            - "hnoca": HNOCA kernel. Inspired by HNOCA-tools :cite:`he2024integrated`

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``mapping_matrix``: Mapping matrix for label transfer.
        """
        if self.knn is None or self.n_neighbors is None:
            raise ValueError("Neighbors have not been computed. Call compute_neighbors() first.")

        logger.info("Computing mapping matrix using method '%s'.", method)
        if method in ["jaccard", "hnoca"]:
            if self.only_yx:
                raise ValueError(
                    "Jaccard and HNOCa methods require both x and y neighbors to be computed. Set only_yx=False."
                )
            xx, yy, xy, yx = self.knn.get_adjacency_matrices()
            jaccard = (yx @ xx.T) + (yy @ xy.T)

            if method == "jaccard":
                jaccard.data /= 4 * self.n_neighbors - jaccard.data
            elif method == "hnoca":
                jaccard.data /= 2 * self.n_neighbors - jaccard.data
                jaccard.data = jaccard.data**2
            self.mapping_matrix = jaccard
        elif method in ["gaussian", "scarches", "inverse_distance", "random"]:
            self.mapping_matrix = self.knn.yx.knn_graph_connectivities(kernel=method)
        else:
            raise NotImplementedError(f"Method '{method}' is not implemented.")

    def transfer_labels(
        self, obs_keys: str | list[str], prediction_postfix: str = "pred", confidence_postfix: str = "conf"
    ) -> None:
        """
        Transfer discrete labels from reference dataset to query dataset for one or more keys.

        Parameters
        ----------
        obs_keys
            One or more keys in ``ref.obs`` to be transferred into ``query.obs`` (must be discrete)
        prediction_postfix
            New ``query.obs`` key added for the transferred labels, by default ``{obs_key}_pred`` for each obs_key.
        confidence_postfix
            New ``query.obs`` key added for the transferred label confidence, by default ``{obs_key}_conf`` for each obs_key.

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``query.obs``: Contains the transferred labels and their confidence scores.
        """
        if self.mapping_matrix is None:
            raise ValueError("Mapping matrix has not been computed. Call compute_mapping_matrix() first.")
        if isinstance(obs_keys, str):
            obs_keys = [obs_keys]

        self.prediction_postfix = prediction_postfix
        self.confidence_postfix = confidence_postfix

        # Loop over each observation key and perform label transfer
        for key in obs_keys:
            logger.info("Predicting labels for key '%s'.", key)
            onehot = OneHotEncoder(dtype=np.float32)
            xtab = onehot.fit_transform(
                self.ref.obs[[key]],
            )  # shape = (n_ref_cells x n_categories), sparse csr matrix, float32
            ytab = self.mapping_matrix @ xtab  # shape = (n_query_cells x n_categories), sparse crs matrix, float32

            pred = pd.Series(
                data=np.array(onehot.categories_[0])[ytab.argmax(axis=1).A1],
                index=self.query.obs_names,
                dtype=self.ref.obs[key].dtype,
            )
            conf = pd.Series(
                ytab.max(axis=1).toarray().ravel(),
                index=self.query.obs_names,
            )

            self.query.obs[f"{key}_{self.prediction_postfix}"] = pred
            self.query.obs[f"{key}_{self.confidence_postfix}"] = conf

            # Add colors if available
            if f"{key}_colors" in self.ref.uns:
                color_lookup = dict(zip(self.ref.obs[key].cat.categories, self.ref.uns[f"{key}_colors"], strict=True))
                self.query.uns[f"{key}_{self.prediction_postfix}_colors"] = [
                    color_lookup.get(cat, "#000000") for cat in pred.cat.categories
                ]

            logger.info("Labels transferred and stored in query.obs['%s'].", f"{key}_{self.prediction_postfix}")

            # Free memory explicitly
            del onehot, xtab, ytab, pred, conf
            gc.collect()

    def transfer_embeddings(self, obsm_keys: str | list[str], prediction_postfix: str = "pred") -> None:
        """
        Transfer embeddings from reference dataset to query dataset for one or more keys.

        Parameters
        ----------
        obsm_keys
            One or more keys in ``ref.obsm`` storing the embeddings to be transferred.
        prediction_postfix
            Postfix to append to the output keys in ``query.obsm`` where the transferred embeddings will be stored.

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``query.obsm``: Contains the transferred embeddings.
        """
        if self.mapping_matrix is None:
            raise ValueError("Mapping matrix has not been computed. Call compute_mapping_matrix() first.")

        if isinstance(obsm_keys, str):
            obsm_keys = [obsm_keys]

        for key in obsm_keys:
            logger.info("Transferring embeddings for key '%s'.", key)

            # Perform matrix multiplication to transfer embeddings
            ref_embeddings = self.ref.obsm[key]  # shape = (n_ref_cells x n_embedding_dims)
            query_embeddings = self.mapping_matrix @ ref_embeddings  # shape = (n_query_cells x n_embedding_dims)

            # Store the transferred embeddings in query.obsm
            output_key = f"{key}_{prediction_postfix}"
            self.query.obsm[output_key] = query_embeddings

            logger.info("Embeddings transferred and stored in query.obsm['%s'].", output_key)

    def transfer_expression(self, layer_key: str) -> None:
        """
        Transfer expression values (e.g., .X or entries from .layers) from reference dataset to a new imputed query AnnData object.

        Parameters
        ----------
        layer_key
            Key in ``ref.layers`` to be transferred. Use "X" to transfer ``ref.X``.

        Returns
        -------
        None

        Notes
        -----
        Creates/updates ``self.query_imputed`` with the transferred data in .X.
        The new AnnData object will have the same cells as the query, but the features (genes) of the reference.
        """
        if self.mapping_matrix is None:
            raise ValueError("Mapping matrix has not been computed. Call compute_mapping_matrix() first.")

        logger.info("Transferring layer for key '%s'.", layer_key)
        # Get the reference layer (or .X if key is "X")
        ref_layer = self.ref.X if layer_key == "X" else self.ref.layers[layer_key]
        query_layer = self.mapping_matrix @ ref_layer  # shape = (n_query_cells x n_ref_features)

        # Always create or update a new AnnData object for the imputed data, placing result in .X
        obs = self.query.obs.copy(deep=False)
        var = self.ref.var.copy(deep=False)
        if self.query_imputed is None:
            self.query_imputed = ad.AnnData(
                X=query_layer,
                obs=obs,
                var=var,
                uns=self.query.uns,
                obsm=self.query.obsm,
                varm=self.ref.varm,
                obsp=None,
                varp=None,
            )
        else:
            self.query_imputed.X = query_layer
        logger.info(
            "Imputed expression for layer '%s' stored in self.query_imputed.X.\n"
            "Note: The feature space now matches the reference (n_vars=%s), not the query (n_vars=%s).",
            layer_key,
            self.ref.n_vars,
            self.query.n_vars,
        )

    def fit(
        self,
        obs_keys: str | list[str] | None = None,
        obsm_keys: str | list[str] | None = None,
        layer_key: str | None = None,
        n_neighbors: int = 30,
        use_rep: str | None = None,
        knn_method: Literal["sklearn", "pynndescent", "rapids"] = "sklearn",
        metric: str = "euclidean",
        only_yx: bool = False,
        mapping_method: Literal["jaccard", "gaussian", "scarches", "inverse_distance", "random", "hnoca"] = "gaussian",
        prediction_postfix: str = "pred",
    ) -> "CellMapper":
        """
        Fit the CellMapper model.

        Parameters
        ----------
        obs_keys
            One or more keys in ``ref.obs`` to be transferred into ``query.obs`` (must be discrete)
        obsm_keys
            One or more keys in ``ref.obsm`` storing the embeddings to be transferred.
        layer_key
            Key in ``ref.layers`` to be transferred. Use "X" to transfer ``ref.X``.
        n_neighbors
            Number of nearest neighbors.
        use_rep
            Data representation based on which to find nearest neighbors.
        knn_method
            Method to use for computing neighbors.
        metric
            Distance metric to use for nearest neighbors.
        only_yx
            If True, only compute the xy neighbors. This is faster, but not suitable for Jaccard or HNOCA methods.
        mapping_method
            Method to use for computing the mapping matrix.
        prediction_postfix
            Postfix added to create new keys in ``query.obs`` for the transferred labels or in ``query.obsm`` for the transferred embeddings.
        """
        self.compute_neighbors(
            n_neighbors=n_neighbors, use_rep=use_rep, method=knn_method, metric=metric, only_yx=only_yx
        )
        self.compute_mappping_matrix(method=mapping_method)
        if obs_keys is not None:
            self.transfer_labels(obs_keys=obs_keys, prediction_postfix=prediction_postfix)
        if obsm_keys is not None:
            self.transfer_embeddings(obsm_keys=obsm_keys, prediction_postfix=prediction_postfix)
        if layer_key is not None:
            self.transfer_expression(layer_key=layer_key)
        if obs_keys is None and obsm_keys is None and layer_key is None:
            logger.warning(
                "Neither ``obs_keys``, ``obsm_keys`` or ``layer_key`` provided. No labels, embeddings or layers were transferred. "
                "Please provide at least one of ``obs_keys``, ``obsm_keys`` or ``layer_key``."
            )

        return self
