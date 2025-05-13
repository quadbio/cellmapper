"""k-NN based mapping of labels, embeddings, and expression values."""

import gc
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse
from sklearn.preprocessing import OneHotEncoder

from cellmapper.evaluate import CellMapperEvaluationMixin
from cellmapper.logging import logger
from cellmapper.utils import create_imputed_anndata, truncated_svd_cross_covariance

from .knn import Neighbors


class CellMapper(CellMapperEvaluationMixin):
    """Mapping of labels, embeddings, and expression values between reference and query datasets."""

    def __init__(self, query: AnnData, reference: AnnData | None = None) -> None:
        """
        Initialize the CellMapper class.

        Parameters
        ----------
        query
            Query dataset.
        reference
            Optional reference dataset.
        """
        self.query = query

        # Handle self-mapping case - use the query as both source and target
        self.reference = reference if reference is not None else query
        self._is_self_mapping = reference is None

        # Update log message to reflect self-mapping if applicable
        if self._is_self_mapping:
            logger.info("Initialized CellMapper for self-mapping with %d cells.", query.n_obs)
        else:
            logger.info(
                "Initialized CellMapper with %d query cells and %d reference cells.",
                query.n_obs,
                self.reference.n_obs,
            )

        # Initialize result containers
        self.knn: Neighbors | None = None
        self._mapping_matrix: csr_matrix | None = None
        self.label_transfer_metrics: dict[str, Any] | None = None
        self.label_transfer_report: pd.DataFrame | None = None
        self.prediction_postfix: str | None = None
        self.confidence_postfix: str | None = None
        self.only_yx: bool | None = None
        self._query_imputed: AnnData | None = None
        self.expression_transfer_metrics: dict[str, Any] | None = None

    def __repr__(self):
        """Return a concise string representation of the CellMapper object."""
        query_summary = f"AnnData(n_obs={self.query.n_obs:,}, n_vars={self.query.n_vars:,})"

        if self._is_self_mapping:
            return f"CellMapper(self-mapping, data={query_summary}, "
        else:
            reference_summary = f"AnnData(n_obs={self.reference.n_obs:,}, n_vars={self.reference.n_vars:,})"
            return f"CellMapper(query={query_summary}, reference={reference_summary}"

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
        n_reference_cells
            Number of reference cells (rows).
        n_query_cells
            Number of query cells (columns).

        Returns
        -------
        Validated and row-normalized mapping matrix.
        """
        # Validate the shape
        if mapping_matrix.shape != (self.query.n_obs, self.reference.n_obs):
            raise ValueError(
                f"Mapping matrix shape mismatch: expected ({self.query.n_obs}, {self.reference.n_obs}), "
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

    def compute_joint_pca(self, n_components: int = 50, key_added: str = "X_pca_joint", **kwargs) -> None:
        """
        Compute a joint PCA on the normalized .X matrices of query and reference, using only overlapping genes.

        Parameters
        ----------
        n_components
            Number of principal components to compute.
        key_added
            Key under which to store the joint PCA embeddings in `.obsm` of both query and reference AnnData objects.
        **kwargs
            Additional keyword arguments to pass to scanpy's `pp.pca` function.

        Notes
        -----
        This method performs an inner join on genes (variables) between the query and reference AnnData objects,
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
        joint = ad.concat([self.reference, self.query], join="inner", label="batch", keys=["reference", "query"])

        # Compute PCA using scanpy
        sc.pp.pca(joint, n_comps=n_components, **kwargs)

        # Assign PCA embeddings back to each object using the batch key
        self.reference.obsm[key_added] = joint.obsm["X_pca"][joint.obs["batch"] == "reference"]
        self.query.obsm[key_added] = joint.obsm["X_pca"][joint.obs["batch"] == "query"]
        logger.info(
            "Joint PCA computed and stored as '%s' in both reference.obsm and query.obsm. "
            "Proceeding to use this as the representation for neighbor search.",
            key_added,
        )

    def compute_dual_pca(
        self,
        n_components: int = 50,
        key_added: str = "X_pca_dual",
        layer: str | None = None,
        var_mask: np.ndarray | str | None = None,
        zero_center: bool = True,
        scale_with_singular: bool = True,
        random_state: int = 0,
    ) -> None:
        """
        Compute a joint embedding using dual PCA on the reference and query datasets.

        This method computes the singular value decomposition (SVD) of the cross-covariance matrix
        between the reference and query datasets using an efficient implementation that doesn't
        materialize the full cross-covariance matrix. It then constructs a joint embedding
        based on the SVD components.

        Parameters
        ----------
        n_components
            Number of components to keep in the embedding.
        key_added
            Key under which to store the joint embedding in `.obsm` of both
            query and reference AnnData objects.
        layer
            Layer to use for the computation. If None, use .X.
        var_mask
            Boolean mask or string mask identifying a subset of variables/genes to use
            for computation. If string, should be a key in .var for both query and reference
            that contains a boolean mask. If None, uses the intersection of all variables
            from both datasets.
        zero_center
            If True, center the data (implicitly for sparse matrices).
        scale_with_singular
            If True (default), scale the singular vectors by the square root of their
            singular values. If False, return the raw singular vectors.
        random_state
            Random seed for reproducibility.

        Returns
        -------
        None
            Embeddings are stored in the .obsm attribute of both query and reference AnnData objects.

        Notes
        -----
        This method follows the approach described in https://xinmingtu.cn/blog/2022/CCA_dual_PCA/
        to create embeddings from the SVD of the cross-covariance matrix between datasets. This is similar to
        Seurat's CCA implementation (CITE), but it (optionally) multiplies with the singular values to create the embeddings.

        In contrast to the existing implementation in the SLAT (CITE) package, we don't compute the covariance matrix
        explicitly, which saves memory and is more efficient for large datasets.

        The implementation handles sparse matrices efficiently using implicit mean centering,
        similar to Scanpy's PCA implementation.
        """
        logger.info(
            "Computing dual PCA between query (%d cells) and reference (%d cells).",
            self.query.n_obs,
            self.reference.n_obs,
        )

        # Handle var_mask parameter
        if var_mask is not None:
            # If var_mask is a string, interpret it as a key in .var
            if isinstance(var_mask, str):
                if var_mask not in self.query.var or var_mask not in self.reference.var:
                    raise ValueError(
                        f"var_mask '{var_mask}' not found in both query.var and reference.var. "
                        "The mask must be present in both datasets."
                    )
                query_mask = self.query.var[var_mask].values
                ref_mask = self.reference.var[var_mask].values

                # Get masked genes from both datasets
                query_genes = self.query.var_names[query_mask]
                ref_genes = self.reference.var_names[ref_mask]

                # Find common genes between the masked sets
                common_genes = np.intersect1d(query_genes, ref_genes)

                logger.info(
                    "Using var_mask '%s': %d genes in query, %d genes in reference, %d common genes.",
                    var_mask,
                    np.sum(query_mask),
                    np.sum(ref_mask),
                    len(common_genes),
                )
            # If var_mask is a boolean array
            elif isinstance(var_mask, np.ndarray):
                if len(var_mask) != len(self.query.var_names) or len(var_mask) != len(self.reference.var_names):
                    raise ValueError(
                        f"Length of var_mask ({len(var_mask)}) must match the number of variables "
                        f"in both query ({self.query.n_vars}) and reference ({self.reference.n_vars})."
                    )
                # Apply the boolean mask to get gene names
                query_genes = self.query.var_names[var_mask]
                ref_genes = self.reference.var_names[var_mask]
                common_genes = np.intersect1d(query_genes, ref_genes)

                logger.info(
                    "Using provided boolean var_mask: %d genes selected, %d common genes.",
                    np.sum(var_mask),
                    len(common_genes),
                )
            else:
                raise TypeError(f"var_mask must be a string key in .var or a boolean array, got {type(var_mask)}.")
        else:
            # Default behavior: use intersection of all genes
            common_genes = np.intersect1d(self.query.var_names, self.reference.var_names)
            logger.info("Using all %d common genes for dual PCA.", len(common_genes))

        if len(common_genes) == 0:
            raise ValueError("No overlapping genes found between query and reference datasets.")

        logger.info("Using %d common genes for dual PCA.", len(common_genes))

        # Extract the expression matrices for the common genes
        if layer is None:
            X_query = self.query[:, common_genes].X
            X_ref = self.reference[:, common_genes].X
        else:
            X_query = self.query[:, common_genes].layers[layer]
            X_ref = self.reference[:, common_genes].layers[layer]

        # Check sparsity types match
        both_sparse = issparse(X_query) and issparse(X_ref)
        both_dense = not issparse(X_query) and not issparse(X_ref)

        if not (both_sparse or both_dense):
            logger.info("Converting matrices to ensure consistent type (both sparse or both dense).")
            if issparse(X_query) and not issparse(X_ref):
                X_ref = csr_matrix(X_ref)
            elif not issparse(X_query) and issparse(X_ref):
                X_query = X_query.toarray() if hasattr(X_query, "toarray") else X_query

        # Compute the SVD of the cross-covariance matrix
        U, s, Vt = truncated_svd_cross_covariance(
            X_query, X_ref, n_components=n_components, zero_center=zero_center, random_state=random_state
        )

        logger.info("SVD of cross-covariance matrix computed successfully.")

        # Construct embeddings following dual PCA formulation
        # X = U * sqrt(S)
        # Y = V * sqrt(S) = (Vt.T) * sqrt(S)
        if scale_with_singular:
            s_sqrt = np.sqrt(s)
            query_embedding = U * s_sqrt[np.newaxis, :]
            reference_embedding = Vt.T * s_sqrt[np.newaxis, :]
        else:
            query_embedding = U
            reference_embedding = Vt.T

        # Store embeddings in the AnnData objects
        self.query.obsm[key_added] = query_embedding
        self.reference.obsm[key_added] = reference_embedding

        # Store variance explained in uns
        explained_variance_ratio = s / np.sum(s)

        self.query.uns[f"{key_added}_params"] = {
            "n_components": n_components,
            "variance": s,
            "variance_ratio": explained_variance_ratio,
            "n_common_genes": len(common_genes),
            "zero_center": zero_center,
            "scale_with_singular": scale_with_singular,
            "var_mask": var_mask if isinstance(var_mask, str) else None,
        }

        # Reference dataset gets the same parameters
        self.reference.uns[f"{key_added}_params"] = self.query.uns[f"{key_added}_params"]

        logger.info(
            "Dual PCA embeddings stored as '%s' in both reference.obsm and query.obsm. "
            "Top %d components explain %.1f%% of the cross-covariance.",
            key_added,
            n_components,
            100 * np.sum(explained_variance_ratio),
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
            All methods return exactly `n_neighbors` neighbors, including the reference cell itself (in self-mapping mode). For faiss and sklearn, distances to self are very small positive numbers, for rapids and sklearn, they are exactly 0.
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
        self.only_yx = only_yx
        if use_rep is None:
            if pca_kwargs is None:
                pca_kwargs = {}
            if self._is_self_mapping:
                sc.pp.pca(self.query, n_comps=n_pca_components, **pca_kwargs)
                use_rep = "X_pca"
            else:
                self.compute_joint_pca(n_components=n_pca_components, key_added=joint_pca_key, **pca_kwargs)
                use_rep = joint_pca_key
        if use_rep == "X":
            xrep = self.reference.X
            yrep = self.query.X
        else:
            xrep = self.reference.obsm[use_rep]
            yrep = self.query.obsm[use_rep]
        self.knn = Neighbors(np.ascontiguousarray(xrep), np.ascontiguousarray(yrep))
        self.knn.compute_neighbors(n_neighbors=n_neighbors, method=method, metric=metric, only_yx=only_yx)

    def compute_mappping_matrix(
        self,
        method: Literal["jaccard", "gaussian", "scarches", "inverse_distance", "random", "hnoca", "equal"] = "gaussian",
    ) -> None:
        """
        Compute the mapping matrix for label transfer.

        Parameters
        ----------
        method
            Method to use for computing the mapping matrix. Options include:

            - "jaccard": Jaccard similarity. Inspired by GLUE :cite:`cao2022multi`
            - "gaussian": Gaussian kernel with (global) bandwith equal to the mean distance.
            - "scarches": scArches kernel. Inspired by scArches :cite:`lotfollahi2022mapping`
            - "inverse_distance": Inverse distance kernel.
            - "random": Random kernel, useful for testing.
            - "hnoca": HNOCA kernel. Inspired by HNOCA-tools :cite:`he2024integrated`
            - "equal": All neighbors are equally weighted (1/n_neighbors).

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``mapping_matrix``: Mapping matrix for label transfer.
        """
        if self.knn is None:
            raise ValueError("Neighbors have not been computed. Call compute_neighbors() first.")

        logger.info("Computing mapping matrix using method '%s'.", method)
        if method in ["jaccard", "hnoca"]:
            if self.only_yx:
                raise ValueError(
                    "Jaccard and HNOCa methods require both x and y neighbors to be computed. Set only_yx=False."
                )
            xx, yy, xy, yx = self.knn.get_adjacency_matrices()
            n_neighbors = self.knn.xx.n_neighbors
            jaccard = (yx @ xx.T) + (yy @ xy.T)

            if method == "jaccard":
                jaccard.data /= 4 * n_neighbors - jaccard.data
            elif method == "hnoca":
                jaccard.data /= 2 * n_neighbors - jaccard.data
                jaccard.data = jaccard.data**2
            self.mapping_matrix = jaccard
        elif method in ["gaussian", "scarches", "inverse_distance", "random", "equal"]:
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
            One or more keys in ``reference.obs`` to be transferred into ``query.obs`` (must be discrete)
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
                self.reference.obs[[key]],
            )  # shape = (n_reference_cells x n_categories), sparse csr matrix, float32
            ytab = self.mapping_matrix @ xtab  # shape = (n_query_cells x n_categories), sparse crs matrix, float32

            pred = pd.Series(
                data=np.array(onehot.categories_[0])[ytab.argmax(axis=1).A1],
                index=self.query.obs_names,
                dtype=self.reference.obs[key].dtype,
            )
            conf = pd.Series(
                ytab.max(axis=1).toarray().ravel(),
                index=self.query.obs_names,
            )

            self.query.obs[f"{key}_{self.prediction_postfix}"] = pred
            self.query.obs[f"{key}_{self.confidence_postfix}"] = conf

            # Add colors if available
            if f"{key}_colors" in self.reference.uns:
                color_lookup = dict(
                    zip(self.reference.obs[key].cat.categories, self.reference.uns[f"{key}_colors"], strict=True)
                )
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
            One or more keys in ``reference.obsm`` storing the embeddings to be transferred.
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
            reference_embeddings = self.reference.obsm[key]  # shape = (n_reference_cells x n_embedding_dims)
            query_embeddings = self.mapping_matrix @ reference_embeddings  # shape = (n_query_cells x n_embedding_dims)

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
            Key in ``reference.layers`` to be transferred. Use "X" to transfer ``reference.X``.

        Returns
        -------
        None

        Notes
        -----
        Creates ``self.query_imputed`` with the transferred data in .X.
        The new AnnData object will have the same cells as the query, but the features (genes) of the reference.
        """
        if self.mapping_matrix is None:
            raise ValueError("Mapping matrix has not been computed. Call compute_mapping_matrix() first.")

        logger.info("Transferring layer for key '%s'.", layer_key)
        # Get the reference layer (or .X if key is "X")
        reference_layer = self.reference.X if layer_key == "X" else self.reference.layers[layer_key]
        query_layer = self.mapping_matrix @ reference_layer  # shape = (n_query_cells x n_reference_features)

        # Create query_imputed using the property setter for consistent behavior
        self.query_imputed = query_layer

        # Create base message and conditionally add note about feature spaces for non-self-mapping
        message = f"Imputed expression for layer '{layer_key}' stored in query_imputed.X."
        if not self._is_self_mapping:
            message += f"\nNote: The feature space matches the reference (n_vars={self.reference.n_vars}), not the query (n_vars={self.query.n_vars})."

        logger.info(message)

    @property
    def query_imputed(self) -> AnnData | None:
        """
        Get the imputed query data.

        Returns
        -------
        AnnData or None
            The imputed query data as an AnnData object, or None if not set.
        """
        return self._query_imputed

    @query_imputed.setter
    def query_imputed(self, value: AnnData | np.ndarray | csr_matrix | pd.DataFrame | None) -> None:
        """
        Set the imputed query data with automatic alignment and validation.

        This property allows flexibly setting imputed data as:
        - An AnnData object
        - A numpy array or sparse matrix
        - A pandas DataFrame

        The setter automatically constructs an AnnData object with proper alignment:
        - Observations (obs, obsm) from the query dataset
        - Features (var, varm) from the reference dataset

        Parameters
        ----------
        value
            The imputed query data to set. Can be AnnData, numpy array, sparse matrix,
            pandas DataFrame, or None to unset.
        """
        if value is None:
            self._query_imputed = None
            return

        # Let the utility function handle all validation and conversion
        self._query_imputed = create_imputed_anndata(
            expression_data=value, query_adata=self.query, reference_adata=self.reference
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
            One or more keys in ``reference.obs`` to be transferred into ``query.obs`` (must be discrete)
        obsm_keys
            One or more keys in ``reference.obsm`` storing the embeddings to be transferred.
        layer_key
            Key in ``reference.layers`` to be transferred. Use "X" to transfer ``reference.X``.
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

        # Create a neighbors object using the factory method
        self.knn = Neighbors.from_distances(distances_matrix, include_self=include_self)

        logger.info(
            "Loaded precomputed distances from '%s' with %d cells and %d neighbors per cell.",
            distances_key,
            distances_matrix.shape[0],
            self.knn.xx.n_neighbors,
        )
