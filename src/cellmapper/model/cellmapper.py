"""k-NN based mapping of labels, embeddings, and expression values."""

import gc
from typing import Any, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from sklearn.preprocessing import OneHotEncoder

from cellmapper.logging import logger
from cellmapper.model.embedding import EmbeddingMixin
from cellmapper.model.evaluate import EvaluationMixin
from cellmapper.model.knn import Neighbors
from cellmapper.utils import create_imputed_anndata, get_n_comps


class CellMapper(EvaluationMixin, EmbeddingMixin):
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

    def compute_neighbors(
        self,
        n_neighbors: int = 30,
        use_rep: str | None = None,
        n_comps: int | None = None,
        method: Literal["sklearn", "pynndescent", "rapids", "faiss"] = "sklearn",
        metric: str = "euclidean",
        only_yx: bool = False,
        fallback_representation: Literal["fast_cca", "joint_pca"] = "fast_cca",
        fallback_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Compute nearest neighbors between reference and query datasets.

        The method computes k-nearest neighbor graphs to enable mapping between
        datasets. If no representation is provided (`use_rep=None`), a fallback
        representation will be computed automatically using either fast CCA
        ,inspired by Seurat v3 :cite:`stuart2019comprehensive`), or joint PCA. In self-mapping mode,
        a simple PCA will be computed on the query dataset.

        Parameters
        ----------
        n_neighbors
            Number of nearest neighbors.
        use_rep
            Data representation based on which to find nearest neighbors. If None,
            a fallback representation will be computed automatically.
        n_comps
            Number of components to use. If a pre-computed representation is provided via `use_rep`,
            we will use the number of components from that representation. Otherwiese, if `use_rep=None`,
            we will compute the given number of components using the fallback representation method.
        method
            Method to use for computing neighbors. "sklearn" and "pynndescent" run on CPU,
            "rapids" and "faiss" run on GPU. Note that all but "pynndescent" perform exact
            neighbor search. With GPU acceleration, "faiss" is usually fastest and more
            memory efficient than "rapids". All methods return exactly `n_neighbors` neighbors,
            including the reference cell itself (in self-mapping mode). For faiss and sklearn,
            distances to self are very small positive numbers, for rapids and sklearn, they are exactly 0.
        metric
            Distance metric to use for nearest neighbors.
        only_yx
            If True, only compute the xy neighbors. This is faster, but not suitable for
            Jaccard or HNOCA methods.
        fallback_representation
            Method to use for computing a cross-dataset representation when `use_rep=None`. Options:

            - "fast_cca": Fast canonical correlation analysis, inspired by Seurat v3 :cite:`stuart2019comprehensive` and
              SLAT :cite:`xia2023spatial`).
            - "joint_pca": Principal component analysis on concatenated datasets.
        fallback_kwargs
            Additional keyword arguments to pass to the fallback representation method.
            For "fast_cca": see :meth:`~cellmapper.EmbeddingMixin.compute_fast_cca`.
            For "joint_pca": see :meth:`~cellmapper.EmbeddingMixin.compute_joint_pca`.

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
        # Handle backward compatibility parameters
        if fallback_kwargs is None:
            fallback_kwargs = {}

        self.only_yx = only_yx

        if use_rep is None:
            logger.warning(
                "No representation provided (`use_rep=None`). Computing a joint representation automatically "
                "using '%s'. For optimal results, consider pre-computing a representation and passing it to `use_rep`.",
                fallback_representation,
            )

            if self._is_self_mapping:
                logger.info("Self-mapping detected. Computing PCA on query dataset for representation.")
                key_added = fallback_kwargs.pop("key_added", "X_pca")
                sc.tl.pca(self.query, n_comps=n_comps, key_added=key_added, **fallback_kwargs)
            else:
                if fallback_representation == "fast_cca":
                    key_added = fallback_kwargs.pop("key_added", "X_cca")
                    self.compute_fast_cca(n_comps=n_comps, key_added=key_added, **fallback_kwargs)
                elif fallback_representation == "joint_pca":
                    key_added = fallback_kwargs.pop("key_added", "X_pca")
                    self.compute_joint_pca(n_comps=n_comps, key_added=key_added, **fallback_kwargs)
                else:
                    raise ValueError(
                        f"Unknown fallback_representation: {fallback_representation}. "
                        "Supported options are 'fast_cca' and 'joint_pca'."
                    )
            use_rep = key_added

        # Extract the representation from the query and reference datasets
        if use_rep == "X":
            xrep = self.reference.X
            yrep = self.query.X
        else:
            xrep = self.reference.obsm[use_rep]
            yrep = self.query.obsm[use_rep]

        # handle the number of components
        n_comps = get_n_comps(n_comps, n_vars=xrep.shape[1])
        xrep = xrep[:, :n_comps]
        yrep = yrep[:, :n_comps]

        self.knn = Neighbors(np.ascontiguousarray(xrep), np.ascontiguousarray(yrep))
        self.knn.compute_neighbors(n_neighbors=n_neighbors, method=method, metric=metric, only_yx=only_yx)

    def compute_mapping_matrix(
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

    def map_obsm(self, key: str, prediction_postfix: str = "pred") -> None:
        """
        Map embeddings from reference dataset to query dataset.

        Uses matrix multiplication to transfer embeddings from the reference
        dataset to the query dataset.

        Parameters
        ----------
        key
            Key in ``reference.obsm`` storing the embeddings to be transferred
        prediction_postfix
            Postfix to append to the output key in ``query.obsm`` where the transferred embeddings will be stored

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

        logger.info("Mapping embeddings for key '%s'.", key)

        # Perform matrix multiplication to transfer embeddings
        reference_embeddings = self.reference.obsm[key]  # shape = (n_reference_cells x n_embedding_dims)
        query_embeddings = self.mapping_matrix @ reference_embeddings  # shape = (n_query_cells x n_embedding_dims)

        # Store the transferred embeddings in query.obsm
        output_key = f"{key}_{prediction_postfix}"
        self.query.obsm[output_key] = query_embeddings

        logger.info("Embeddings mapped and stored in query.obsm['%s'].", output_key)

    def map_layers(self, key: str) -> None:
        """
        Map expression values from reference dataset to query dataset.

        Transfers expression values (e.g., .X or entries from .layers) from reference
        dataset to a new imputed query AnnData object using matrix multiplication.

        Parameters
        ----------
        key
            Key in ``reference.layers`` to be transferred. Use "X" to transfer ``reference.X``

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

        logger.info("Mapping layer for key '%s'.", key)
        # Get the reference layer (or .X if key is "X")
        reference_layer = self.reference.X if key == "X" else self.reference.layers[key]
        query_layer = self.mapping_matrix @ reference_layer  # shape = (n_query_cells x n_reference_features)

        # Create query_imputed using the property setter for consistent behavior
        self.query_imputed = query_layer

        # Create base message and conditionally add note about feature spaces for non-self-mapping
        message = f"Expression for layer '{key}' mapped and stored in query_imputed.X."
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

    def map(
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
        Map data from reference to query datasets.

        Parameters
        ----------
        obs_keys
            One or more keys in ``reference.obs`` to be mapped into ``query.obs``.
        obsm_keys
            One or more keys in ``reference.obsm`` storing the embeddings to be mapped.
        layer_key
            Key in ``reference.layers`` to be mapped. Use "X" to map ``reference.X``.
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
            Postfix added to create new keys in ``query.obs`` for the mapped labels or in ``query.obsm`` for the mapped embeddings.
        """
        self.compute_neighbors(
            n_neighbors=n_neighbors, use_rep=use_rep, method=knn_method, metric=metric, only_yx=only_yx
        )
        self.compute_mapping_matrix(method=mapping_method)
        if obs_keys is not None:
            # Handle both single key and list of keys for backward compatibility
            if isinstance(obs_keys, str):
                self.map_obs(key=obs_keys, prediction_postfix=prediction_postfix)
            else:
                for obs_key in obs_keys:
                    self.map_obs(key=obs_key, prediction_postfix=prediction_postfix)
        if obsm_keys is not None:
            # Handle both single key and list of keys for backward compatibility
            if isinstance(obsm_keys, str):
                self.map_obsm(key=obsm_keys, prediction_postfix=prediction_postfix)
            else:
                for obsm_key in obsm_keys:
                    self.map_obsm(key=obsm_key, prediction_postfix=prediction_postfix)
        if layer_key is not None:
            self.map_layers(key=layer_key)
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

    def map_obs(self, key: str, prediction_postfix: str = "pred", confidence_postfix: str = "conf") -> None:
        """
        Map observation data from reference dataset to query dataset.

        Automatically detects whether the data is categorical or numerical and applies
        the appropriate mapping strategy. For categorical data, uses one-hot encoding
        followed by matrix multiplication and argmax. For numerical data, uses direct
        matrix multiplication.

        Parameters
        ----------
        key
            Key in ``reference.obs`` to be transferred into ``query.obs``
        prediction_postfix
            Postfix added to create new keys in ``query.obs`` for the transferred data
        confidence_postfix
            Postfix added to create new keys in ``query.obs`` for confidence scores
            (only applicable for categorical data)

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``query.obs``: Contains the transferred data and confidence scores (for categorical data).
        """
        if self.mapping_matrix is None:
            raise ValueError("Mapping matrix has not been computed. Call compute_mapping_matrix() first.")

        if key not in self.reference.obs.columns:
            raise KeyError(f"Key '{key}' not found in reference.obs")

        # Set postfix attributes for compatibility with evaluation methods
        self.prediction_postfix = prediction_postfix
        self.confidence_postfix = confidence_postfix

        reference_data = self.reference.obs[key]

        # Detect data type
        is_categorical = (
            isinstance(reference_data.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(reference_data)
            or pd.api.types.is_string_dtype(reference_data)
        )

        if is_categorical:
            logger.info("Mapping categorical data for key '%s' using one-hot encoding.", key)
            self._map_obs_categorical(key, prediction_postfix, confidence_postfix)
        else:
            logger.info("Mapping numerical data for key '%s' using direct matrix multiplication.", key)
            self._map_obs_numerical(key, prediction_postfix)

    def _map_obs_categorical(self, key: str, prediction_postfix: str, confidence_postfix: str) -> None:
        """Map categorical observation data using one-hot encoding."""
        onehot = OneHotEncoder(dtype=np.float32)
        xtab = onehot.fit_transform(
            self.reference.obs[[key]],
        )  # shape = (n_reference_cells x n_categories), sparse csr matrix, float32
        ytab = self.mapping_matrix @ xtab  # shape = (n_query_cells x n_categories), sparse csr matrix, float32

        pred = pd.Series(
            data=np.array(onehot.categories_[0])[ytab.argmax(axis=1).A1],
            index=self.query.obs_names,
            dtype=self.reference.obs[key].dtype,
        )
        conf = pd.Series(
            ytab.max(axis=1).toarray().ravel(),
            index=self.query.obs_names,
        )

        self.query.obs[f"{key}_{prediction_postfix}"] = pred
        self.query.obs[f"{key}_{confidence_postfix}"] = conf

        # Add colors if available
        if f"{key}_colors" in self.reference.uns:
            color_lookup = dict(
                zip(self.reference.obs[key].cat.categories, self.reference.uns[f"{key}_colors"], strict=True)
            )
            self.query.uns[f"{key}_{prediction_postfix}_colors"] = [
                color_lookup.get(cat, "#000000") for cat in pred.cat.categories
            ]

        logger.info("Categorical data mapped and stored in query.obs['%s'].", f"{key}_{prediction_postfix}")

        # Free memory explicitly
        del onehot, xtab, ytab, pred, conf
        gc.collect()

    def _map_obs_numerical(self, key: str, prediction_postfix: str) -> None:
        """Map numerical observation data using direct matrix multiplication."""
        reference_values = np.array(self.reference.obs[key]).reshape(-1, 1)  # shape = (n_reference_cells, 1)
        mapped_values = self.mapping_matrix @ reference_values  # shape = (n_query_cells, 1)

        pred = pd.Series(
            data=mapped_values.ravel(),
            index=self.query.obs_names,
            dtype=self.reference.obs[key].dtype,
        )

        self.query.obs[f"{key}_{prediction_postfix}"] = pred

        logger.info("Numerical data mapped and stored in query.obs['%s'].", f"{key}_{prediction_postfix}")
