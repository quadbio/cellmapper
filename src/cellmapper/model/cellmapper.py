"""k-NN based mapping of labels, embeddings, and expression values."""

from typing import Any, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import OneHotEncoder

from cellmapper._docs import d
from cellmapper.constants import PackageConstants
from cellmapper.logging import logger
from cellmapper.model.embedding import EmbeddingMixin
from cellmapper.model.evaluate import EvaluationMixin
from cellmapper.model.kernel import Kernel
from cellmapper.model.mapping_operator import MappingOperator
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
        if reference is None:
            self.reference = query
            self._is_self_mapping = True
        elif reference is query:
            # Same object passed twice - treat as self-mapping
            logger.warning(
                "The same AnnData object was passed as both query and reference. Initializing in self-mapping mode."
            )
            self.reference = query
            self._is_self_mapping = True
        else:
            self.reference = reference
            self._is_self_mapping = False

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
        self.knn: Kernel | None = None
        self._mapping_operator: MappingOperator | None = None
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
    def mapping_operator(self) -> MappingOperator:
        """
        Get the mapping operator for applying matrix powers.

        The mapping operator encapsulates the mapping matrix and provides methods
        for applying matrix powers M^t for t-step diffusion processes.

        Returns
        -------
        MappingOperator
            The mapping operator containing the validated and normalized mapping matrix

        Raises
        ------
        ValueError
            If the mapping matrix has not been computed yet
        """
        if self._mapping_operator is None:
            raise ValueError("Mapping matrix has not been computed. Call compute_mapping_matrix() first.")
        return self._mapping_operator

    @d.dedent
    def compute_neighbors(
        self,
        n_neighbors: int = 30,
        use_rep: str | None = None,
        n_comps: int | None = None,
        knn_method: Literal["sklearn", "pynndescent", "rapids", "faiss-cpu", "faiss-gpu"] = "sklearn",
        knn_dist_metric: str = "euclidean",
        random_state: int = 0,
        only_yx: bool = False,
        neighbors_kwargs: dict[str, Any] | None = None,
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
        %(n_neighbors)s
        %(use_rep)s
        n_comps
            Number of components to use. If a pre-computed representation is provided via `use_rep`,
            we will use the number of components from that representation. Otherwiese, if `use_rep=None`,
            we will compute the given number of components using the fallback representation method.
        %(knn_method)s
        %(knn_dist_metric)s
        random_state
            Random seed for reproducibility. Only used by "pynndescent" method.
        %(only_yx)s
        neighbors_kwargs
            Additional keyword arguments to pass to the neighbors computation method.
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
            if self._is_self_mapping:
                logger.warning(
                    "No representation provided (`use_rep=None`) and self-mapping mode detected. Computing a joint representation automatically using PCA."
                )
                key_added = fallback_kwargs.pop("key_added", "X_pca")
                sc.tl.pca(self.query, n_comps=n_comps, key_added=key_added, **fallback_kwargs)
            else:
                logger.warning(
                    "No representation provided (`use_rep=None`). Computing a joint representation automatically "
                    "using '%s'. For optimal results, consider pre-computing a representation and passing it to `use_rep`.",
                    fallback_representation,
                )
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

        self.knn = Kernel(
            np.ascontiguousarray(xrep),
            None if self._is_self_mapping else np.ascontiguousarray(yrep),
            is_self_mapping=self._is_self_mapping,
        )
        self.knn.compute_neighbors(
            n_neighbors=n_neighbors,
            knn_method=knn_method,
            knn_dist_metric=knn_dist_metric,
            only_yx=self.only_yx,
            random_state=random_state,
            **(neighbors_kwargs or {}),
        )

    @d.dedent
    def compute_mapping_matrix(
        self,
        kernel_method: Literal[
            "jaccard",
            "gauss",
            "scarches",
            "inverse_distance",
            "random",
            "hnoca",
            "equal",
            "umap",
        ]
        | None = None,
        symmetrize: bool | None = None,
        self_edges: bool | None = None,
        n_eigenvectors: int = 50,
        eigen_solver: Literal["partial", "complete"] = "partial",
    ) -> None:
        """
        Compute the mapping matrix for label transfer.

        Parameters
        ----------
        %(kernel_method)s
        %(symmetrize)s
        %(self_edges)s
        n_eigenvectors
            Number of eigenvectors to compute for spectral decomposition. Only relevant
            when using spectral methods for matrix powers. Default is 50.
        eigen_solver
            Eigendecomposition method for spectral approach:
            - "partial": Uses sparse eigendecomposition, faster (default)
            - "complete": Uses complete eigendecomposition, exact for testing

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``mapping_operator``: Mapping operator to transfer labels, embeddings, or expression values.
        """
        if self.knn is None:
            raise ValueError("Neighbors have not been computed. Call compute_neighbors() first.")
        assert self.knn.yx is not None, "Neighbors object must have yx neighbors computed."

        # Default mapping method if not provided
        if kernel_method is None:
            kernel_method = (
                PackageConstants.DEFAULT_SELF_MAPPING_KERNEL_METHOD
                if self._is_self_mapping
                else PackageConstants.DEFAULT_CROSS_MAPPING_KERNEL_METHOD  # type: ignore[assignment]
            )
        # Set defaults for symmetrize
        if symmetrize is None:
            symmetrize = self._is_self_mapping  # True for self-mapping, False for cross-mapping
        if self_edges is None:
            self_edges = self._is_self_mapping  # same

        logger.info("Computing mapping matrix using kernel method '%s'.", kernel_method)

        # Compute kernel matrix using the new unified method
        self.knn.compute_kernel_matrix(
            kernel_method=kernel_method,
            symmetrize=symmetrize,
            self_edges=self_edges,
        )

        # Validate expected shape before creating mapping operator
        expected_shape = (self.query.n_obs, self.reference.n_obs)
        actual_shape = self.knn.kernel_matrix.shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"Kernel matrix shape {actual_shape} does not match expected shape {expected_shape}. "
                f"Expected ({self.query.n_obs} query cells, {self.reference.n_obs} reference cells)."
            )

        # Create mapping operator with the computed matrix (simplified interface)
        self._mapping_operator = MappingOperator(
            kernel_matrix=self.knn,  # Pass the Kernel object directly
            n_eigenvectors=n_eigenvectors,
            eigen_solver=eigen_solver,
        )

    @d.dedent
    def map_obsm(
        self,
        key: str,
        t: int | None = None,
        diffusion_method: Literal["iterative", "spectral"] = "iterative",
        prediction_postfix: str = "pred",
    ) -> None:
        """
        Map embeddings with optional multi-step diffusion.

        Uses matrix multiplication to transfer embeddings from the reference
        dataset to the query dataset. For t > 1, applies matrix powers representing
        t-step diffusion processes (only supported in self-mapping mode).

        When the reference embeddings are stored as a pandas DataFrame, the diffusion_method
        preserves the DataFrame structure by reconstructing it with the query cell
        index and the original column names after mapping.

        Parameters
        ----------
        key
            Key in ``reference.obsm`` storing the embeddings to be transferred
        %(t)s
        %(diffusion_method)s
        %(prediction_postfix)s

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``query.obsm``: Contains the transferred embeddings. If the reference embeddings
          were a pandas DataFrame, the transferred embeddings will also be a DataFrame
          with the same column names and the query cell names as the index.
        """
        if self._mapping_operator is None:
            raise ValueError("Mapping matrix has not been computed. Call compute_mapping_matrix() first.")

        # Log the actual operation being performed
        if t is None:
            logger.info("Mapping embeddings for key '%s' using direct multiplication", key)
        else:
            logger.info(
                "Mapping embeddings for key '%s' with t=%d steps using %s diffusion_method", key, t, diffusion_method
            )

        # Perform matrix power multiplication to transfer embeddings
        reference_data = self.reference.obsm[key]  # shape = (n_reference_cells x n_embedding_dims)

        # Handle pandas DataFrame case
        if isinstance(reference_data, pd.DataFrame):
            # Extract values for the mapping operation
            columns = reference_data.columns
            reference_values = reference_data.values
            is_dataframe = True
        else:
            reference_values = reference_data  # type: ignore[assignment]
            is_dataframe = False

        # Apply matrix power while preserving sparsity
        query_data = self.mapping_operator.apply(
            reference_values,
            t=t,
            diffusion_method=diffusion_method,
        )  # shape = (n_query_cells x n_embedding_dims)

        if is_dataframe:
            query_data = pd.DataFrame(
                data=query_data.toarray() if issparse(query_data) else query_data,  # type: ignore[attr-defined]
                index=self.query.obs_names,
                columns=columns,
            )

        # Store the transferred embeddings in query.obsm with descriptive key
        output_key = f"{key}_{prediction_postfix}"
        self.query.obsm[output_key] = query_data
        logger.info("Embeddings mapped and stored in query.obsm['%s']", output_key)

    @d.dedent
    def map_layers(
        self, key: str, t: int | None = None, diffusion_method: Literal["iterative", "spectral"] = "iterative"
    ) -> None:
        """
        Map expression values with optional multi-step diffusion.

        Transfers expression values (e.g., .X or entries from .layers) from reference
        dataset to a new imputed query AnnData object using matrix multiplication.
        For t > 1, applies matrix powers representing t-step diffusion processes
        (only supported in self-mapping mode).

        Parameters
        ----------
        key
            Key in ``reference.layers`` to be transferred. Use "X" to transfer ``reference.X``
        %(t)s
        %(diffusion_method)s

        Returns
        -------
        None

        Notes
        -----
        Creates ``self.query_imputed`` with the transferred data in .X.
        The new AnnData object will have the same cells as the query, but the features (genes) of the reference.
        """
        if self._mapping_operator is None:
            raise ValueError("Mapping matrix has not been computed. Call compute_mapping_matrix() first.")

        # Log the actual operation being performed
        if t is None:
            logger.info("Mapping layer for key '%s' using direct multiplication", key)
        else:
            logger.info(
                "Mapping layer for key '%s' with t=%d steps using %s diffusion_method", key, t, diffusion_method
            )

        # Get the reference layer (or .X if key is "X")
        reference_layer = self.reference.X if key == "X" else self.reference.layers[key]

        # Apply matrix power while preserving sparsity
        query_layer = self.mapping_operator.apply(
            reference_layer, t=t, diffusion_method=diffusion_method
        )  # shape = (n_query_cells x n_reference_features)

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

    @d.dedent
    def map(
        self,
        obs_keys: str | list[str] | None = None,
        obsm_keys: str | list[str] | None = None,
        layer_key: str | None = None,
        t: int | None = None,
        diffusion_method: Literal["iterative", "spectral"] = "iterative",
        n_neighbors: int = 30,
        use_rep: str | None = None,
        knn_method: Literal["sklearn", "pynndescent", "rapids"] = "sklearn",
        knn_dist_metric: str = "euclidean",
        only_yx: bool = False,
        kernel_method: Literal[
            "jaccard",
            "gauss",
            "scarches",
            "inverse_distance",
            "random",
            "hnoca",
            "equal",
            "umap",
        ]
        | None = None,
        symmetrize: bool | None = None,
        self_edges: bool | None = None,
        prediction_postfix: str = "pred",
        subset_categories: None | list[str] | str = None,
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
        %(t)s
        %(diffusion_method)s
        %(n_neighbors)s
        %(use_rep)s
        %(knn_method)s
        %(knn_dist_metric)s
        %(only_yx)s
        %(kernel_method)s
        %(symmetrize)s
        %(self_edges)s
        %(prediction_postfix)s
        %(subset_categories)s
        """
        if self.knn is None:
            self.compute_neighbors(
                n_neighbors=n_neighbors,
                use_rep=use_rep,
                knn_method=knn_method,
                knn_dist_metric=knn_dist_metric,
                only_yx=only_yx,
            )
        if self._mapping_operator is None:
            self.compute_mapping_matrix(kernel_method=kernel_method, symmetrize=symmetrize, self_edges=self_edges)

        if obs_keys is not None:
            # Normalize to list for consistent handling
            obs_keys_list = [obs_keys] if isinstance(obs_keys, str) else obs_keys
            for obs_key in obs_keys_list:
                self.map_obs(
                    key=obs_key,
                    t=t,
                    diffusion_method=diffusion_method,
                    prediction_postfix=prediction_postfix,
                    subset_categories=subset_categories,
                )
        if obsm_keys is not None:
            # Normalize to list for consistent handling
            obsm_keys_list = [obsm_keys] if isinstance(obsm_keys, str) else obsm_keys
            for obsm_key in obsm_keys_list:
                self.map_obsm(
                    key=obsm_key, t=t, diffusion_method=diffusion_method, prediction_postfix=prediction_postfix
                )
        if layer_key is not None:
            self.map_layers(key=layer_key, t=t, diffusion_method=diffusion_method)
        if obs_keys is None and obsm_keys is None and layer_key is None:
            logger.warning(
                "Neither ``obs_keys``, ``obsm_keys`` or ``layer_key`` provided. No labels, embeddings or layers were transferred. "
                "Please provide at least one of ``obs_keys``, ``obsm_keys`` or ``layer_key``."
            )

        return self

    def load_precomputed_distances(self, distances_key: str = "distances", remove_last_neighbor: bool = False) -> None:
        """
        Load precomputed distances from the AnnData object.

        This method is only available in self-mapping mode.

        Parameters
        ----------
        distances_key
            Key in adata.obsp where the precomputed distances are stored.
        remove_last_neighbor
            If True, removes the last neighbor from the distances matrix.
            This is useful for direct comparisons with scanpy.

        Returns
        -------
        None

        Notes
        -----
        Updates the following attributes:

        - ``knn``: Neighbors object constructed from the precomputed distances.

        For symmetrization of connectivity matrices, use the ``symmetrize`` parameter
        in ``compute_mapping_matrix()`` after loading the distances.
        """
        if not self._is_self_mapping:
            raise ValueError("load_precomputed_distances is only available in self-mapping mode.")

        # Access the precomputed distances
        if distances_key not in self.query.obsp:
            raise ValueError(f"No distances found at key '{distances_key}' in query.obsp")

        distances_matrix = self.query.obsp[distances_key]
        if distances_matrix is None:
            raise ValueError(f"Distances matrix at key '{distances_key}' is None")

        # Store shape before potential conversion
        n_cells = distances_matrix.shape[0]

        # Convert to csr_matrix if not already
        if not isinstance(distances_matrix, csr_matrix):
            distances_matrix = csr_matrix(distances_matrix)

        # Create a neighbors object using the factory method
        self.knn = Kernel.from_distances(distances_matrix, remove_last_neighbor)

        # Type assertion for mypy - from_distances creates a valid neighbors object with xx
        assert self.knn.xx is not None
        logger.info(
            "Loaded precomputed distances from '%s' with %d cells and %d neighbors per cell.",
            distances_key,
            n_cells,
            self.knn.xx.n_neighbors,
        )

    @d.dedent
    def map_obs(
        self,
        key: str,
        t: int | None = None,
        diffusion_method: Literal["iterative", "spectral"] = "iterative",
        prediction_postfix: str = "pred",
        confidence_postfix: str = "conf",
        return_probabilities: bool = False,
        subset_categories: None | list[str] | str = None,
    ) -> np.ndarray | csr_matrix | None:
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
        %(t)s
        %(diffusion_method)s
        %(prediction_postfix)s
        confidence_postfix
            Postfix added to create new keys in ``query.obs`` for confidence scores
            (only applicable for categorical data)
        return_probabilities
            If True, return the probability matrix for categorical data.
            Only applicable for categorical data. The matrix is never densified.
        %(subset_categories)s

        Returns
        -------
        np.ndarray, csr_matrix or None
            For categorical data with ``return_probabilities=True``: dense or sparse matrix
            of shape (n_query_cells, n_categories) containing probabilities.
            For numerical data or when ``return_probabilities=False``: None.

        Notes
        -----
        Updates the following attributes:

        - ``query.obs``: Contains the transferred data and confidence scores (for categorical data).
        """
        if self._mapping_operator is None:
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

        # Handle subset_categories parameter and warnings
        if subset_categories is not None:
            if not is_categorical:
                logger.warning(
                    "subset_categories parameter specified for numerical data in key '%s'. This parameter will be ignored for numerical data.",
                    key,
                )
                subset_categories = None
            else:
                # Convert single string to list for consistent handling
                if isinstance(subset_categories, str):
                    subset_categories = [subset_categories]

                # Check if specified categories exist in the data
                available_categories = set(
                    reference_data.cat.categories if hasattr(reference_data, "cat") else reference_data.unique()
                )
                invalid_categories = set(subset_categories) - available_categories

                if invalid_categories:
                    logger.warning(
                        "Some specified categories for key '%s' do not exist in the data and will be ignored: %s. Available categories: %s",
                        key,
                        list(invalid_categories),
                        list(available_categories),
                    )
                    # Filter out invalid categories
                    subset_categories = [cat for cat in subset_categories if cat in available_categories]

                    # If no valid categories remain, set to None to use all
                    if not subset_categories:
                        logger.warning(
                            "No valid categories remaining for key '%s' after filtering. Using all available categories.",
                            key,
                        )
                        subset_categories = None

        # Log the operation being performed
        data_type = "categorical" if is_categorical else "numerical"
        if t is None:
            logger.info("Mapping %s data for key '%s' using direct multiplication.", data_type, key)
        else:
            logger.info(
                "Mapping %s data for key '%s' with t=%d steps using %s diffusion_method.",
                data_type,
                key,
                t,
                diffusion_method,
            )

        if is_categorical:
            return self._map_obs_categorical(
                key,
                prediction_postfix,
                confidence_postfix,
                t,
                diffusion_method,
                return_probabilities,
                subset_categories,
            )
        else:
            if return_probabilities:
                logger.warning("return_probabilities=True is only applicable for categorical data, ignoring.")
            self._map_obs_numerical(key, prediction_postfix, t, diffusion_method)
            return None

    def _map_obs_categorical(
        self,
        key: str,
        prediction_postfix: str,
        confidence_postfix: str,
        t: int | None,
        diffusion_method: Literal["iterative", "spectral"],
        return_probabilities: bool = False,
        subset_categories: None | list[str] = None,
    ) -> np.ndarray | csr_matrix | None:
        """Map categorical observation data using one-hot encoding."""
        # Get the reference data
        reference_data = self.reference.obs[key]

        if subset_categories is not None:
            # Create a filtered version of reference data for one-hot encoding
            # Only include rows that have the desired categories
            mask = reference_data.isin(subset_categories)

            # Create a filtered DataFrame with only the subset categories
            filtered_reference_data = reference_data.copy()
            filtered_reference_data[~mask] = pd.NA  # Set non-subset categories to missing

            # Create one-hot encoding only for the subset categories
            onehot = OneHotEncoder(dtype=np.float32, handle_unknown="ignore")
            # Create a DataFrame with only subset categories for fitting
            subset_df = pd.DataFrame({key: pd.Categorical(subset_categories, categories=subset_categories)})
            onehot.fit(subset_df)

            # Transform the full reference data (missing values will be ignored)
            xtab = onehot.transform(filtered_reference_data.to_frame())
        else:
            # Use the original approach for all categories
            onehot = OneHotEncoder(dtype=np.float32)
            xtab = onehot.fit_transform(self.reference.obs[[key]])

        # Apply the mapping
        ytab = self.mapping_operator.apply(
            xtab, t=t, diffusion_method=diffusion_method
        )  # shape = (n_query_cells x n_categories), sparse csr matrix, float32

        pred = pd.Series(
            data=np.array(onehot.categories_[0])[ytab.argmax(axis=1).A1 if issparse(ytab) else ytab.argmax(axis=1)],
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
                color_lookup.get(cat, "#383838") for cat in pred.cat.categories
            ]

        logger.info("Categorical data mapped and stored in query.obs['%s'].", f"{key}_{prediction_postfix}")

        # Return probabilities if requested (never densify)
        if return_probabilities:
            return ytab
        else:
            return None

    def _map_obs_numerical(
        self, key: str, prediction_postfix: str, t: int | None, diffusion_method: Literal["iterative", "spectral"]
    ) -> None:
        """Map numerical observation data using direct matrix multiplication."""
        reference_values = np.array(self.reference.obs[key]).reshape(-1, 1)  # shape = (n_reference_cells, 1)
        mapped_values = self.mapping_operator.apply(
            reference_values, t=t, diffusion_method=diffusion_method
        )  # shape = (n_query_cells, 1)

        pred = pd.Series(
            data=mapped_values.ravel(),
            index=self.query.obs_names,
        )

        self.query.obs[f"{key}_{prediction_postfix}"] = pred

        logger.info("Numerical data mapped and stored in query.obs['%s'].", f"{key}_{prediction_postfix}")
