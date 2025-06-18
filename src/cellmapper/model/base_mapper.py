"""Base abstraction for mapping functionality across different dimensions."""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from sklearn.preprocessing import OneHotEncoder

from cellmapper.logging import logger
from cellmapper.model.knn import Neighbors
from cellmapper.utils import get_n_comps


class BaseMapper(ABC):
    """
    Abstract base class for mapping functionality across different dimensions.

    This class provides the core mapping logic that can be specialized for
    different dimensions (observations vs variables) while maintaining the
    same fundamental matrix multiplication approach.
    """

    def __init__(self, query: AnnData, reference: AnnData | None = None) -> None:
        """
        Initialize the BaseMapper.

        Parameters
        ----------
        query
            Query dataset.
        reference
            Optional reference dataset. If None, uses query for self-mapping.
        """
        self.query = query
        self.reference = reference if reference is not None else query
        self._is_self_mapping = reference is None

        # Initialize core attributes
        self.knn: Neighbors | None = None
        self._mapping_matrix: csr_matrix | None = None
        self.prediction_postfix: str | None = None
        self.confidence_postfix: str | None = None
        self.only_yx: bool | None = None

        # Log initialization
        if self._is_self_mapping:
            logger.info(
                "Initialized %s for self-mapping with %d %s.",
                self.__class__.__name__,
                self._get_query_size(),
                self._get_dimension_name(),
            )
        else:
            logger.info(
                "Initialized %s with %d query %s and %d reference %s.",
                self.__class__.__name__,
                self._get_query_size(),
                self._get_dimension_name(),
                self._get_reference_size(),
                self._get_dimension_name(),
            )

    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def _get_dimension_name(self) -> str:
        """Get the name of the dimension being mapped (e.g., 'cells', 'genes')."""
        pass

    @abstractmethod
    def _get_query_size(self) -> int:
        """Get the size of the query dataset in the relevant dimension."""
        pass

    @abstractmethod
    def _get_reference_size(self) -> int:
        """Get the size of the reference dataset in the relevant dimension."""
        pass

    @abstractmethod
    def _get_target_index(self) -> pd.Index:
        """Get the index for the target dimension."""
        pass

    @abstractmethod
    def _apply_mapping_to_matrix(self, matrix: np.ndarray | csr_matrix) -> np.ndarray | csr_matrix:
        """Apply the mapping matrix to a data matrix."""
        pass

    @abstractmethod
    def _store_mapping_result(
        self,
        target_container: dict | AnnData,
        target_key: str,
        postfix: str,
        data: pd.Series | np.ndarray,
        data_type: str,
    ) -> None:
        """Store the mapping result in the appropriate container."""
        pass

    @abstractmethod
    def _compute_fallback_representation_self(self, n_comps: int | None, fallback_kwargs: dict) -> str:
        """Compute fallback representation for self-mapping."""
        pass

    @abstractmethod
    def _compute_fallback_representation_joint(self, method: str, n_comps: int | None, fallback_kwargs: dict) -> str:
        """Compute joint fallback representation for cross-mapping."""
        pass

    @abstractmethod
    def _extract_representations(self, use_rep: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract the representations for neighbor computation."""
        pass

    @abstractmethod
    def _get_reference_data_container(self) -> pd.DataFrame:
        """Get the container with reference data (for obs: reference.obs, for var: reference.var)."""
        pass

    @property
    def mapping_matrix(self) -> csr_matrix | None:
        """Get the mapping matrix."""
        return self._mapping_matrix

    @mapping_matrix.setter
    def mapping_matrix(self, value: csr_matrix | coo_matrix | csc_matrix | None) -> None:
        """Set and validate the mapping matrix."""
        if value is not None:
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

        Returns
        -------
        Validated and row-normalized mapping matrix.
        """
        expected_shape = (self._get_query_size(), self._get_reference_size())

        if mapping_matrix.shape != expected_shape:
            raise ValueError(
                f"Mapping matrix shape mismatch: expected {expected_shape}, but got {mapping_matrix.shape}."
            )

        # Row normalize the matrix
        row_sums = mapping_matrix.sum(axis=1).A1
        if np.any(row_sums == 0):
            logger.warning("Some rows in the mapping matrix have a sum of zero. These rows will be left unchanged.")
        row_sums[row_sums == 0] = 1
        if not np.allclose(row_sums, 1):
            logger.info("Row-normalizing the mapping matrix.")
        mapping_matrix = mapping_matrix.multiply(1 / row_sums[:, None])

        return csr_matrix(mapping_matrix.astype(np.float32))

    def compute_neighbors(
        self,
        n_neighbors: int = 30,
        use_rep: str | None = None,
        n_comps: int | None = None,
        method: Literal["sklearn", "pynndescent", "rapids", "faiss"] = "sklearn",
        metric: str = "euclidean",
        only_yx: bool = False,
        fallback_representation: Literal["fast_cca", "joint_pca"] = "joint_pca",
        fallback_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Compute nearest neighbors between reference and query.

        Parameters
        ----------
        n_neighbors
            Number of nearest neighbors.
        use_rep
            Data representation based on which to find nearest neighbors.
        n_comps
            Number of components to use.
        method
            Method to use for computing neighbors.
        metric
            Distance metric to use for nearest neighbors.
        only_yx
            If True, only compute the xy neighbors.
        fallback_representation
            Method for computing cross-dataset representation when use_rep=None.
        fallback_kwargs
            Additional keyword arguments for fallback representation method.
        """
        if fallback_kwargs is None:
            fallback_kwargs = {}

        self.only_yx = only_yx

        # Handle fallback representation computation
        if use_rep is None:
            if self._is_self_mapping:
                logger.warning(
                    "No representation provided and self-mapping mode detected. Computing PCA automatically."
                )
                key_added = self._compute_fallback_representation_self(n_comps, fallback_kwargs)
            else:
                logger.warning(
                    "No representation provided. Computing joint representation using '%s'.",
                    fallback_representation,
                )
                key_added = self._compute_fallback_representation_joint(
                    fallback_representation, n_comps, fallback_kwargs
                )
            use_rep = key_added

        # Extract representations
        xrep, yrep = self._extract_representations(use_rep)

        # Handle number of components
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
        Compute the mapping matrix.

        Parameters
        ----------
        method
            Method to use for computing the mapping matrix.
        """
        if self.knn is None:
            raise ValueError("Neighbors have not been computed. Call compute_neighbors() first.")

        logger.info("Computing mapping matrix using method '%s'.", method)

        if method in ["jaccard", "hnoca"]:
            if self.only_yx:
                raise ValueError(
                    "Jaccard and HNOCA methods require both x and y neighbors to be computed. Set only_yx=False."
                )
            xx, yy, xy, yx = self.knn.get_adjacency_matrices()
            n_neighbors = self.knn.xx.n_neighbors  # type: ignore[attr-defined]
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

    def _map_data(
        self,
        data: pd.Series,
        prediction_postfix: str,
        confidence_postfix: str | None = None,
        target_container: dict | AnnData | None = None,
        target_key: str | None = None,
    ) -> None:
        """
        Map data using appropriate method based on data type.

        Parameters
        ----------
        data
            Data to map.
        prediction_postfix
            Postfix for prediction results.
        confidence_postfix
            Postfix for confidence scores (categorical data only).
        target_container
            Container to store results in. Defaults to query.
        target_key
            Key for storing results. Defaults to data series name.
        """
        if self.mapping_matrix is None:
            raise ValueError("Mapping matrix has not been computed. Call compute_mapping_matrix() first.")

        if target_container is None:
            target_container = self.query
        if target_key is None:
            target_key = str(data.name) if hasattr(data, "name") and data.name else "data"

        # Detect data type
        is_categorical = (
            isinstance(data.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(data)
            or pd.api.types.is_string_dtype(data)
        )

        if is_categorical:
            if confidence_postfix is None:
                confidence_postfix = "conf"
            self._map_categorical(data, prediction_postfix, confidence_postfix, target_container, target_key)
        else:
            self._map_numerical(data, prediction_postfix, target_container, target_key)

    def _map_categorical(
        self,
        data: pd.Series,
        prediction_postfix: str,
        confidence_postfix: str,
        target_container: dict | AnnData,
        target_key: str,
    ) -> None:
        """Map categorical data using one-hot encoding."""
        onehot = OneHotEncoder(dtype=np.float32)
        data_values = np.asarray(data.values).reshape(-1, 1)
        xtab = onehot.fit_transform(data_values)

        # Apply mapping
        ytab = self._apply_mapping_to_matrix(xtab)

        # Use the same approach as original CellMapper for sparse matrix handling
        pred_indices = ytab.argmax(axis=1).A1  # type: ignore[attr-defined]
        conf_values = ytab.max(axis=1).toarray().ravel()  # type: ignore[attr-defined]

        pred = pd.Series(
            data=np.array(onehot.categories_[0])[pred_indices],
            index=self._get_target_index(),
            dtype=data.dtype,
        )
        conf = pd.Series(
            conf_values,
            index=self._get_target_index(),
        )

        # Store results
        self._store_mapping_result(target_container, target_key, prediction_postfix, pred, "categorical")
        self._store_mapping_result(target_container, target_key, confidence_postfix, conf, "confidence")

        # Handle color mapping if colors are available and target is AnnData
        if isinstance(target_container, AnnData):
            self._copy_categorical_colors(target_key, prediction_postfix, pred, target_container)

        logger.info("Categorical data mapped and stored with key '%s'.", f"{target_key}_{prediction_postfix}")

    def _map_numerical(
        self, data: pd.Series, prediction_postfix: str, target_container: dict | AnnData, target_key: str
    ) -> None:
        """Map numerical data using direct matrix multiplication."""
        # Reshape and apply mapping
        reference_values = np.array(data).reshape(-1, 1)
        mapped_values = self._apply_mapping_to_matrix(reference_values)

        # Handle both sparse and dense results efficiently
        if hasattr(mapped_values, "toarray"):
            mapped_data = mapped_values.toarray().ravel()  # type: ignore[attr-defined]
        else:
            mapped_data = mapped_values.ravel()  # type: ignore[attr-defined]

        pred = pd.Series(
            data=mapped_data,
            index=self._get_target_index(),
        )

        self._store_mapping_result(target_container, target_key, prediction_postfix, pred, "numerical")

        logger.info("Numerical data mapped and stored with key '%s'.", f"{target_key}_{prediction_postfix}")

    def _map_matrix(self, matrix: np.ndarray | csr_matrix) -> np.ndarray | csr_matrix:
        """Map a matrix (embeddings or expression data)."""
        if self.mapping_matrix is None:
            raise ValueError("Mapping matrix has not been computed. Call compute_mapping_matrix() first.")

        return self._apply_mapping_to_matrix(matrix)

    def _copy_categorical_colors(
        self,
        target_key: str,
        prediction_postfix: str,
        pred: pd.Series,
        target_container: AnnData,
    ) -> None:
        """Copy categorical colors from reference to target container."""
        if f"{target_key}_colors" in self.reference.uns:
            reference_data_container = self._get_reference_data_container()
            color_lookup = dict(
                zip(
                    reference_data_container[target_key].cat.categories,
                    self.reference.uns[f"{target_key}_colors"],
                    strict=True,
                )
            )
            target_container.uns[f"{target_key}_{prediction_postfix}_colors"] = [
                color_lookup.get(cat, "#383838") for cat in pred.cat.categories
            ]
