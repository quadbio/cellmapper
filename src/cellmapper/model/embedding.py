import anndata as ad
import numpy as np
import scanpy as sc
from scanpy.get import _check_mask, _get_obs_rep
from scipy.sparse import csr_matrix, issparse

from cellmapper.logging import logger
from cellmapper.utils import get_n_comps, truncated_svd_cross_covariance


class EmbeddingMixin:
    """Mixing class to compute joint embeddings for two datasets."""

    def compute_joint_pca(self, n_comps: int | None = None, key_added: str = "X_pca", **kwargs) -> None:
        """
        Compute a joint PCA on the normalized .X matrices of query and reference, using only overlapping genes.

        Parameters
        ----------
        n_comps
            Number of principal components to compute.
        key_added
            Key under which to store the joint PCA embeddings in `.obsm` of both query and reference AnnData objects.
        **kwargs
            Additional keyword arguments to pass to scanpy's `pp.pca` function.

        Notes
        -----
        This method performs an inner join on genes (variables) between the query and reference AnnData objects,
        concatenates the normalized expression matrices, and computes a joint PCA using Scanpy. The resulting
        PCA embeddings are stored in `.obsm[key_added]` for both objects. Consider using `compute_fast_cca` for
        improved cross-dataset integration.
        """
        # Concatenate with inner join on genes
        joint = ad.concat(
            [self.reference, self.query], join="inner", label="batch", keys=["reference", "query"], merge="same"
        )

        # Compute PCA using scanpy
        sc.pp.pca(joint, n_comps=n_comps, **kwargs)

        # assign back to AnnData objects
        self._set_embedding(
            X_query=joint.obsm["X_pca"][joint.obs["batch"] == "query"],
            X_ref=joint.obsm["X_pca"][joint.obs["batch"] == "reference"],
            key_added=key_added,
            method="joint_pca",
            n_comps=n_comps,
            n_common_genes=joint.n_vars,
        )

    def compute_fast_cca(
        self,
        n_comps: int | None = None,
        key_added: str = "X_cca",
        layer: str | None = None,
        mask_var: np.ndarray | str | None = None,
        zero_center: bool = True,
        scale_with_singular: bool = False,
        l2_scale: bool = True,
        random_state: int = 0,
        implicit: bool = True,
    ) -> None:
        """
        Compute a joint embedding using fast CCA on the reference and query datasets.

        This method computes the singular value decomposition (SVD) of the cross-covariance matrix
        between the reference and query datasets using an efficient implementation that doesn't
        materialize the full cross-covariance matrix. It then constructs a joint embedding
        based on the SVD components.

        Parameters
        ----------
        n_comps
            Number of components to keep in the embedding.
        key_added
            Key under which to store the joint embedding in `.obsm` of both
            query and reference AnnData objects.
        layer
            Layer to use for the computation. If None, use .X.
        mask_var
            Boolean mask or string mask identifying a subset of variables/genes to use
            for computation. If string, should be a key in .var for both query and reference
            that contains a boolean mask. If None, uses the intersection of all variables
            from both datasets.
        zero_center
            If True, center the data (implicitly for sparse matrices).
        scale_with_singular
            If True, scale the singular vectors by the square root of their
            singular values. If False, return the raw singular vectors.
        l2_scale
            If True, scale the matrices of singular vectors to have l2 norm of 1 per observation.
        random_state
            Random seed for reproducibility.
        implicit
            Whether to use implicit mean centering and covariance computation.

        Returns
        -------
        None
            Embeddings are stored in the .obsm attribute of both query and reference AnnData objects.

        Notes
        -----
        This is a fast implementation of Canonical Correlation Analysis (CCA) that computes
        the SVD of the cross-covariance matrix between the query and reference datasets.
        It is designed to be memory-efficient and fast, especially for large datasets.
        The method uses an efficient SVD implementation that avoids explicitly constructing the
        cross-covariance matrix, which can be very large for high-dimensional data.
        The method is particularly useful for integrating datasets with potentially different
        gene expression profiles, as it focuses on the shared variance between the two datasets.


        This method is a fast re-implementation of Seurat's CCA approach.
        """
        logger.info(
            "Computing fast CCA between query (%d cells) and reference (%d cells).",
            self.query.n_obs,
            self.reference.n_obs,
        )

        # Concatenate with inner join on genes
        joint = ad.concat(
            [self.reference, self.query], join="inner", label="batch", keys=["reference", "query"], merge="same"
        )

        # Potentially subset along the features
        mask_var = _check_mask(joint, mask_var, "var")
        joint = joint[:, mask_var] if mask_var is not None else joint

        X_query = _get_obs_rep(joint[joint.obs["batch"] == "query"], layer=layer)
        X_ref = _get_obs_rep(joint[joint.obs["batch"] == "reference"], layer=layer)

        n_common_genes = X_query.shape[1]
        logger.info(
            "Using %d common genes between query and reference datasets.",
            n_common_genes,
        )

        # determine number of components to keep
        n_comps = get_n_comps(n_comps, n_vars=n_common_genes)

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
            X_query,
            X_ref,
            n_comps=n_comps,
            zero_center=zero_center,
            random_state=random_state,
            implicit=implicit,
        )
        V = Vt.T

        logger.info("SVD of cross-covariance matrix computed successfully.")

        # Construct embeddings following dual PCA formulation
        # X = U * sqrt(S)
        # Y = V * sqrt(S) = (Vt.T) * sqrt(S)
        if scale_with_singular:
            s_sqrt = np.sqrt(s)
            U = U * s_sqrt[np.newaxis, :]
            V = V * s_sqrt[np.newaxis, :]

        if l2_scale:
            U /= np.linalg.norm(U, axis=1)[:, None]
            V /= np.linalg.norm(V, axis=1)[:, None]

        # assign back to AnnData objects
        self._set_embedding(
            X_query=U,
            X_ref=V,
            key_added=key_added,
            method="fast_cca",
            n_comps=n_comps,
            n_common_genes=n_common_genes,
        )

    def _set_embedding(
        self,
        X_query: np.ndarray,
        X_ref: np.ndarray,
        key_added: str,
        method: str,
        n_comps: int,
        n_common_genes: int,
    ) -> None:
        """
        Set the joint embedding in the query and reference AnnData objects.

        Parameters
        ----------
        X_query
            Joint embedding for the query dataset.
        X_ref
            Joint embedding for the reference dataset.
        key_added
            Key under which to store the joint embedding in `.obsm` of both
            query and reference AnnData objects.
        """
        self.query.obsm[key_added] = X_query
        self.reference.obsm[key_added] = X_ref

        params = {
            "n_comps": n_comps,
            "n_common_genes": n_common_genes,
            "method": method,
        }

        # Reference dataset gets the same parameters
        self.reference.uns[f"{key_added}_params"] = params
        self.query.uns[f"{key_added}_params"] = params

        logger.info(
            "Embedding computed with method '%s' stored as '%s' in both reference.obsm and query.obsm. ",
            method,
            key_added,
        )
