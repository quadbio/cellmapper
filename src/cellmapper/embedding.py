import anndata as ad
import numpy as np
import scanpy as sc
from scanpy.get import _check_mask, _get_obs_rep
from scipy.sparse import csr_matrix, issparse

from cellmapper.logging import logger
from cellmapper.utils import truncated_svd_cross_covariance


class EmbeddingMixin:
    """Mixing class to compute joint embeddings for two datasets."""

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
        mask_var: np.ndarray | str | None = None,
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
        mask_var
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

        # Concatenate with inner join on genes
        joint = ad.concat([self.reference, self.query], join="inner", label="batch", keys=["reference", "query"])

        # Potentially subset along the features
        mask_var = _check_mask(joint, mask_var, "var")
        X_query = _get_obs_rep(joint[joint.obs["batch"] == "query", mask_var], layer=layer)
        X_ref = _get_obs_rep(joint[joint.obs["batch"] == "reference", mask_var], layer=layer)

        n_common_genes = X_query.shape[1]
        logger.info(
            "Using %d common genes between query and reference datasets.",
            n_common_genes,
        )

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
            "n_common_genes": n_common_genes,
            "zero_center": zero_center,
            "scale_with_singular": scale_with_singular,
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
