import anndata as ad
import numpy as np
import scanpy as sc
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
