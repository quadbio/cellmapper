"""Shared documentation for cellmapper."""

from docrep import DocstringProcessor

__all__ = ["d"]

_t = """\
t
    Number of diffusion time steps. This parameter controls the degree of
    smoothing applied by the diffusion operator. Larger values lead to more
    smoothing."""

_diffusion_method = """\
diffusion_method
    Method for computing powers of the mapping matrix (only valid in self-mapping mode). Options are "iterative" for
    iterative matrix multiplication (inspired by MAGIC :cite:`van2018recovering`) and "spectral" for
    eigendecomposition-based approach. """

_prediction_postfix = """\
prediction_postfix
    Postfix to add to mapped variables to identify them as predictions."""

_symmetrize = """\
symmetrize
    If True, create a symmetrize connectivity matrix. Only valid for square matrices (self-mapping).
    If None (default), uses True for self-mapping and False for cross-mapping."""

_self_edges = """\
self_edges
    Control self-edges (diagonal entries) for square matrices (self-mapping).
    If None (default), uses False for self-mapping (scanpy style) and None for cross-mapping.
    This controls whether or not the kernel used to compute the connectivities is supplied with self-edges.
    It does not determine whether the final connectivity matrix has self edges. For example, the `umap`
    kernel expectes self-edges, but does not produce them in the final connectivity matrix."""

_knn_method = """\
knn_method
    Method for computing k-nearest neighbors. Options include:
    - "sklearn": Scikit-learn's NearestNeighbors. See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    - "pynndescent": Pynndescent's approximate nearest neighbors. See https://pynndescent.readthedocs.io/en/latest/
    - "rapids": RAPIDS cuML's NearestNeighbors (GPU). See https://docs.rapids.ai/api/cuml/stable/api.html#cuml.neighbors.NearestNeighbors
    - "faiss-cpu": Facebook AI Similarity Search (FAISS) on CPU. See https://faiss.ai/
    - "faiss-gpu": Facebook AI Similarity Search (FAISS) on GPU. See https://faiss.ai/


    All methods return exactly `n_neighbors` neighbors, including the reference cell itself (in self-mapping mode)."""

_only_yx = """\
only_yx
    If True, only compute the xy neighbors. In self-mapping mode, this is
    automatically set to True for efficiency since all neighbor matrices contain the same information.
    This is faster, but not suitable for Jaccard or HNOCA methods in cross-mapping mode."""

_kernel_method = """\
kernel_method
    Method to use for computing the mapping matrix. Options include:

    - "jaccard": Jaccard similarity. Inspired by GLUE :cite:`cao2022multi`
    - "gauss": Gaussian kernel with (global) bandwith equal to the mean distance.
    - "scarches": scArches kernel. Inspired by scArches :cite:`lotfollahi2022mapping`
    - "inverse_distance": Inverse distance kernel.
    - "random": Random kernel, useful for testing.
    - "hnoca": HNOCA kernel. Inspired by HNOCA-tools :cite:`he2024integrated`
    - "equal": All neighbors are equally weighted (1/n_neighbors).
    - "umap": UMAP fuzzy simplicial set connectivities. Only available for self-mapping with true k-NN graphs."""

_comparison_method = """\
comparison_method
    Method to use for comparing the mapping results. Options include:

    - "pearson": Pearson correlation coefficient.
    - "spearman": Spearman rank correlation coefficient.
    - "js": Jenson-Shanon divergence.
    - "rmse": Root Mean Square Error."""

_layer_key = """\
layer_key
    Key in `self.query.layers` to use as the original expression. Use "X" to use `self.query.X`."""


_n_neighbors = """\
n_neighbors
    Number of nearest neighbors. This parameter controls the sparsity of the connectivity matrix. """

_use_rep = """\
use_rep
    Data representation based on which to find nearest neighbors. If None, a fallback representation will be
    computed automatically. """

_knn_dist_metric = """\
knn_dist_metric
    Distance metric to use for nearest neighbors. See the knn algorithms documentation for details. """

_subset_categories = """\
subset_categories
    For categorical data, optionally specify a subset of categories to include in the mapping.
    If None (default), all categories are included. If specified, only the listed categories
    will be mapped, and others will be ignored. For numerical data, this parameter is ignored
    with a warning. Can be a single category string or a list of category strings."""


d = DocstringProcessor(
    t=_t,
    diffusion_method=_diffusion_method,
    prediction_postfix=_prediction_postfix,
    symmetrize=_symmetrize,
    self_edges=_self_edges,
    knn_method=_knn_method,
    only_yx=_only_yx,
    kernel_method=_kernel_method,
    comparison_method=_comparison_method,
    layer_key=_layer_key,
    n_neighbors=_n_neighbors,
    use_rep=_use_rep,
    knn_dist_metric=_knn_dist_metric,
    subset_categories=_subset_categories,
)
