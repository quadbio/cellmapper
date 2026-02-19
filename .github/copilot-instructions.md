# Copilot Instructions for CellMapper

## Project Overview

**CellMapper** is a k-NN-based tool for mapping cells across representations to
transfer labels, embeddings, and expression values. It works for millions of
cells, on CPU and GPU, across molecular modalities, between spatial and
non-spatial data. The core idea is to separate the method (k-NN graph with
kernels) from the application (mapping across arbitrary representations).

### Domain Context
- **k-NN mapping**: Compute k-nearest neighbors between query and reference
  datasets, apply graph kernel to create mapping matrix, use it to transfer
  labels/embeddings/expression.
- **Joint embeddings**: CellMapper expects pre-computed joint embeddings in
  `.obsm` from tools like scVI, scANVI, GimVI, ENVI, GLUE, or implements
  baseline methods (PCA, CCA).

### Key Dependencies
- **Core**: anndata, scanpy, numpy, pandas, scipy, scikit-learn
- **k-NN backends**: pynndescent, sklearn, faiss (CPU/GPU), rapids (GPU)
- **Optional**: squidpy (for spatial), scvi-tools, harmony-pytorch (for tutorials)

## Architecture

### Core Components
1. **`src/cellmapper/model/cellmapper.py`**: Main `CellMapper` class with `map()` method
   - Inherits from `EvaluationMixin` and `EmbeddingMixin`
   - Handles both query-to-reference and self-mapping modes
   - Core methods: `map()`, `map_obs()`, `map_obsm()`, `map_layers()`
2. **`src/cellmapper/model/neighbors.py`**: k-NN graph computation with multiple backends
3. **`src/cellmapper/model/kernel.py`**: Graph kernels for creating mapping matrices
4. **`src/cellmapper/model/mapping_operator.py`**: Mapping matrix with matrix powers for diffusion
5. **`src/cellmapper/model/evaluate.py`**: Metrics for evaluating transfer quality
6. **`src/cellmapper/model/embedding.py`**: Baseline joint embedding methods (PCA, CCA)
7. **`src/cellmapper/utils.py`**: Utilities (library size adjustment, imputed data creation)

## Project-Specific Patterns

### Basic Usage
```python
from cellmapper import CellMapper

cmap = CellMapper(query, reference).map(
    use_rep="X_joint",
    obs_keys="celltype",
    obsm_keys="X_umap",
    layer_key="counts",
)

# Self-mapping (for spatial contextualization, denoising)
cmap_self = CellMapper(query).map(use_rep="X_pca", layer_key="counts")
```

### k-NN Backends
- **pynndescent**: Fast approximate k-NN, CPU-only
- **sklearn**: Exact k-NN, CPU-only, slower for large datasets
- **faiss**: Exact/approximate k-NN, supports CPU and GPU (via faiss-gpu)
- **rapids**: GPU-accelerated k-NN using cuML

### Mapping Workflow
1. Compute k-NN graph between query and reference (or self)
2. Apply kernel to k-NN graph to create mapping matrix M
3. Transfer data: `query_data = M @ reference_data`
4. Optionally apply matrix powers `M^t` for diffusion
5. Evaluate transfer quality with metrics

## Common Gotchas

1. **Joint embeddings required**: Most use cases require pre-computed joint embedding in `.obsm`. Don't assume PCA is sufficient for complex mappings.
2. **Sparse matrices**: Check `scipy.sparse.issparse(adata.X)` before operations. Mapping matrices are typically dense.
3. **Self-mapping mode**: If `reference` is `None` or same as `query`, automatically enters self-mapping mode.
4. **k-NN backends**: faiss requires `faiss-cpu` or `faiss-gpu`, rapids requires CUDA environment. Handle gracefully with fallbacks.

## Related Resources

- **squidpy docs**: https://squidpy.readthedocs.io/ (for spatial analysis)
- **faiss docs**: https://github.com/facebookresearch/faiss
