# Copilot Instructions for CellMapper

## Important Notes
- Avoid drafting summary documents or endless markdown files. Just summarize in chat what you did, why, and any open questions.
- Don't update Jupyter notebooks - those are managed manually.
- When running terminal commands, use `uv run` to execute commands within the project's virtual environment (e.g., `uv run python script.py`).
- **Testing: ALWAYS use `hatch test`, NEVER `uv run pytest` or standalone pytest.** Hatch manages the test matrix (Python versions, dependencies) that CI uses. See "Testing Strategy" section for details.
- Rather than making assumptions, ask for clarification when uncertain.
- **GitHub workflows**: Use GitHub CLI (`gh`) when possible. For GitHub MCP server tools, ensure Docker Desktop is running first (`open -a "Docker Desktop"`).

## Project Overview

**CellMapper** is a k-NN-based tool for mapping cells across representations to transfer labels, embeddings, and expression values. It works for millions of cells, on CPU and GPU, across molecular modalities, between spatial and non-spatial data. The core idea is to separate the method (k-NN graph with kernels) from the application (mapping across arbitrary representations).

### Domain Context (Brief)
- **AnnData**: Standard single-cell data structure. Contains `.X`, `.obs`, `.var`, `.obsm` (embeddings), `.layers`.
- **k-NN mapping**: Compute k-nearest neighbors between query and reference datasets, apply graph kernel to create mapping matrix, use it to transfer labels/embeddings/expression.
- **Joint embeddings**: CellMapper expects pre-computed joint embeddings in `.obsm` from tools like scVI, scANVI, GimVI, ENVI, GLUE, or implements baseline methods (PCA, CCA).
- **Use cases**: Transfer labels from dissociated to spatial data, map embeddings between datasets, compute presence scores in atlases, identify spatial niches, evaluate mapping quality.

### Key Dependencies
- **Core**: anndata, scanpy, numpy, pandas, scipy, scikit-learn
- **k-NN backends**: pynndescent, sklearn, faiss (CPU/GPU), rapids (GPU)
- **Optional**: squidpy (for spatial), scvi-tools, harmony-pytorch (for tutorials)

## Architecture & Code Organization

### Module Structure (follows scverse conventions)
- Use `AnnData` objects as primary data structure
- Type annotations use modern syntax: `str | None` instead of `Optional[str]`
- Supports Python 3.11, 3.12, 3.13 (see `pyproject.toml`)
- Avoid local imports unless necessary for circular import resolution

### Core Components
1. **`src/cellmapper/model/cellmapper.py`**: Main `CellMapper` class with `map()` method
   - Inherits from `EvaluationMixin` and `EmbeddingMixin`
   - Handles both query-to-reference and self-mapping modes
   - Core methods: `map()`, `map_obs()`, `map_obsm()`, `map_layers()`
2. **`src/cellmapper/model/neighbors.py`**: k-NN graph computation with multiple backends
3. **`src/cellmapper/model/kernel.py`**: Graph kernels for creating mapping matrices
4. **`src/cellmapper/model/mapping_operator.py`**: Encapsulates mapping matrix with matrix powers for diffusion
5. **`src/cellmapper/model/evaluate.py`**: Metrics for evaluating label/expression transfer quality
6. **`src/cellmapper/model/embedding.py`**: Baseline joint embedding methods (PCA, CCA)
7. **`src/cellmapper/utils.py`**: Utilities (library size adjustment, imputed data creation)

## Development Workflow

### Environment Management (uv-based)
```bash
# Create/sync virtual environment
uv sync                        # install project with default dependencies
uv sync --extra test           # include test dependencies
uv sync --extra doc            # include documentation dependencies
uv sync --all-extras           # include all optional dependencies

# Run commands in virtual environment
uv run python script.py        # run any Python script
uv run pytest tests/           # run tests directly (alternative to hatch)

# Testing via hatch (recommended, runs test matrix, uses uv internally)
hatch test                     # test with highest Python version
hatch test --all               # test all Python 3.11, 3.13, pre-release deps

# Documentation
hatch run docs:build           # build Sphinx docs
hatch run docs:open            # open in browser
hatch run docs:clean           # clean build artifacts

# Environment inspection
hatch env show                 # list environments
```

### Testing Strategy
- Test matrix defined in `[[tool.hatch.envs.hatch-test.matrix]]` in `pyproject.toml`
- Tests Python 3.11 & 3.13 with stable deps, 3.13 with pre-release deps
- CI extracts test config from pyproject.toml (`.github/workflows/test.yaml`)
- Tests live in `tests/`, fixtures in `tests/conftest.py`
- **Always run tests via `hatch test`**, NOT standalone pytest

### Code Quality Tools
- **Ruff**: Linting and formatting (120 char line length)
- **Biome**: JSON/JSONC formatting with trailing commas
- **Pre-commit**: Auto-runs ruff, biome. Install with `pre-commit install`
- Use `git pull --rebase` if pre-commit.ci commits to your branch

## Documentation Conventions

### Docstring Style (NumPy format via Napoleon)
```python
def map_obs(
    self,
    obs_keys: str | list[str],
    *,  # keyword-only marker
    prediction_postfix: str = "_predicted",
    confidence_postfix: str = "_confidence",
) -> pd.DataFrame:
    """Short one-line description.

    Extended description if needed.

    Parameters
    ----------
    obs_keys
        Keys in reference.obs to transfer to query.
    prediction_postfix
        Suffix for predicted column names.
    confidence_postfix
        Suffix for confidence score column names.

    Returns
    -------
    DataFrame with transferred labels and confidence scores.
    """
```

### Sphinx & Documentation
- API docs auto-generated from `docs/api.md` using `autosummary`
- Tutorials in `docs/notebooks/tutorials/` rendered via myst-nb (`.ipynb` only)
- Add external packages to `intersphinx_mapping` in `docs/conf.py`
- See `docs/contributing.md` for detailed documentation guidelines

## Key Configuration Files

### `pyproject.toml`
- **Build**: `hatchling` with `hatch-vcs` for git-based versioning
- **Dependencies**: Minimal runtime deps; optional extras for `[test]`, `[doc]`, `[tutorials]`
- **Ruff**: 120 char line length, NumPy docstring convention
- **Test matrix**: Python 3.11 & 3.13 (stable), 3.13 (pre-release)

### Version Management
- Version from git tags via `hatch-vcs`
- Release: Create GitHub release with tag `vX.X.X`
- Follows **Semantic Versioning**

## Project-Specific Patterns

### Basic Usage Pattern
```python
from cellmapper import CellMapper

# Assume query and reference have joint embedding in .obsm["X_joint"]
cmap = CellMapper(query, reference).map(
    use_rep="X_joint",
    obs_keys="celltype",           # transfer labels
    obsm_keys="X_umap",             # transfer UMAP
    layer_key="counts",             # transfer expression
)

# Self-mapping (for spatial contextualization, denoising)
cmap_self = CellMapper(query).map(
    use_rep="X_pca",
    layer_key="counts",
)
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

### AnnData Conventions
- Check matrix format: `adata.X` may be sparse or dense
- Use `adata.layers[key]` for alternative representations (e.g., counts, log-normalized)
- Joint embeddings stored in `adata.obsm["X_<method>"]`
- Transferred data goes back into query's `.obs`, `.obsm`, `.layers`

### Testing with AnnData
```python
# From conftest.py - example fixture pattern
@pytest.fixture
def adata_spatial():
    """Small spatial AnnData object with spatial coordinates."""
    adata = ad.AnnData(
        X=np.random.randn(100, 50).astype(np.float32),
        obs=pd.DataFrame({"celltype": ["A", "B"] * 50}),
        obsm={"spatial": np.random.rand(100, 2)},
    )
    sc.pp.pca(adata)
    return adata
```

## Common Gotchas

1. **Hatch for testing**: Always use `hatch test`, never standalone `pytest`. CI matches hatch test matrix.
2. **Joint embeddings required**: Most use cases require pre-computed joint embedding in `.obsm`. Don't assume PCA is sufficient for complex mappings.
3. **Sparse matrices**: Check `scipy.sparse.issparse(adata.X)` before operations. Mapping matrices are typically dense.
4. **Self-mapping mode**: If `reference` is `None` or same as `query`, automatically enters self-mapping mode.
5. **k-NN backends**: faiss requires `faiss-cpu` or `faiss-gpu`, rapids requires CUDA environment. Handle gracefully with fallbacks.
6. **Pre-commit conflicts**: Use `git pull --rebase` to integrate pre-commit.ci fixes.
7. **Line length**: Ruff set to 120 chars, but keep docstrings readable (~80 chars per line).

## Related Resources

- **Contributing guide**: `docs/contributing.md`
- **Tutorials**: `docs/notebooks/tutorials/`
- **scanpy docs**: https://scanpy.readthedocs.io/
- **faiss docs**: https://github.com/facebookresearch/faiss
- **squidpy docs**: https://squidpy.readthedocs.io/ (for spatial analysis)
