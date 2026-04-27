# AGENTS.md — CellMapper

CellMapper is a Python package for k-NN-based mapping of cells across
representations to transfer labels, embeddings, and expression values. It works
for millions of cells, on CPU and GPU, across molecular modalities, and between
spatial and non-spatial data. The core idea is to separate the method (k-NN
graph + kernel → mapping matrix) from the application (transfer across arbitrary
representations).
Key frameworks: AnnData/Scanpy, numpy/scipy, scikit-learn, pynndescent, faiss,
RAPIDS cuML.

## Trust Order

When sources disagree:
1. PR description and changed code
2. This file (`AGENTS.md`)
3. `REVIEW_GUIDE.md`
4. Tests and fixtures
5. Public docs in `docs/` and `README.md`

Every fact should have one owner. This file owns invariants and the reference
table below — everything else is delegated.

## Where To Find What

| Topic | Source of truth |
|-------|----------------|
| User-facing overview, use cases, quickstart | `README.md` |
| Public API reference | `docs/api.md` and autosummary under `docs/generated/` |
| Tutorials (query→reference, spatial mapping, spatial smoothing, data denoising) | `docs/notebooks/tutorials/` |
| Contributor setup, environments, docs build | `docs/contributing.md` |
| Release notes | `docs/changelog.md` |
| PR review workflow and risk areas | `REVIEW_GUIDE.md` |
| Test fixtures | `tests/conftest.py`, `tests/data/` |
| Kernel taxonomy and tunable thresholds (sklearn warning cutoff, spectral threshold) | `src/cellmapper/constants.py` |
| Optional-dependency gating | `src/cellmapper/check.py` |
| Method-level behavior (parameters, return shapes) | docstrings in `src/cellmapper/model/` |

## Critical Invariants

- **Self-mapping mode** activates when `reference is None` **or** `reference is query` (object identity). See `CellMapper.__init__` in `src/cellmapper/model/cellmapper.py`.
- **Reference is read-only.** `.map()` never mutates the reference AnnData. `query` is mutated in place for `map_obs` / `map_obsm`. Expression transfer produces a separate `query_imputed` AnnData object, not a view.
- **Output key naming** follows `{key}{prediction_postfix}` and `{key}{confidence_postfix}` in `query.obs` / `query.obsm`. Postfixes are user-controllable on the per-method entrypoints (`map_obs`, `map_obsm`); `.map()` also exposes `prediction_postfix`.
- **`.map()` auto-chains** `compute_neighbors` → `compute_mapping_matrix` → `map_obs/obsm/layers` based on missing state. Callers that use these methods directly must respect that ordering.
- **Mapping matrix is row-stochastic and float32.** Sparse inputs are stored as `scipy.sparse.csr_matrix`; dense inputs stay dense. Zero-neighbor rows are left as-is. See `MappingOperator._validate_and_normalize_mapping_matrix`.
- **Matrix powers `t > 1` are self-mapping-only** (`MappingOperator._validate_power` raises otherwise).
- **Optional k-NN backends fail fast.** `check.check_deps()` is called at backend construction with clear install hints — no silent fallback. Supported backends: `sklearn`, `pynndescent`, `faiss-cpu`, `faiss-gpu`, `rapids`.
- **Kernel taxonomy lives in `constants.py`** (`JACCARD_BASED_KERNELS`, `CONNECTIVITY_BASED_KERNELS`, `SELF_MAPPING_ONLY_KERNELS`). Kernels in `SELF_MAPPING_ONLY_KERNELS` require a square neighbor matrix.
- **`Neighbors` strips self-edges** from storage for square matrices; `n_neighbors` counts non-self neighbors.
- **`query_imputed` is always assembled via `utils.create_imputed_anndata`**. Setter accepts `AnnData | ndarray | csr_matrix | DataFrame | None`; result has `obs`/`obsm` from query and `var`/`varm` from reference.
- **Public API surface = `__init__.py` `__all__`**: `CellMapper`, `Kernel`, `Neighbors`, `logger`. `EvaluationMixin` and `EmbeddingMixin` are internal. Do not re-export helpers from the top-level package.
- **Tests mirror source layout.** `src/cellmapper/X.py` → `tests/test_X.py`; `src/cellmapper/model/X.py` → `tests/model/test_X.py`.

## Development Commands

Python 3.11 and 3.14 (see the `hatch-test` matrix in `pyproject.toml`).

```bash
hatch test                        # run tests (highest Python)
hatch test --all                  # full matrix
hatch run docs:build              # build Sphinx docs
pre-commit run --all-files        # lint and format
```

Focused runs (with `uv`):

```bash
uv run pytest tests/model
uv run pytest tests/model/test_mapping_operator.py
uv run pytest tests/test_utils.py
```
