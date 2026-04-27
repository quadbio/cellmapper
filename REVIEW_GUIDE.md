# CellMapper Review Guide

Agent-neutral PR review playbook. Written for **review agents running on GitHub** — use the imperative voice.

**Scope: review only.** Produce comments and suggestions. Do **not** push commits, modify files, or apply fixes. Flag issues and suggest diffs in comments; leave the edits to the author.

Architecture, invariants, and commands live in `AGENTS.md`. Do not restate them here — link.

## Workflow

1. Read the PR body.
2. Check CI (`gh pr checks <num>`, `gh run view <run-id> --log-failed`) and investigate failures before commenting.
3. Map changed paths to tests (see below) and check whether the change touches an invariant from `AGENTS.md`.
4. Prioritize behavioral regressions, numerical correctness, and public-contract changes over style.

## High-Risk Areas

Pointers only — see `AGENTS.md` for the actual invariants.

- **Mapping-matrix construction** (`model/mapping_operator.py`): normalization, dtype, sparsity handling. Silent shifts possible without test failures.
- **Matrix powers / diffusion** (`model/mapping_operator.py`, `_validate_power`, `_apply_iterative`, `_apply_spectral`): self-mapping gate, iterative-vs-spectral behavior.
- **Self-mapping detection** (`model/cellmapper.py::CellMapper.__init__`): identity check drives all downstream mode-dependent logic.
- **Kernel taxonomy** (`constants.py`, `model/kernel.py`, `model/neighbors.py`): new kernels must land in the right set in `constants.py`.
- **k-NN backend gating** (`model/_knn_backend.py`, `check.py`): optional deps must route through `check.check_deps()`.
- **AnnData output contract** (`model/cellmapper.py::map_obs / map_obsm / map_layers`): key naming, what gets written where, `query_imputed` construction via `utils.create_imputed_anndata`.
- **Public API surface** (`src/cellmapper/__init__.py`): new re-exports commit the project to an API.

## Changed-Path Test Lookup

Tests mirror the source tree.

| Changed path | Primary tests |
|--------------|---------------|
| `src/cellmapper/model/cellmapper.py` | `tests/model/test_query_to_reference_mapping.py`, `tests/model/test_self_mapping.py` |
| `src/cellmapper/model/kernel.py` | `tests/model/test_kernel.py` |
| `src/cellmapper/model/mapping_operator.py` | `tests/model/test_mapping_operator.py` |
| `src/cellmapper/model/neighbors.py` | `tests/model/test_neighbors.py` |
| `src/cellmapper/model/embedding.py` | `tests/model/test_embedding.py` |
| `src/cellmapper/model/evaluate.py` | `tests/model/test_evaluate.py` |
| `src/cellmapper/model/_knn_backend.py` | `tests/model/test_neighbors.py`, `tests/model/test_kernel.py` |
| `src/cellmapper/check.py` | `tests/test_check.py` |
| `src/cellmapper/utils.py` | `tests/test_utils.py` |
| End-to-end behavioral change | also `tests/test_basic.py` |
| Fixture changes | `tests/conftest.py`, `tests/data/` |

## Testing

- **New code** should be covered. Reuse fixtures from `tests/conftest.py`; prefer `pytest.mark.parametrize`; favor few meaningful tests over many redundant ones.
- **Failing CI** is not to be waved through. Distinguish critical regressions from flakes; escalate critical ones.
- **Modified tests** — scrutinize *how*. Relaxed tolerances, removed assertions, deleted cases, or loosened matrices are red flags. Require explicit justification in the PR body.

## Documentation Impact

Behavior or API changes often touch docs in multiple places. Point at the **owning file** (see the `AGENTS.md` "Where To Find What" table) — don't duplicate content in the review.

- Public symbol / API changes → `docs/api.md`, autosummary, `README.md` quickstart, source docstrings.
- Contributor workflow or env changes → `docs/contributing.md`.
- Tutorials under `docs/notebooks/tutorials/` → flag stale imports, outputs, or API usage.
- Invariants / commands → `AGENTS.md`.
- Review workflow / risk areas / test lookup → this file.
- `CLAUDE.md` and `.github/copilot-instructions.md` should stay thin pointers — flag any PR that re-adds content here.

## Checklist

- Invariants in `AGENTS.md` preserved?
- CI green (or failures investigated)?
- Test coverage adequate and not silently weakened?
- Public contracts (AnnData output, mapping matrix format, public API surface) unchanged — or explicitly called out in the PR body?
- Affected human- and agent-facing docs updated?
- PR scope tight, no unrelated bundling?

## PR Metadata

This repo uses `.github/PULL_REQUEST_TEMPLATE.md`. Treat its sections as the preferred summary surface.
