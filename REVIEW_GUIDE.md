# CellMapper Review Guide

This file is the canonical, agent-neutral source of truth for automated PR
review in this repo. It is written for **agents performing PR reviews on
GitHub** — use the imperative voice and be concrete.

**Scope: review only.** Your job is to produce review comments and suggestions
on the PR. Do **not** push commits, modify files, or apply fixes yourself. Any
changes are the author's call. Flag issues, ask questions, and suggest concrete
diffs in comments when helpful — but leave the decision and the edits to the
user.

Use `AGENTS.md` for architecture, invariants, and commands. Use this guide for
review workflow, risk areas, testing checks, documentation-impact checks, and
test lookup.

## Review-First Workflow

1. Read the PR body first when it is present.
2. Check CI status (`gh pr checks <num>`, `gh run view <run-id> --log-failed`) and investigate any test or lint failures before commenting.
3. Identify changed modules and map them to matching tests (see [Changed-Path Test Lookup](#changed-path-test-lookup)).
4. Check whether the change touches a repo invariant from `AGENTS.md`.
5. Prioritize behavioral regressions (mapping-matrix semantics, key-naming contract, optional-dep gating) over style feedback.
6. Verify that docs (human- and agent-facing) did not become stale — see [Documentation Impact](#documentation-impact).

## High-Risk Areas

- **Mapping matrix semantics:**
  the matrix is a row-stochastic CSR float32. Changes to normalization, dtype,
  or sparsity handling silently affect every transfer.
- **Matrix powers and diffusion:**
  `t > 1` is gated to self-mapping mode via `MappingOperator._validate_power`.
  Iterative preserves sparsity; spectral always returns dense. Changes here can
  break the denoising / spatial-smoothing workflows.
- **Self-mapping detection:**
  activates on `reference is None` or `reference is query`. Changes to the
  identity check or to the warning path can regress self-mapping workflows.
- **Kernel taxonomy:**
  new kernels must be registered in `src/cellmapper/constants.py`
  (`JACCARD_BASED_KERNELS`, `CONNECTIVITY_BASED_KERNELS`,
  `SELF_MAPPING_ONLY_KERNELS`). `umap` requires square neighbor matrices.
- **k-NN backend gating:**
  all optional backends (`faiss-cpu`, `faiss-gpu`, `rapids`, `pynndescent`)
  must route through `check.check_deps()` and fail with a clear install hint.
  No silent fallback.
- **Output contract on AnnData:**
  results land on `query` only (never `reference`), using
  `{key}{prediction_postfix}` / `{key}{confidence_postfix}`. Renaming these or
  changing postfix defaults breaks downstream user code and tutorials.
- **`query_imputed` construction:**
  all paths must go through `utils.create_imputed_anndata`. Bypassing it
  breaks the `obs`/`obsm` from query + `var`/`varm` from reference contract.
- **Public API surface:**
  only symbols in `src/cellmapper/__init__.py` `__all__` are public. Flag new
  re-exports and prefer keeping internals (e.g. `EvaluationMixin`,
  `EmbeddingMixin`) private.

## Changed-Path Test Lookup

Tests mirror the source tree.

| Changed path | Matching tests |
|--------------|----------------|
| `src/cellmapper/model/cellmapper.py` | `tests/model/test_query_to_reference_mapping.py`, `tests/model/test_self_mapping.py` |
| `src/cellmapper/model/kernel.py` | `tests/model/test_kernel.py` |
| `src/cellmapper/model/mapping_operator.py` | `tests/model/test_mapping_operator.py` |
| `src/cellmapper/model/neighbors.py` | `tests/model/test_neighbors.py` |
| `src/cellmapper/model/embedding.py` | `tests/model/test_embedding.py` |
| `src/cellmapper/model/evaluate.py` | `tests/model/test_evaluate.py` |
| `src/cellmapper/model/_knn_backend.py` | `tests/model/test_neighbors.py`, `tests/model/test_kernel.py` |
| `src/cellmapper/check.py` | `tests/test_check.py` |
| `src/cellmapper/utils.py` | `tests/test_utils.py` |
| Any end-to-end behavioral change | also check `tests/test_basic.py` |

Cross-cutting fixture changes: inspect `tests/conftest.py` and `tests/data/`.

## Testing

Apply these checks whenever the PR touches code or tests.

**New code.** Confirm that new behavior is covered by tests.
- Reuse fixtures from `tests/conftest.py` rather than creating parallel ones.
- Prefer `pytest.mark.parametrize` over many near-identical tests.
- Favor few meaningful tests over many redundant ones; flag low-value tests that only duplicate existing coverage.

**Failing tests.** If CI is red, do not wave it through.
- Inspect which tests fail and why (`gh pr checks`, `gh run view --log-failed`).
- Distinguish critical regressions (mapping-matrix semantics, key-naming contract, backend gating) from trivial or flaky failures.
- Surface critical failures back to the author and ask them to fix before merge.

**Modified tests.** Scrutinize *how* existing tests were changed.
- PRs that only relax thresholds, remove assertions, delete cases, or loosen `parametrize` matrices are a red flag — tests-working-around-tests defeats the purpose.
- Require an explicit justification in the PR body for any weakened assertion; do not accept silently.

## Documentation Impact

A single behavioral or API change often touches docs in multiple places. Check
both audiences and ask the author to update what is stale. Point to the
**owning file** for each topic (see the `AGENTS.md` "Where To Find What" table)
rather than duplicating content in your review.

**Human-facing docs (`docs/`, Sphinx/RTD).**
- API signature or public symbol changes → `docs/api.md` and autosummary entries; any prose referencing the symbol.
- Contributor workflow, environment, or build changes → `docs/contributing.md`.
- Tutorials under `docs/notebooks/tutorials/` → flag stale imports, outputs, or API usage.
- Method-level behavior → docstrings in `src/cellmapper/`.

**Agent-facing docs (repo root and `.github/`).**
- Invariants or development commands changed → `AGENTS.md` (Critical Invariants, Development Commands).
- Review workflow, risk areas, or testing conventions changed → `REVIEW_GUIDE.md` (this file).
- Repo structure, new top-level docs, or moved pointers → `AGENTS.md` "Where To Find What" table, `CLAUDE.md`, `.github/copilot-instructions.md`.

If behavior changes but the relevant docs do not, call it out explicitly in the
review and request the update.

## Review Checklist

- Does the change preserve the invariants in `AGENTS.md`?
- Does CI pass, and were any failures investigated? (See [Testing](#testing).)
- Is test coverage adequate and non-redundant, and are modified tests not simply weakened? (See [Testing](#testing).)
- Does it alter the mapping-matrix contract (row-stochastic, CSR, float32) or the matrix-power self-mapping gate?
- Does it change the output key-naming contract or the `query_imputed` construction path?
- Are optional-dependency imports properly gated via `check.check_deps()`?
- Are all affected human- and agent-facing docs updated? (See [Documentation Impact](#documentation-impact).)
- Is the PR scope tight — no unrelated changes bundled in — and is the public API surface kept minimal?

## PR Metadata

This repo uses a structured PR template
(`.github/PULL_REQUEST_TEMPLATE.md`).

Reviewers and agents should treat these sections as the preferred summary surface:
- summary
- behavior or invariants changed
- tests run
- reviewer focus
- context
- open questions or follow-ups
