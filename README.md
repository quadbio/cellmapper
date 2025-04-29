# CellMapper

[![Tests][badge-tests]][tests]
[![Coverage][badge-coverage]][coverage]
[![Pre-commit.ci][badge-pre-commit]][pre-commit]
[![PyPI][badge-pypi]][pypi]

[badge-tests]: https://github.com/quadbio/cellmapper/actions/workflows/test.yaml/badge.svg
[badge-coverage]: https://codecov.io/gh/quadbio/cellmapper/branch/main/graph/badge.svg
[badge-pre-commit]: https://results.pre-commit.ci/badge/github/quadbio/cellmapper/main.svg
[badge-pypi]: https://img.shields.io/pypi/v/cellmapper.svg

k-NN-based mapping of cells across representations to tranfer labels, embeddings and expression values. Works for millions of cells, on CPU and GPU, across molecular modalities, between spatial and non-spatial data, for arbitrary query and reference datasets. Using [faiss][] to compute k-NN graphs, CellMapper takes about 30 seconds to transfer cell type labels from 1.5M cells to 1.5M cells on a single RTX 4090 with 60 GB CPU memory.

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are two alternative options to install ``cellmapper``:

- **Install the latest release from [PyPI][]**:

  ```bash
  pip install cellmapper
  ```

- **Install the latest development version**:

  ```bash
  pip install git+https://github.com/quadbio/cellmapper.git@main
  ```

## Getting started

This package assumes that you have ``ref`` and ``query`` AnnData objects, with a joint embedding computed and stored in ``.obsm``. We explicilty do not compute this joint embedding, but there are plenty of method you can use to get such joint embeddings, e.g. [GimVI][] or [ENVI][] for spatial mapping, [GLUE][], [MIDAS][] and [MOFA+][] for modality translation, and [scVI][], [scANVI][] and [scArches][] for query-to-reference mapping - this is just a small selection!

With a joint embedding in ``.obsm["X_joint"]`` at hand, the simplest way to use ``CellMapper`` is as follows:
```Python
from cellmapper import CellMapper

cmap = CellMapper(ref, query).fit(
    use_rep="X_joint", obs_keys="celltype", obsm_keys="X_umap", layer_key="X"
    )
```

This will transfer data from the reference to the query dataset, including celltype labels stored in ``ref.obs``, a UMAP embedding stored in ``ref.obsm``, and expression values stored in ``ref.X``.

There are many ways to customize this, e.g. use different ways to compute k-NN graphs and to turn them into mapping matrices, and we implement a few methods to evaluate whether your k-NN transfer was sucessful.

## Release notes

See the [changelog][].

## Contact

If you found a bug, please use the [issue tracker][].

## Citation

Please cite this GitHub repo if you find CellMapper useful for your research.

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/quadbio/cellmapper/issues
[tests]: https://github.com/quadbio/cellmapper/actions/workflows/test.yaml
[changelog]: https://cellmapper.readthedocs.io/en/latest/changelog.html
[api documentation]: https://cellmapper.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/cellmapper
[coverage]: https://codecov.io/gh/quadbio/cellmapper
[pre-commit]: https://results.pre-commit.ci/latest/github/quadbio/cellmapper/main
[pypi]: https://pypi.org/project/cellmapper/
[faiss]: https://github.com/facebookresearch/faiss

[GimVI]: https://docs.scvi-tools.org/en/stable/api/reference/scvi.external.GIMVI.html#
[ENVI]: https://scenvi.readthedocs.io/en/latest/#
[GLUE]: https://scglue.readthedocs.io/en/latest/
[MIDAS]: https://scmidas.readthedocs.io/en/latest/
[MOFA+]: https://muon.readthedocs.io/en/latest/omics/multi.html
[scVI]: https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html
[scANVI]: https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCANVI.html
[scArches]: https://docs.scarches.org/en/latest/
