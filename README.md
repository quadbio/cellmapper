# CellMapper

[![Tests][badge-tests]][tests]
[![Coverage][badge-coverage]][coverage]
[![Pre-commit.ci][badge-pre-commit]][pre-commit]

[badge-tests]: https://github.com/quadbio/cellmapper/actions/workflows/test.yaml/badge.svg
[badge-coverage]: https://codecov.io/gh/quadbio/cellmapper/branch/main/graph/badge.svg
[badge-pre-commit]: https://results.pre-commit.ci/badge/github/quadbio/cellmapper/main.svg

k-NN-based mapping of cells across representations to tranfer labels, embeddings and expression values. Works for millions of cells, on CPU and GPU, across molecular modalities, between spatial and non-spatial data, for arbitrary query and reference datasets. Using `faiss` to compute k-NN graphs, CellMapper takes about 30 seconds to transfer cell type labels from 1.5M cells to 1.5M cells on a single RTX 4090 with 60 GB CPU memory.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install cellmapper:

<!--
1) Install the latest release of `cellmapper` from [PyPI][]:

```bash
pip install cellmapper
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/quadbio/cellmapper.git@main
```

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
