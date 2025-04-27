# CellMapper

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://github.com/quadbio/cellmapper/actions/workflows/test.yaml/badge.svg
[badge-docs]: https://img.shields.io/readthedocs/cellmapper

k-NN-based mapping of cells across representations to tranfer labels, embeddings and expression values. Works for millions of cells, on CPU and GPU, across molecular modalities, between spatial and non-spatial data, for arbitrary query and reference datasets.

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

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

Please cite this GitHub repo if you find CellMapper useful for your research.

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/quadbio/cellmapper/issues
[tests]: https://github.com/quadbio/cellmapper/actions/workflows/test.yaml
[documentation]: https://cellmapper.readthedocs.io
[changelog]: https://cellmapper.readthedocs.io/en/latest/changelog.html
[api documentation]: https://cellmapper.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/cellmapper
