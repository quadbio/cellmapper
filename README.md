# CellMapper

[![Tests][badge-tests]][tests]
[![Coverage][badge-coverage]][coverage]
[![Pre-commit.ci][badge-pre-commit]][pre-commit]
[![PyPI][badge-pypi]][pypi]
[![Documentation][badge-docs]][docs]
[![Downloads][badge-downloads]][downloads]
[![Zenodo][badge-zenodo]][zenodo]

[badge-tests]: https://github.com/quadbio/cellmapper/actions/workflows/test.yaml/badge.svg
[badge-coverage]: https://codecov.io/gh/quadbio/cellmapper/branch/main/graph/badge.svg
[badge-pre-commit]: https://results.pre-commit.ci/badge/github/quadbio/cellmapper/main.svg
[badge-pypi]: https://img.shields.io/pypi/v/cellmapper.svg
[badge-docs]: https://img.shields.io/readthedocs/cellmapper
[badge-downloads]: https://static.pepy.tech/badge/cellmapper
[badge-zenodo]: https://zenodo.org/badge/973714072.svg

k-NN-based mapping of cells across representations to transfer labels, embeddings and expression values. Works for millions of cells, on CPU and GPU, across molecular modalities, between spatial and non-spatial data, for arbitrary query and reference datasets. Using [faiss][] to compute k-NN graphs, CellMapper takes about 30 seconds to transfer cell type labels from 1.5M cells to 1.5M cells on a single RTX 4090 with 60 GB CPU memory.

Inspired by previous tools, including scanpy's [ingest][] and the [HNOCA-tools][] packages. Check out the 📚 [docs][] to learn more, in particular our [tutorials][].

## ✨ Key use cases

- 🧬 Transfer cell type labels and expression values from dissociated to spatial datasets.
- ↔️ Transfer embeddings between arbitrary query and reference datasets.
- 📊 Compute presence scores for query datasets in large reference atlasses.
- 🗺️ Identify niches in spatial datasets by contextualizing latent spaces in spatial coordinates.
- 📈 Evaluate the results of transferring labels, embeddings and feature spaces using a variety of metrics.

The core idea of `CellMapper` is to separate the method (k-NN graph with some kernel applied to get a mapping matrix) from the application (mapping across arbitrary representations), to be flexible and fast. The tool currently supports [pynndescent][], [sklearn][], [faiss][] and [rapids][] for neighborhood search, implements a variety of graph kernels, and is closely integrated with `AnnData` objects.

## 📦 Installation

You need to have 🐍 Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are two alternative options to install ``cellmapper``:

- 🚀 Install the latest release from [PyPI][]:

  ```bash
  pip install cellmapper
  ```

- 🛠️ Install the latest development version:

  ```bash
  pip install git+https://github.com/quadbio/cellmapper.git@main
  ```

## 🏁 Getting started

This package assumes that you have ``query`` and ``reference`` AnnData objects, with a joint embedding computed and stored in ``.obsm``. While we implement some baseline approaches to compute joint embeddings (PCA and a fast reimplementation of CCA), we typically expect you to provide a pre-computed joint embedding from some task-specific representation learning tools, e.g. [GimVI][] or [ENVI][] for spatial mapping, [GLUE][], [MIDAS][] and [MOFA+][] for modality translation, and [scVI][], [scANVI][] and [scArches][] for query-to-reference mapping - this is just a small selection!

With a joint embedding in ``.obsm["X_joint"]`` at hand, the simplest way to use ``CellMapper`` is as follows:
```Python
from cellmapper import CellMapper

cmap = CellMapper(query, reference).map(
    use_rep="X_joint", obs_keys="celltype", obsm_keys="X_umap", layer_key="X"
    )
```

This will transfer data from the reference to the query dataset, including celltype labels stored in ``reference.obs``, a UMAP embedding stored in ``reference.obsm``, and expression values stored in ``reference.X``.

There are many ways to customize this, e.g. use different ways to compute k-NN graphs and to turn them into mapping matrices, and we implement a few methods to evaluate whether your k-NN transfer was sucessful. The tool also implements a `self-mapping` mode (only a query object, no reference), which is useful for spatial contextualization and data denoising. Check out the 📚 [docs][] to learn more.

## 📝 Release notes

See the [changelog][].

## 📬 Contact

If you found a bug, please use the [issue tracker][].

## 📖 Citation
Please use our [zenodo][] entry to cite this software.

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/quadbio/cellmapper/issues
[tests]: https://github.com/quadbio/cellmapper/actions/workflows/test.yaml
[changelog]: https://cellmapper.readthedocs.io/en/latest/changelog.html
[docs]: https://cellmapper.readthedocs.io/
[tutorials]: https://cellmapper.readthedocs.io/en/latest/notebooks/tutorials/index.html
[pypi]: https://pypi.org/project/cellmapper
[coverage]: https://codecov.io/gh/quadbio/cellmapper
[pre-commit]: https://results.pre-commit.ci/latest/github/quadbio/cellmapper/main
[pypi]: https://pypi.org/project/cellmapper/
[downloads]: https://pepy.tech/project/cellmapper
[zenodo]: https://doi.org/10.5281/zenodo.15683594

[faiss]: https://github.com/facebookresearch/faiss
[pynndescent]: https://github.com/lmcinnes/pynndescent
[sklearn]: https://scikit-learn.org/stable/modules/neighbors.html
[rapids]: https://docs.rapids.ai/api/cuml/stable/api/#nearest-neighbors

[ingest]: https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.ingest.html
[HNOCA-tools]: https://devsystemslab.github.io/HNOCA-tools/

[GimVI]: https://docs.scvi-tools.org/en/stable/api/reference/scvi.external.GIMVI.html#
[ENVI]: https://scenvi.readthedocs.io/en/latest/#
[GLUE]: https://scglue.readthedocs.io/en/latest/
[MIDAS]: https://scmidas.readthedocs.io/en/latest/
[MOFA+]: https://muon.readthedocs.io/en/latest/omics/multi.html
[scVI]: https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html
[scANVI]: https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCANVI.html
[scArches]: https://docs.scarches.org/en/latest/
