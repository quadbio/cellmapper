# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

## [v0.2.2]
### Added
- Enabled subsetting categories before mapping .obs values {pr}`46`

### Changed
- Updated the README a bit {pr}`44`
- Updated tutorials to work with new parameter names {pr}`43`

### Fices
- Fixed a small bug where hvg masks would not be propagated correctly to joint pca computation {pr}`48`

## [v0.2.1]
### Changed
- Move some duplicated docstrings into a central _docs.py file {pr}`41`

### Added
- Added some tests for edge cases in the `MappingOperator` class {pr}`41`
- Treat faiss-cpu and faiss-gpu separately {pr}`41`


## [v0.2.0]

### Added
- Added a tutorial on same-modality query to reference mapping {pr}`38`
- Added a tutorial on data smoothing {pr}`37`
- Added an option to return the mapping probabilities for categorical `.obs` mapping {pr}`39`
- Added a `MappingOperator` class which allows for iterative mapping matrix applicatino in self-mapping mode {pr}`35`
- Add the `umap` method to compute symmetric k-NN connectivities in self-mapping mode {pr}`34`

### Changed
- Refectored the neighbors classes into a `Neighobrs` and a `Kernel` class and moved symmetrization into the `Kernel` class {pr}`36`


## [v0.1.4]
### Changed
- Rename mapping methods to `map_obs`, `map_obsm`, and `map_layers`, and improve support for numerical `.obs` annotations {pr}`30`.


## [v0.1.3]

### Added
- Added a tutorial on spatial contextualization and niche identification {pr}`23`.
- Implemented a self-mapping mode with only a query dataset {pr}`21`.
- Allow importing a pre-computed dataset of transfered expression values {pr}`21`.
- Allow importing pre-computed neighborhood matrices {pr}`21`.
- Add a tutorial on spatial contextualization and niche identification {pr}`21`.
- Add an equal-weight kernel {pr}`22`.

## [v0.1.2]

### Added
- Included tests for the `check` module, and more tests for the main classes {pr}`15`.
- Implemented the computation of presence scores, following HNOCA-tools {pr}`16`.
- Added a `groupby` parameter to expression transfer evaluation {pr}`16`.
- Added a `test_var_key` parameter to expression transfer evaluation {pr}`19`.
- Added a tutorial on spatial mapping {pr}`19`.

## [v0.1.1]

### Changed
- Switched to `vcs`-based versioning {pr}`5`.

### Added
- Added PyPI badge.

## [v0.1.0]
Initial package release.
