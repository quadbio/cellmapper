# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

## [v0.2.5]

### Added
- Added `batch_size` parameter to `Kernel.compute_neighbors()` for backend-agnostic batching {pr}`60`
- Added `subset` parameter to `plot_confusion_matrix()` for filtering cells {pr}`60`
- Added Python 3.14 support {pr}`62`

### Changed
- Default postfixes now include underscore: `"_pred"` and `"_conf"` instead of `"pred"` and `"conf"` {pr}`60`
- Switched from MathJax to KaTeX for documentation math rendering {pr}`61`
- Updated GitHub Actions to latest versions (checkout v5, setup-uv v7) {pr}`61`
- Updated test matrix to Python 3.11/3.14 {pr}`62`

### Fixed
- Fixed `PackageNotFoundError` when checking rapids availability in conda environments {pr}`60`
- Fixed stale k-NN state when `compute_neighbors()` fails {pr}`60`
- Fixed `plot_confusion_matrix()` handling of NaN values in both y_true and y_pred {pr}`60`
- Fixed `plot_confusion_matrix()` handling of mismatched category sets {pr}`60`
- Fixed `plot_confusion_matrix()` handling of float categories {pr}`60`
- Fixed `importlib.resources.files()` compatibility with Python 3.14 {pr}`62`

## [v0.2.4]

### Changed
- Deprecated Python 3.10 support {pr}`55`
- Updated to cookiecutter-scverse template v0.6.0 {pr}`54`

## [v0.2.3]

### Changed
- Improved returning probabilities (after mapping categorical obs fields) to always return a DataFrame {pr}`49`

### Added
- Add the possibility to adjust the library size after `.map_layers` {pr}`50`
- Added the option to turn off post-processing of the presence scores, so that they can be first smoothed and then processed, like in HNOCA-tools {pr}`51`

## [v0.2.2]
### Added
- Enabled subsetting categories before mapping .obs values {pr}`46`

### Changed
- Updated the README a bit {pr}`44`
- Updated tutorials to work with new parameter names {pr}`43`

### Fixed
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
