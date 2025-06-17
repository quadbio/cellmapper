# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

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
