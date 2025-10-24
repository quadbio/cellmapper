from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from cellmapper.model.cellmapper import CellMapper


@pytest.fixture
def sample_distances():
    # 3 samples, 2 neighbors each
    return np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=np.float64)


@pytest.fixture
def sample_indices():
    # 3 samples, 2 neighbors each
    return np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)


@pytest.fixture
def small_data():
    # 5 points in 2D, easy to check neighbors
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], dtype=np.float64)
    y = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], dtype=np.float64)
    return x, y


@pytest.fixture
def precomputed_leiden():
    """Fixture to load precomputed leiden clustering."""
    data_path = files("tests.data") / "precomputed_leiden.csv"
    leiden_cl = pd.read_csv(str(data_path), index_col=0)["leiden"]

    return leiden_cl


@pytest.fixture
def adata_pbmc3k(precomputed_leiden):
    adata = sc.datasets.pbmc3k()

    # basic cell and gene filtering
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Saving count data
    adata.layers["counts"] = adata.X.copy()

    # Normalizing to median total counts
    sc.pp.normalize_total(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)

    # compute hvgs
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    # PCA
    sc.tl.pca(adata, mask_var="highly_variable")

    # Compute diffusion pseudotime for testing numerical obs mapping
    # First compute neighbors and diffusion map
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    sc.tl.diffmap(adata)

    # Set root cell for pseudotime computation (use first cell)
    adata.uns["iroot"] = 0
    sc.tl.dpt(adata)

    # Load precomputed leiden clustering
    adata.obs["leiden"] = precomputed_leiden.astype("str").astype("category")

    return adata


@pytest.fixture
def adata_spatial(adata_pbmc3k):
    """Fixture to load spatial data with realistic coordinates for testing distance calculations.

    This can be used to test the loading of pre-computed distance matrices. It contains toy spatial
    coordinates and batches, so that neighbors can be computed with either scanpy or squidpy.

    Spatial patterns:
    - Batch A: Concentric rings structure (inner and outer ring)
    - Batch B: Grid-like pattern

    The patterns have some overlap in the center of the coordinate system, creating a
    biologically plausible scenario where different cell types might be present in the same area.
    """
    # Get number of observations
    n_obs = adata_pbmc3k.n_obs
    n_half = n_obs // 2

    # Introduce batch categories first
    adata_pbmc3k.obs["batch"] = np.repeat(["A", "B"], repeats=n_half).tolist()
    adata_pbmc3k.obs["batch"] = adata_pbmc3k.obs["batch"].astype("category")

    # Create coordinates using numpy's random generator with fixed seed
    rng = np.random.Generator(np.random.PCG64(42))
    coords = np.zeros((n_obs, 2))

    # First half - batch A: concentric rings structure
    # Inner ring
    inner_count = n_half // 3
    angles_inner = rng.uniform(0, 2 * np.pi, inner_count)
    radii_inner = rng.normal(5, 1, inner_count)  # Smaller radius with less variation
    coords[:inner_count, 0] = radii_inner * np.cos(angles_inner)
    coords[:inner_count, 1] = radii_inner * np.sin(angles_inner)

    # Outer ring
    outer_count = n_half - inner_count
    angles_outer = rng.uniform(0, 2 * np.pi, outer_count)
    radii_outer = rng.normal(15, 2, outer_count)  # Larger radius with more variation
    coords[inner_count:n_half, 0] = radii_outer * np.cos(angles_outer)
    coords[inner_count:n_half, 1] = radii_outer * np.sin(angles_outer)

    # Second half - batch B: grid-like structure with noise that overlaps with batch A
    # Create a grid that overlaps with the inner ring from batch A
    grid_size = int(np.sqrt(n_half))
    x_grid = np.linspace(-10, 10, grid_size)
    y_grid = np.linspace(-10, 10, grid_size)

    # Create all combinations of x and y coordinates
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Add some noise to the grid points
    grid_points += rng.normal(0, 1, size=grid_points.shape)

    # Use the grid points (or as many as needed)
    max_points = min(grid_points.shape[0], n_half)
    coords[n_half : n_half + max_points] = grid_points[:max_points]

    # If we need more points than the grid provides, add random points
    if max_points < n_half:
        remaining = n_half - max_points
        random_points = rng.uniform(-10, 10, size=(remaining, 2))
        coords[n_half + max_points :] = random_points

    # Add the spatial coordinates to the AnnData object
    adata_pbmc3k.obsm["spatial"] = coords

    return adata_pbmc3k


@pytest.fixture
def query_reference_adata(adata_pbmc3k):
    # Define the number of query cells and genes
    n_query_cells = 500
    n_query_genes = 300
    n_reference_cells = adata_pbmc3k.n_obs - n_query_cells

    # Create modality annotations in the AnnData object
    adata_pbmc3k.obs["modality"] = (
        np.repeat("query", repeats=n_query_cells).tolist() + np.repeat("reference", repeats=n_reference_cells).tolist()
    )
    adata_pbmc3k.obs["modality"] = adata_pbmc3k.obs["modality"].astype("category")

    # use these annotations to create query and reference AnnData objects
    query = adata_pbmc3k[adata_pbmc3k.obs["modality"] == "query"].copy()
    reference = adata_pbmc3k[adata_pbmc3k.obs["modality"] == "reference"].copy()

    # Subset genes in the query AnnData object
    query_genes = (
        query.var[query.var["highly_variable"]].sort_values("means", ascending=False).head(n_query_genes).index.tolist()
    )
    query = query[:, query_genes].copy()

    # Introduce two deterministic batch categories in the query AnnData object
    query.obs["batch"] = np.repeat(["A", "B"], repeats=n_query_cells // 2).tolist()
    query.obs["batch"] = query.obs["batch"].astype("category")

    return query, reference


@pytest.fixture
def cmap(query_reference_adata):
    query, reference = query_reference_adata

    # Create a CellMapper object
    cmap = CellMapper(
        query=query,
        reference=reference,
    )

    # Compute neighbors and mapping matrix
    cmap.compute_neighbors(n_neighbors=30, use_rep="X_pca", knn_method="sklearn")
    cmap.compute_mapping_matrix(kernel_method="gauss")

    return cmap


@pytest.fixture
def expected_label_transfer_metrics():
    return {
        "accuracy": 0.954,
        "precision": 0.955,
        "recall": 0.954,
        "f1_weighted": 0.951,
        "f1_macro": 0.899,
        "excluded_fraction": 0.0,
    }


@pytest.fixture
def expected_expression_transfer_metrics():
    return {
        "comparison_method": "pearson",
        "average": 0.376,
        "n_shared_genes": 300,
        "n_test_genes": 300,
    }


@pytest.fixture
def evaluation_methods():
    """Fixture providing a list of evaluation methods for expression transfer."""
    return ["pearson", "spearman", "js", "rmse"]


@pytest.fixture
def random_imputed_data(query_reference_adata):
    """Fixture providing random expression data with the right shape for query_imputed testing."""
    query, reference = query_reference_adata
    return np.random.rand(query.n_obs, reference.n_vars)


@pytest.fixture
def sparse_imputed_data(query_reference_adata):
    """Fixture providing a sparse expression matrix for query_imputed testing."""
    from scipy.sparse import csr_matrix

    query, reference = query_reference_adata
    return csr_matrix(np.random.rand(query.n_obs, reference.n_vars))


@pytest.fixture
def dataframe_imputed_data(query_reference_adata):
    """Fixture providing a pandas DataFrame with expression data for query_imputed testing."""
    import pandas as pd

    query, reference = query_reference_adata
    return pd.DataFrame(
        np.random.rand(query.n_obs, reference.n_vars), index=query.obs_names, columns=reference.var_names
    )


@pytest.fixture
def custom_anndata_imputed(query_reference_adata):
    """Fixture providing a custom AnnData object for query_imputed testing."""
    import anndata as ad

    query, reference = query_reference_adata
    return ad.AnnData(X=np.random.rand(query.n_obs, reference.n_vars), obs=query.obs.copy(), var=reference.var.copy())


@pytest.fixture
def invalid_shape_data(query_reference_adata):
    """Fixture providing expression matrices with invalid shapes."""
    query, reference = query_reference_adata
    too_few_cells = np.random.rand(query.n_obs - 5, reference.n_vars)
    too_few_genes = np.random.rand(query.n_obs, reference.n_vars - 10)
    return {"too_few_cells": too_few_cells, "too_few_genes": too_few_genes}


@pytest.fixture
def small_symmetric_matrix():
    """Small symmetric matrix for MappingOperator testing."""
    matrix = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.8], [0.5, 0.8, 0.0]], dtype=np.float64)
    return matrix


@pytest.fixture
def small_asymmetric_matrix():
    """Small asymmetric matrix for MappingOperator testing."""
    matrix = np.array([[0.0, 1.0, 0.5], [0.8, 0.0, 0.9], [0.3, 0.6, 0.0]], dtype=np.float64)
    return matrix


@pytest.fixture
def rectangular_matrix():
    """Rectangular matrix for cross-mapping testing."""
    matrix = np.array([[0.8, 0.2, 0.1, 0.0], [0.1, 0.7, 0.3, 0.2], [0.0, 0.1, 0.8, 0.5]], dtype=np.float64)
    return matrix


@pytest.fixture
def matrix_with_zero_rows():
    """Matrix with zero row sums for edge case testing."""
    matrix = np.array(
        [
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0],  # Zero row
            [0.5, 0.8, 0.0],
        ],
        dtype=np.float64,
    )
    return matrix


@pytest.fixture
def sparse_symmetric_matrix():
    """Sparse symmetric matrix for testing."""
    from scipy.sparse import csr_matrix

    matrix = np.array(
        [[0.0, 1.0, 0.0, 0.5], [1.0, 0.0, 0.8, 0.0], [0.0, 0.8, 0.0, 0.3], [0.5, 0.0, 0.3, 0.0]], dtype=np.float64
    )
    return csr_matrix(matrix)


@pytest.fixture
def test_data_for_mapping():
    """Test data matrix for mapping operations."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    return data
