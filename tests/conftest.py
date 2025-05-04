from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from cellmapper.cellmapper import CellMapper


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

    # Load precomputed leiden clustering
    adata.obs["leiden"] = precomputed_leiden.astype("str").astype("category")

    return adata


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
    cmap.compute_neighbors(n_neighbors=30, use_rep="X_pca", method="sklearn")
    cmap.compute_mappping_matrix(method="gaussian")

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
        "method": "pearson",
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
