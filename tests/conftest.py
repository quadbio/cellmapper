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
def query_ref_adata(adata_pbmc3k):
    # Define the number of query cells and genes
    n_query_cells = 500
    n_query_genes = 300
    n_ref_cells = adata_pbmc3k.n_obs - n_query_cells

    # Create modality annotations in the AnnData object
    adata_pbmc3k.obs["modality"] = (
        np.repeat("query", repeats=n_query_cells).tolist() + np.repeat("ref", repeats=n_ref_cells).tolist()
    )
    adata_pbmc3k.obs["modality"] = adata_pbmc3k.obs["modality"].astype("category")

    # use these annotations to create query and reference AnnData objects
    query = adata_pbmc3k[adata_pbmc3k.obs["modality"] == "query"].copy()
    ref = adata_pbmc3k[adata_pbmc3k.obs["modality"] == "ref"].copy()

    # Subset genes in the query AnnData object
    query_genes = (
        query.var[query.var["highly_variable"]].sort_values("means", ascending=False).head(n_query_genes).index.tolist()
    )
    query = query[:, query_genes].copy()

    return query, ref


@pytest.fixture
def cmap(query_ref_adata):
    query, ref = query_ref_adata

    # Create a CellMapper object
    cmap = CellMapper(
        query=query,
        ref=ref,
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
        "average_correlation": 0.376,
        "n_genes": 300,
        "n_valid_genes": 300,
    }
