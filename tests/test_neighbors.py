import numpy as np

from cellmapper.knn import Neighbors


def assert_adjacency_equal(neigh1, neigh2, attrs=("xx", "yy", "xy", "yx")):
    for attr in attrs:
        # Direct attribute access instead of getattr for Ruff compliance
        adj1 = getattr(neigh1, attr).boolean_adjacency().toarray()
        adj2 = getattr(neigh2, attr).boolean_adjacency().toarray()
        assert np.allclose(adj1, adj2), f"Adjacency matrices ({attr}) differ between methods"


def test_neighbors_sklearn_vs_pynndescent(small_data):
    x, y = small_data
    n_neighbors = 3
    # sklearn
    neigh_skl = Neighbors(x, y)
    neigh_skl.compute_neighbors(n_neighbors=n_neighbors, method="sklearn")
    # pynndescent
    neigh_pynn = Neighbors(x, y)
    neigh_pynn.compute_neighbors(n_neighbors=n_neighbors, method="pynndescent")
    # Compare adjacency matrices
    assert_adjacency_equal(neigh_skl, neigh_pynn)


def test_neighbors_repr(small_data):
    x, y = small_data
    neigh = Neighbors(x, y)
    r = repr(neigh)
    assert "Neighbors(" in r and "xrep_shape" in r and "yrep_shape" in r
    neigh.compute_neighbors(n_neighbors=2, method="sklearn")
    r2 = repr(neigh)
    assert "xx=True" in r2 and "yy=True" in r2 and "xy=True" in r2 and "yx=True" in r2
