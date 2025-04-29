import numpy as np
import pytest

from cellmapper.knn import Neighbors


def assert_adjacency_equal(neigh1, neigh2, attrs=("xx", "yy", "xy", "yx")):
    for attr in attrs:
        adj1 = getattr(neigh1, attr).boolean_adjacency().toarray()
        adj2 = getattr(neigh2, attr).boolean_adjacency().toarray()
        assert np.allclose(adj1, adj2), f"Adjacency matrices ({attr}) differ between methods"


class TestNeighbors:
    @pytest.mark.parametrize("only_yx", [False, True])
    def test_neighbors_sklearn_vs_pynndescent(self, small_data, only_yx):
        x, y = small_data
        n_neighbors = 3
        # sklearn
        neigh_skl = Neighbors(x, y)
        neigh_skl.compute_neighbors(n_neighbors=n_neighbors, method="sklearn", only_yx=only_yx)
        # pynndescent
        neigh_pynn = Neighbors(x, y)
        neigh_pynn.compute_neighbors(n_neighbors=n_neighbors, method="pynndescent", only_yx=only_yx)
        if only_yx:
            with pytest.raises(ValueError):
                neigh_skl.get_adjacency_matrices()
            with pytest.raises(ValueError):
                neigh_pynn.get_adjacency_matrices()
        else:
            assert_adjacency_equal(neigh_skl, neigh_pynn)
        # Always compare connectivities (yx)
        conn_skl = neigh_skl.yx.knn_graph_connectivities()
        conn_pynn = neigh_pynn.yx.knn_graph_connectivities()
        assert np.allclose(conn_skl.toarray(), conn_pynn.toarray(), atol=1e-6)

    def test_neighbors_repr(self, small_data):
        x, y = small_data
        neigh = Neighbors(x, y)
        r = repr(neigh)
        assert "Neighbors(" in r and "xrep_shape" in r and "yrep_shape" in r
        neigh.compute_neighbors(n_neighbors=2, method="sklearn")
        r2 = repr(neigh)
        assert "xx=True" in r2 and "yy=True" in r2 and "xy=True" in r2 and "yx=True" in r2
