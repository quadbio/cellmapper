import numpy as np
import pytest

from cellmapper.cellmapper import CellMapper


def assert_metrics_close(actual: dict, expected: dict, atol=1e-3):
    for key, exp in expected.items():
        act = actual[key]
        if isinstance(exp, float | int | np.floating | np.integer):
            assert np.isclose(act, exp, atol=atol), f"{key}: {act} != {exp}"
        else:
            assert act == exp, f"{key}: {act} != {exp}"


class TestCellMapper:
    def test_label_transfer(self, cmap, expected_label_transfer_metrics):
        cmap.transfer_labels(obs_keys="leiden")
        cmap.evaluate_label_transfer(label_key="leiden")
        assert_metrics_close(cmap.label_transfer_metrics, expected_label_transfer_metrics)

    def test_embedding_transfer(self, cmap):
        cmap.transfer_embeddings(obsm_keys="X_pca")
        assert "X_pca_pred" in cmap.query.obsm
        assert cmap.query.obsm["X_pca_pred"].shape[0] == cmap.query.n_obs

    def test_expression_transfer(self, cmap, expected_expression_transfer_metrics):
        cmap.transfer_expression(layer_key="X")
        cmap.evaluate_expression_transfer(layer_key="X", method="pearson")
        assert_metrics_close(cmap.expression_transfer_metrics, expected_expression_transfer_metrics)

    @pytest.mark.parametrize("method", ["gaussian", "scarches", "random", "inverse_distance", "jaccard", "hnoca"])
    def test_compute_mapping_matrix_all_methods(self, cmap, method):
        cmap.compute_mappping_matrix(method=method)
        assert cmap.mapping_matrix is not None

    @pytest.mark.parametrize("layer_key", ["X", "counts"])
    def test_expression_transfer_layers(self, cmap, layer_key):
        cmap.transfer_expression(layer_key=layer_key)
        assert cmap.query_imputed is not None
        assert cmap.query_imputed.X.shape[0] == cmap.query.n_obs

    @pytest.mark.parametrize(
        "joint_pca_key,n_pca_components,pca_kwargs",
        [
            ("pca_joint", 10, {}),
            ("custom_pca", 5, {"svd_solver": "arpack"}),
        ],
    )
    def test_compute_neighbors_joint_pca(self, cmap, joint_pca_key, n_pca_components, pca_kwargs):
        cmap.compute_neighbors(
            n_neighbors=3,
            use_rep=None,
            joint_pca_key=joint_pca_key,
            n_pca_components=n_pca_components,
            pca_kwargs=pca_kwargs,
        )
        assert joint_pca_key in cmap.ref.obsm
        assert joint_pca_key in cmap.query.obsm
        assert cmap.ref.obsm[joint_pca_key].shape[1] == n_pca_components
        assert cmap.query.obsm[joint_pca_key].shape[1] == n_pca_components

    @pytest.mark.parametrize(
        "obs_keys,obsm_keys,layer_key",
        [
            ("leiden", None, None),
            (None, "X_pca", None),
            (None, None, "X"),
            ("leiden", "X_pca", None),
            ("leiden", None, "X"),
            (None, "X_pca", "X"),
            ("leiden", "X_pca", "X"),
        ],
    )
    def test_fit_various_combinations(self, cmap, obs_keys, obsm_keys, layer_key):
        cmap.fit(obs_keys=obs_keys, obsm_keys=obsm_keys, layer_key=layer_key)
        if obs_keys is not None:
            keys = [obs_keys] if isinstance(obs_keys, str) else obs_keys
            for key in keys:
                assert f"{key}_pred" in cmap.query.obs
        if obsm_keys is not None:
            keys = [obsm_keys] if isinstance(obsm_keys, str) else obsm_keys
            for key in keys:
                assert f"{key}_pred" in cmap.query.obsm
        if layer_key is not None:
            assert cmap.query_imputed is not None
            assert cmap.query_imputed.X.shape[0] == cmap.query.n_obs

    def test_transfer_labels_self_mapping(self, query_ref_adata):
        """Check mapping to self."""
        _, ref = query_ref_adata
        cm = CellMapper(ref, ref)
        cm.fit(
            knn_method="sklearn",
            mapping_method="jaccard",
            obs_keys="leiden",
            use_rep="X_pca",
            n_neighbors=1,
            prediction_postfix="transfer",
        )
        assert "leiden_transfer" in ref.obs
        assert len(ref.obs["leiden_transfer"]) == len(ref.obs["leiden"])
        # Check that all predicted labels are valid categories
        assert set(ref.obs["leiden_transfer"].cat.categories) <= set(ref.obs["leiden"].cat.categories)
        # If mapping to self, labels should match
        assert ref.obs["leiden_transfer"].equals(ref.obs["leiden"])
