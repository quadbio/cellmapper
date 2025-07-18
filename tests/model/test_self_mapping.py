import numpy as np
import pytest
import scanpy as sc
from scanpy.neighbors import Neighbors
from scipy.linalg import subspace_angles
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.stats import pearsonr

from cellmapper.model.cellmapper import CellMapper


def analyze_by_cell_and_gene(
    matrix_iterative: np.ndarray | csr_matrix, matrix_spectral: np.ndarray | csr_matrix
) -> tuple[np.ndarray, np.ndarray]:
    """Analyze differences at cell and gene levels.

    Parameters
    ----------
    matrix_iterative
        First matrix (typically from iterative method)
    matrix_spectral
        Second matrix (typically from spectral method)

    Returns
    -------
    cell_correlations
        Per-cell correlations between the two matrices
    gene_correlations
        Per-gene correlations between the two matrices
    """
    # Per-cell correlations
    cell_correlations = []
    for i in range(matrix_iterative.shape[0]):
        correlation, _ = pearsonr(matrix_iterative[i, :], matrix_spectral[i, :])
        cell_correlations.append(correlation)

    # Per-gene correlations
    gene_correlations = []
    for j in range(matrix_iterative.shape[1]):
        correlation, _ = pearsonr(matrix_iterative[:, j], matrix_spectral[:, j])
        gene_correlations.append(correlation)

    cell_correlations = np.array(cell_correlations)
    gene_correlations = np.array(gene_correlations)

    return cell_correlations, gene_correlations


class TestUMAPConnectivityValidation:
    """Test UMAP connectivity compatibility between scanpy and CellMapper implementations."""

    @pytest.mark.parametrize(
        "transformer,remove_last_neighbor",
        [
            ("sklearn", False),
            ("pynndescent", True),
        ],
    )
    def test_connectivities_from_distances(self, adata_pbmc3k, transformer, remove_last_neighbor):
        """
        Test that CellMapper can exactly reproduce scanpy's UMAP connectivities.

        This test validates the first part of the connectivity test tutorial:
        1. Compute k-NN graph in scanpy with UMAP method
        2. Import distances into CellMapper
        3. Validate that distances match (accounting for self-edge differences)
        4. Validate that applying UMAP kernel gives equivalent results

        Parameters
        ----------
        transformer
            The transformer to use ('sklearn' or 'pynndescent')
        remove_last_neighbor
            Whether to remove the last neighbor when loading distances
        """
        # Step 1: Compute neighbors with scanpy using UMAP method
        nbhs = Neighbors(adata_pbmc3k)
        nbhs.compute_neighbors(
            n_neighbors=8,
            use_rep="X_pca",
            n_pcs=50,
            knn=True,
            method="umap",
            metric="euclidean",
            transformer=transformer,
        )

        # Store scanpy results
        adata_pbmc3k.obsp["distances"] = nbhs.distances

        # Validate importated distances match the original
        cmap = CellMapper(adata_pbmc3k)
        cmap.load_precomputed_distances(remove_last_neighbor=False)
        assert (nbhs.distances - cmap.knn.yx.knn_graph_distances).nnz == 0, (
            "Imported distances don't match scanpy distances"
        )

        # Validate connectivities match scanpy's. Note, this is a bit weird, we now have to remove the last neighbor, only for pynndescent
        cmap.load_precomputed_distances(remove_last_neighbor=remove_last_neighbor)
        conn_cmap = cmap.knn.yx.knn_graph_connectivities(kernel="umap", self_edges=True)

        assert (nbhs.connectivities - conn_cmap).nnz == 0, "Connectivity matrices should be identical"

        # Step 6: Validate matrix properties
        assert (conn_cmap - conn_cmap.T).nnz == 0, "CellMapper connectivity matrix should be symmetric"
        assert np.allclose(conn_cmap.diagonal(), 0), "CellMapper connectivity matrix should have no self-edges"

    def test_full_cellmapper_computation(self, adata_pbmc3k):
        """
        Test that CellMapper's full computation pipeline exactly reproduces scanpy results.

        This test validates the last section of the connectivity test tutorial:
        1. Compute neighbors with scanpy using sklearn
        2. Compute neighbors with CellMapper using the same parameters
        3. Compare PCA representations (should be identical)
        4. Compare neighbor indices and distances (should be identical)
        5. Compare UMAP connectivities (should be identical)

        Only tests with sklearn as it's the only method that gives deterministic results.
        """
        transformer = "sklearn"
        n_neighbors = 8
        n_pcs = 50

        # Step 1: Compute neighbors with scanpy
        nbhs_sc = Neighbors(adata_pbmc3k)
        nbhs_sc.compute_neighbors(
            n_neighbors=n_neighbors,
            use_rep="X_pca",
            n_pcs=n_pcs,
            knn=True,
            method="umap",
            metric="euclidean",
            transformer=transformer,
        )

        # Step 2: Compute neighbors with CellMapper
        cmap = CellMapper(adata_pbmc3k)
        cmap.compute_neighbors(
            n_neighbors=n_neighbors,
            use_rep="X_pca",
            n_comps=n_pcs,
            knn_method=transformer,
            knn_dist_metric="euclidean",
        )

        # Extract CellMapper results
        assert cmap.knn is not None, "KNN should be computed"

        # Step 5: Compare distances
        assert (nbhs_sc.distances - cmap.knn.yx.knn_graph_distances).nnz == 0, (
            "Imported distances don't match scanpy distances"
        )

        # Step 6: Compute and compare UMAP connectivities
        conn_cmap = cmap.knn.yx.knn_graph_connectivities(kernel="umap", symmetrize=True, self_edges=True)

        # Check connectivity matrices are identical
        assert (conn_cmap - nbhs_sc.connectivities).nnz == 0, "UMAP connectivity matrices should be identical"

        # Validate matrix properties
        assert (conn_cmap - conn_cmap.T).nnz == 0, "CellMapper connectivity matrix should be symmetric"
        assert np.allclose(conn_cmap.diagonal(), 0), "CellMapper connectivity matrix should have no self-edges"


class TestSelfMapping:
    """Tests for self-mapping functionality in CellMapper."""

    def test_self_mapping_initialization(self, adata_pbmc3k):
        """Test that self-mapping mode is correctly detected when reference=None."""
        # Initialize with only reference
        cm = CellMapper(adata_pbmc3k)
        assert cm._is_self_mapping
        assert cm.reference is adata_pbmc3k
        assert cm.query is adata_pbmc3k

    @pytest.mark.parametrize("obs_key", ["leiden", "dpt_pseudotime"])
    def test_identity_mapping(self, adata_pbmc3k, obs_key):
        """Test that with n_neighbors=1, self-mapping preserves original labels exactly."""
        # Initialize with self-mapping
        cm = CellMapper(adata_pbmc3k)
        cm.map(
            knn_method="sklearn",
            kernel_method="jaccard",
            obs_keys=obs_key,
            use_rep="X_pca",
            n_neighbors=1,
            prediction_postfix="pred",
            # For n_neighbors=1 identity mapping, we need self-edges and no symmetrization
            symmetrize=False,
            self_edges=True,
        )

        # With n_neighbors=1, labels should be perfectly preserved
        assert f"{obs_key}_pred" in adata_pbmc3k.obs
        assert len(adata_pbmc3k.obs[f"{obs_key}_pred"]) == len(adata_pbmc3k.obs[obs_key])

        # Labels should match exactly when n_neighbors=1
        assert adata_pbmc3k.obs[f"{obs_key}_pred"].equals(adata_pbmc3k.obs[obs_key])

    def test_all_operations_self_mapping(self, adata_pbmc3k):
        """Test the full pipeline in self-mapping mode."""
        # Initialize with self-mapping
        cm = CellMapper(adata_pbmc3k)

        # Test with typical parameters
        cm.compute_neighbors(n_neighbors=5, use_rep="X_pca")
        cm.compute_mapping_matrix(kernel_method="gauss")

        # Test label transfer
        cm.map_obs(key="leiden")
        assert "leiden_pred" in cm.query.obs
        # With n_neighbors>1, self-mapped labels might not be 100% identical

        # Test embedding transfer
        cm.map_obsm(key="X_pca")
        assert "X_pca_pred" in cm.query.obsm

        # Test expression transfer
        cm.map_layers(key="X")
        assert cm.query_imputed is not None

        # Test evaluation functions
        cm.evaluate_label_transfer(label_key="leiden")
        assert cm.label_transfer_metrics is not None

        cm.evaluate_expression_transfer(layer_key="X", comparison_method="pearson")
        assert cm.expression_transfer_metrics is not None

    @pytest.mark.parametrize("n_neighbors", [5, 15, 30])
    def test_load_scanpy_distances(self, adata_spatial, n_neighbors):
        """Test loading distances computed with scanpy.pp.neighbors."""

        # Compute neighbors with scanpy
        sc.pp.neighbors(adata_spatial, n_neighbors=n_neighbors, use_rep="X_pca")

        # Initialize CellMapper in self-mapping mode
        cm = CellMapper(adata_spatial)

        # Load precomputed distances
        cm.load_precomputed_distances(distances_key="distances")

        # Verify the neighbors were properly loaded. Note that scanpy will return values for n_neighbors-1 neighbors.
        assert cm.knn is not None
        assert cm.knn.xx.n_neighbors + 1 == n_neighbors

        # Test the full pipeline with precomputed distances
        cm.compute_mapping_matrix(kernel_method="gauss")
        cm.map_obs(key="leiden")

        assert "leiden_pred" in cm.query.obs
        assert "leiden_conf" in cm.query.obs

    @pytest.mark.parametrize(
        "squidpy_params",
        [
            # Test basic KNN approach
            {
                "n_neighs": 10,
                "library_key": None,
            },
            # Test with library_key
            {"n_neighs": 8, "library_key": "batch"},
            # Test Delaunay triangulation
            {
                "delaunay": True,
                "library_key": None,
            },
            # Test radius with set_diag=True
            {"radius": 10.0, "set_diag": True, "library_key": "batch", "coord_type": "generic"},
            # Test percentile with library_key
            {"percentile": 99.0, "library_key": "batch"},
        ],
    )
    def test_load_squidpy_distances(self, adata_spatial, squidpy_params):
        """Test loading distances computed with squidpy.gr.spatial_neighbors with various configurations."""
        # Skip test if squidpy is not installed
        pytest.importorskip("squidpy")
        import squidpy as sq

        # Compute spatial neighbors with squidpy using the provided parameters
        sq.gr.spatial_neighbors(adata_spatial, spatial_key="spatial", **squidpy_params)

        # Initialize CellMapper in self-mapping mode
        cm = CellMapper(adata_spatial)

        print(adata_spatial)

        # Load precomputed distances
        cm.load_precomputed_distances(distances_key="spatial_distances")

        # Verify the neighbors were properly loaded
        assert cm.knn is not None

        # Additional checks based on specific parameters
        if "delaunay" in squidpy_params and squidpy_params["delaunay"]:
            # Delaunay triangulation typically has more connections
            assert cm.knn.xx.n_neighbors >= 3, "Delaunay should create at least a few connections per cell"

        if "set_diag" in squidpy_params and squidpy_params["set_diag"]:
            assert (
                adata_spatial.obsp["spatial_connectivities"].diagonal()
                == cm.knn.xx.boolean_adjacency(self_edges=True).diagonal()
            ).all()

        # Test the mapping pipeline
        cm.compute_mapping_matrix(kernel_method="gauss")
        cm.map_obs(key="leiden")

        assert "leiden_pred" in cm.query.obs
        assert "leiden_conf" in cm.query.obs

    def test_load_distances_behavior(self, adata_spatial):
        """Test loading precomputed distances and verifying neighbor behavior."""

        # Compute neighbors with scanpy
        sc.pp.neighbors(adata_spatial, n_neighbors=10, use_rep="X_pca")

        # Initialize CellMapper in self-mapping mode
        cm = CellMapper(adata_spatial)

        # Load precomputed distances
        cm.load_precomputed_distances(distances_key="distances")

        # Verify that neighbors were loaded
        assert cm.knn is not None

        # With the new behavior, self-edges are removed from stored arrays
        # but the original n_neighbors from the distance matrix is preserved
        for i in range(min(10, cm.knn.xx.n_samples)):  # Check first 10 cells
            assert i not in cm.knn.xx.indices[i], "Self-edges should be removed from stored arrays"

        # Test with self_edges=True in adjacency methods
        adjacency_with_self = cm.knn.xx.boolean_adjacency(self_edges=True)
        adjacency_without_self = cm.knn.xx.boolean_adjacency(self_edges=False)

        # With self_edges=True, diagonal should be True
        assert adjacency_with_self.diagonal().all(), "Diagonal should be True with self_edges=True"

        # With self_edges=False, diagonal should be False
        assert not adjacency_without_self.diagonal().any(), "Diagonal should be False with self_edges=False"

        # Test the mapping pipeline
        cm.compute_mapping_matrix(kernel_method="gauss")
        cm.map_obs(key="leiden")

        assert "leiden_pred" in cm.query.obs
        assert "leiden_conf" in cm.query.obs

    def test_self_mapping_without_rep(self, adata_pbmc3k):
        """Test self-mapping when use_rep=None, testing automatic PCA computation."""
        # Initialize with self-mapping
        cm = CellMapper(adata_pbmc3k)

        # Test with no representation provided
        cm.compute_neighbors(n_neighbors=5, use_rep=None, n_comps=10)
        cm.compute_mapping_matrix(kernel_method="gauss")

        # Verify joint PCA was computed
        assert "X_pca" in adata_pbmc3k.obsm
        assert adata_pbmc3k.obsm["X_pca"].shape[1] == 10

        # Test rest of pipeline
        cm.map_obs(key="leiden")
        assert "leiden_pred" in cm.query.obs

    @pytest.mark.parametrize(
        "kernel_method,self_edges,knn_method",
        [
            ("gauss", False, "sklearn"),
            ("gauss", False, "pynndescent"),
            ("umap", True, "sklearn"),
            ("umap", True, "pynndescent"),
        ],
    )
    def test_map_obs_pseudotime_self_mapping(self, adata_pbmc3k, kernel_method, self_edges, knn_method):
        """Test mapping pseudotime values in self-mapping mode - should have high correlation."""

        # Create CellMapper and compute mapping matrix
        cmap = CellMapper(adata_pbmc3k)
        cmap.compute_neighbors(n_neighbors=30, use_rep="X_pca", knn_method=knn_method)
        cmap.compute_mapping_matrix(kernel_method=kernel_method, self_edges=self_edges)

        # check that there are no self-edges in the connectivities.
        # essentially, this makes sure we're not using each cells own value to predict itself
        assert cmap.mapping_operator.matrix is not None, "Mapping matrix should be computed"
        assert (cmap.mapping_operator.matrix.diagonal() == 0).all(), "Self edges should be removed from mapping matrix"

        # Map pseudotime
        cmap.map_obs(key="dpt_pseudotime")

        # Check that pseudotime was mapped
        assert "dpt_pseudotime_pred" in cmap.query.obs

        # Check correlation between actual and predicted pseudotime values
        query_original_pt = adata_pbmc3k.obs["dpt_pseudotime"]
        query_predicted_pt = adata_pbmc3k.obs["dpt_pseudotime_pred"]

        correlation, _ = pearsonr(query_original_pt, query_predicted_pt)

        # Cross-mapping should still have reasonably high correlation, though lower than self-mapping
        assert correlation > 0.99, f"Cross-mapping pseudotime correlation too low: {correlation}"


class TestCellMapperImputation:
    """Test CellMapper imputation functionality with iterative and spectral approaches."""

    def test_symmetric_eigendecomposition_equivalence(self, adata_pbmc3k):
        """
        Test that symmetric eigendecomposition approach gives equivalent results
        to direct eigendecomposition of the asymmetric matrix.
        """

        # Create CellMapper and compute mapping matrix
        cmap = CellMapper(adata_pbmc3k)
        cmap.compute_neighbors(only_yx=True, use_rep="X_pca")
        cmap.compute_mapping_matrix(eigen_solver="partial", n_eigenvectors=100)

        # Get eigendecomposition from CellMapper (symmetric approach)
        mop = cmap.mapping_operator
        eigenvalues_cmap, eigenvectors_cmap = mop._eigendecomposition

        # Compute eigendecomposition directly on the non-symmetric matrix
        eigenvalues, eigenvectors = eigs(
            mop.matrix,
            k=mop.n_eigenvectors,
            which="LM",  # Largest magnitude
            return_eigenvectors=True,
        )

        # Sort by eigenvalue magnitude (descending) for proper diffusion ordering
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Check for complex eigenvalues and convert to real
        if np.iscomplexobj(eigenvalues) and not np.allclose(np.imag(eigenvalues), 0):
            pytest.skip("Complex eigenvalues detected, skipping test")

        # Convert to real arrays (safe since we checked for complex values)
        eigenvalues_real = np.real(eigenvalues) if np.iscomplexobj(eigenvalues) else eigenvalues
        eigenvectors_real = np.real(eigenvectors) if np.iscomplexobj(eigenvectors) else eigenvectors

        # Test eigenvalues are equivalent
        assert np.allclose(eigenvalues_real, eigenvalues_cmap, atol=1e-5), "Eigenvalues differ"

        # Test eigenvectors span the same subspace (using subspace angles)
        assert np.allclose(subspace_angles(eigenvectors_real, eigenvectors_cmap), 0, atol=1e-3), "Eigenvectors differ"

    def test_iterative_vs_complete_spectral_similarity(self, adata_pbmc3k):
        """Test that iterative and complete spectral approaches give highly similar results."""

        # Use a smaller subset for complete eigendecomposition (for speed)
        bdata = adata_pbmc3k[:1000].copy()

        # compute highly variable genes on the subset and filter to those
        sc.pp.highly_variable_genes(bdata, subset=True)

        # Create CellMapper and compute mapping matrix with complete eigendecomposition
        cmap = CellMapper(bdata)
        cmap.compute_neighbors(only_yx=True, use_rep="X_pca")
        cmap.compute_mapping_matrix(eigen_solver="complete")

        # Get spectral imputation
        cmap.map_layers("X", t=10, diffusion_method="spectral")
        imputed_spectral = cmap.query_imputed.X  # returns dense matrix

        # Get iterative imputation
        cmap.map_layers("X", t=10, diffusion_method="iterative")
        imputed_iterative = cmap.query_imputed.X  # returns sparse matrix

        # Compare the two imputed gene expression matrices
        cell_corrs, gene_corrs = analyze_by_cell_and_gene(imputed_spectral, imputed_iterative.toarray())

        # Assert high similarity
        assert cell_corrs.mean() > 0.99, f"Cell correlations not high enough: {cell_corrs.mean():.6f}"
        assert gene_corrs.mean() > 0.95, f"Gene correlations not high enough: {gene_corrs.mean():.6f}"

    def test_iterative_vs_partial_spectral_similarity(self, adata_pbmc3k):
        """Test that iterative and partial spectral approaches give highly similar results."""

        # Filter to highly variable genes
        adata = adata_pbmc3k[:, adata_pbmc3k.var["highly_variable"]].copy()

        # Create CellMapper and compute mapping matrix with partial eigendecomposition
        cmap = CellMapper(adata)
        cmap.compute_neighbors(only_yx=True, use_rep="X_pca")
        cmap.compute_mapping_matrix(eigen_solver="partial", n_eigenvectors=100)

        # Get spectral imputation
        cmap.map_layers("X", t=10, diffusion_method="spectral")
        imputed_spectral = cmap.query_imputed.X

        # Get iterative imputation
        cmap.map_layers("X", t=10, diffusion_method="iterative")
        imputed_iterative = cmap.query_imputed.X

        # Compare the two imputed gene expression matrices
        cell_corrs, gene_corrs = analyze_by_cell_and_gene(imputed_spectral, imputed_iterative.toarray())

        # Assert high similarity
        assert cell_corrs.mean() > 0.99, f"Cell correlations not high enough: {cell_corrs.mean():.6f}"
        assert gene_corrs.mean() > 0.95, f"Gene correlations not high enough: {gene_corrs.mean():.6f}"
