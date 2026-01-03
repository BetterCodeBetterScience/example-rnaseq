"""Unit tests for dimensionality reduction module."""

import numpy as np
import pytest

from example_rnaseq.dimensionality_reduction import (
    compute_neighbors,
    compute_umap,
    plot_pca_qc,
    plot_umap_qc,
    run_harmony_integration,
)
from example_rnaseq.preprocessing import (
    log_transform,
    normalize_counts,
    run_pca,
    select_highly_variable_genes,
)


@pytest.fixture
def adata_with_pca(minimal_adata):
    """Create AnnData with PCA computed."""
    adata = minimal_adata.copy()
    adata.layers["counts"] = adata.X.copy()

    # Preprocess
    normalize_counts(adata)
    log_transform(adata)
    select_highly_variable_genes(adata, n_top_genes=20)
    run_pca(adata)

    # Add total_counts for plotting
    adata.obs["total_counts"] = np.array(adata.X.sum(axis=1)).flatten()

    return adata


class TestRunHarmonyIntegration:
    """Tests for Harmony batch correction."""

    @pytest.fixture
    def harmonypy_available(self):
        """Check if harmonypy is available."""
        try:
            import harmonypy
            return True
        except ImportError:
            return False

    def test_runs_without_error(self, adata_with_pca, harmonypy_available):
        """Test that Harmony runs and produces corrected coordinates."""
        if not harmonypy_available:
            pytest.skip("harmonypy not installed")

        original_pca = adata_with_pca.obsm["X_pca"].copy()
        adata, use_rep = run_harmony_integration(adata_with_pca.copy())

        # When harmony is available, it should return harmony coordinates
        assert use_rep == "X_pca_harmony"
        assert "X_pca_harmony" in adata.obsm

        # The harmony output should be different from the original PCA
        # (unless there's only one batch, but test data has multiple donors)
        harmony_coords = adata.obsm["X_pca_harmony"]
        assert not np.allclose(harmony_coords, original_pca), (
            "Harmony output should differ from original PCA"
        )

    def test_creates_harmony_embedding(self, adata_with_pca, harmonypy_available):
        """Test that Harmony creates a new embedding with correct shape."""
        if not harmonypy_available:
            pytest.skip("harmonypy not installed")

        adata, use_rep = run_harmony_integration(adata_with_pca.copy())

        assert use_rep == "X_pca_harmony"
        assert "X_pca_harmony" in adata.obsm
        assert adata.obsm["X_pca_harmony"].shape[0] == adata.n_obs
        assert adata.obsm["X_pca_harmony"].shape == adata.obsm["X_pca"].shape

    def test_preserves_original_pca(self, adata_with_pca, harmonypy_available):
        """Test that original PCA is preserved after Harmony."""
        if not harmonypy_available:
            pytest.skip("harmonypy not installed")

        original_pca = adata_with_pca.obsm["X_pca"].copy()
        adata, _ = run_harmony_integration(adata_with_pca.copy())

        np.testing.assert_array_equal(adata.obsm["X_pca"], original_pca)

    def test_fallback_without_harmony(self, adata_with_pca, monkeypatch):
        """Test that function falls back to PCA when harmony is not available."""
        # Mock the import to simulate harmonypy not being installed
        import scanpy.external as sce

        def mock_harmony(*args, **kwargs):
            raise ImportError("harmonypy not installed")

        monkeypatch.setattr(sce.pp, "harmony_integrate", mock_harmony)

        adata, use_rep = run_harmony_integration(adata_with_pca.copy())

        # Should fall back to X_pca
        assert use_rep == "X_pca"


class TestComputeNeighbors:
    """Tests for neighbor computation."""

    def test_creates_neighbor_graph(self, adata_with_pca):
        """Test that neighbor graph is created."""
        adata = adata_with_pca.copy()
        adata, use_rep = run_harmony_integration(adata)

        compute_neighbors(adata, n_neighbors=10, n_pcs=10, use_rep=use_rep)

        assert "neighbors" in adata.uns
        assert "connectivities" in adata.obsp
        assert "distances" in adata.obsp

    def test_respects_n_neighbors(self, adata_with_pca):
        """Test that n_neighbors parameter is used."""
        adata = adata_with_pca.copy()
        adata, use_rep = run_harmony_integration(adata)

        compute_neighbors(adata, n_neighbors=5, n_pcs=10, use_rep=use_rep)

        # The number of neighbors should be stored
        assert adata.uns["neighbors"]["params"]["n_neighbors"] == 5


class TestComputeUmap:
    """Tests for UMAP computation."""

    def test_creates_umap_embedding(self, adata_with_pca):
        """Test that UMAP embedding is created."""
        adata = adata_with_pca.copy()
        adata, use_rep = run_harmony_integration(adata)
        compute_neighbors(adata, n_neighbors=10, n_pcs=10, use_rep=use_rep)

        compute_umap(adata, init_pos="spectral")

        assert "X_umap" in adata.obsm
        assert adata.obsm["X_umap"].shape == (adata.n_obs, 2)

    def test_umap_values_are_finite(self, adata_with_pca):
        """Test that UMAP values are finite."""
        adata = adata_with_pca.copy()
        adata, use_rep = run_harmony_integration(adata)
        compute_neighbors(adata, n_neighbors=10, n_pcs=10, use_rep=use_rep)
        compute_umap(adata, init_pos="spectral")

        assert np.all(np.isfinite(adata.obsm["X_umap"]))


class TestPlotFunctions:
    """Tests for plotting functions (smoke tests)."""

    def test_plot_pca_qc_no_error(self, adata_with_pca, temp_output_dir):
        """Test that PCA QC plot runs without error."""
        plot_pca_qc(adata_with_pca, temp_output_dir)
        assert (temp_output_dir / "pca_cell_type.png").exists()

    def test_plot_umap_qc_no_error(self, adata_with_pca, temp_output_dir):
        """Test that UMAP QC plot runs without error."""
        adata = adata_with_pca.copy()
        adata, use_rep = run_harmony_integration(adata)
        compute_neighbors(adata, n_neighbors=10, n_pcs=10, use_rep=use_rep)
        compute_umap(adata, init_pos="spectral")

        plot_umap_qc(adata, temp_output_dir)
        assert (temp_output_dir / "umap_total_counts.png").exists()
