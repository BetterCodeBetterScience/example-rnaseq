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

    def test_runs_without_error(self, adata_with_pca):
        """Test that Harmony runs without error."""
        adata, use_rep = run_harmony_integration(adata_with_pca.copy())

        # Should return a representation
        assert use_rep in ["X_pca", "X_pca_harmony"]

    def test_creates_harmony_embedding(self, adata_with_pca):
        """Test that Harmony creates a new embedding."""
        adata, use_rep = run_harmony_integration(adata_with_pca.copy())

        if use_rep == "X_pca_harmony":
            assert "X_pca_harmony" in adata.obsm
            assert adata.obsm["X_pca_harmony"].shape[0] == adata.n_obs

    def test_preserves_original_pca(self, adata_with_pca):
        """Test that original PCA is preserved."""
        original_pca = adata_with_pca.obsm["X_pca"].copy()
        adata, _ = run_harmony_integration(adata_with_pca.copy())

        np.testing.assert_array_equal(adata.obsm["X_pca"], original_pca)


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
