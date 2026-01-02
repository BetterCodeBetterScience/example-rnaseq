"""Unit tests for clustering module."""

import numpy as np
import pandas as pd
import pytest

from example_rnaseq.clustering import (
    compute_cluster_celltype_overlap,
    plot_clusters,
    run_leiden_clustering,
)
from example_rnaseq.dimensionality_reduction import (
    compute_neighbors,
    compute_umap,
    run_harmony_integration,
)
from example_rnaseq.preprocessing import (
    log_transform,
    normalize_counts,
    run_pca,
    select_highly_variable_genes,
)


@pytest.fixture
def adata_with_neighbors(minimal_adata):
    """Create AnnData with neighbor graph computed."""
    adata = minimal_adata.copy()
    adata.layers["counts"] = adata.X.copy()

    # Preprocess
    normalize_counts(adata)
    log_transform(adata)
    select_highly_variable_genes(adata, n_top_genes=20)
    run_pca(adata)

    # Add total_counts
    adata.obs["total_counts"] = np.array(adata.X.sum(axis=1)).flatten()

    # Harmony and neighbors
    adata, use_rep = run_harmony_integration(adata)
    compute_neighbors(adata, n_neighbors=10, n_pcs=10, use_rep=use_rep)

    return adata


@pytest.fixture
def adata_with_umap(adata_with_neighbors):
    """Create AnnData with UMAP computed."""
    adata = adata_with_neighbors.copy()
    compute_umap(adata, init_pos="spectral")
    return adata


class TestRunLeidenClustering:
    """Tests for Leiden clustering."""

    def test_creates_cluster_assignments(self, adata_with_neighbors):
        """Test that cluster assignments are created."""
        adata = adata_with_neighbors.copy()
        run_leiden_clustering(adata, resolution=0.5, key_added="test_leiden")

        assert "test_leiden" in adata.obs.columns
        assert adata.obs["test_leiden"].notna().all()

    def test_all_cells_assigned(self, adata_with_neighbors):
        """Test that all cells get a cluster assignment."""
        adata = adata_with_neighbors.copy()
        run_leiden_clustering(adata, resolution=0.5, key_added="clusters")

        assert len(adata.obs["clusters"]) == adata.n_obs

    def test_resolution_affects_cluster_count(self, adata_with_neighbors):
        """Test that higher resolution creates more clusters."""
        adata_low = adata_with_neighbors.copy()
        adata_high = adata_with_neighbors.copy()

        run_leiden_clustering(adata_low, resolution=0.1, key_added="clusters")
        run_leiden_clustering(adata_high, resolution=2.0, key_added="clusters")

        n_clusters_low = adata_low.obs["clusters"].nunique()
        n_clusters_high = adata_high.obs["clusters"].nunique()

        # Higher resolution should create more clusters (usually)
        assert n_clusters_high >= n_clusters_low

    def test_default_key_name(self, adata_with_neighbors):
        """Test default key naming."""
        adata = adata_with_neighbors.copy()
        run_leiden_clustering(adata)

        assert "leiden_1.0" in adata.obs.columns


class TestComputeClusterCelltypeOverlap:
    """Tests for cluster-celltype contingency table."""

    def test_returns_dataframe(self, adata_with_neighbors):
        """Test that result is a DataFrame."""
        adata = adata_with_neighbors.copy()
        run_leiden_clustering(adata, key_added="leiden")

        contingency = compute_cluster_celltype_overlap(adata, cluster_key="leiden")

        assert isinstance(contingency, pd.DataFrame)

    def test_contingency_sums_to_total(self, adata_with_neighbors):
        """Test that contingency table sums to total cells."""
        adata = adata_with_neighbors.copy()
        run_leiden_clustering(adata, key_added="leiden")

        contingency = compute_cluster_celltype_overlap(adata, cluster_key="leiden")

        assert contingency.values.sum() == adata.n_obs

    def test_row_index_is_clusters(self, adata_with_neighbors):
        """Test that rows are cluster IDs."""
        adata = adata_with_neighbors.copy()
        run_leiden_clustering(adata, key_added="leiden")

        contingency = compute_cluster_celltype_overlap(adata, cluster_key="leiden")

        assert set(contingency.index) == set(adata.obs["leiden"].unique())

    def test_column_index_is_cell_types(self, adata_with_neighbors):
        """Test that columns are cell types."""
        adata = adata_with_neighbors.copy()
        run_leiden_clustering(adata, key_added="leiden")

        contingency = compute_cluster_celltype_overlap(adata, cluster_key="leiden")

        assert set(contingency.columns) == set(adata.obs["cell_type"].unique())


class TestPlotClusters:
    """Tests for cluster plotting (smoke tests)."""

    def test_plot_clusters_no_error(self, adata_with_umap, temp_output_dir):
        """Test that cluster plot runs without error."""
        adata = adata_with_umap.copy()
        run_leiden_clustering(adata, key_added="leiden")

        plot_clusters(adata, cluster_key="leiden", figure_dir=temp_output_dir)
        assert (temp_output_dir / "umap_cell_type_leiden.png").exists()
