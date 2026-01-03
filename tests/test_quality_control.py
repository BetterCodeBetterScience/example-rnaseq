"""Unit tests for quality control module."""

import numpy as np
import pytest
import scipy.sparse as sp

from example_rnaseq.quality_control import (
    annotate_gene_types,
    apply_qc_filters,
    calculate_qc_metrics,
    filter_doublets,
    plot_doublets,
    plot_hemoglobin_distribution,
    plot_qc_metrics,
)


class TestAnnotateGeneTypes:
    """Tests for gene type annotation."""

    def test_annotates_mitochondrial_genes(self, minimal_adata):
        """Test mitochondrial gene detection."""
        adata = annotate_gene_types(minimal_adata.copy())

        assert "mt" in adata.var.columns
        mt_genes = adata.var[adata.var["mt"]]["feature_name"].tolist()
        assert "MT-CO1" in mt_genes
        assert "MT-CO2" in mt_genes
        assert "MT-ND1" in mt_genes

    def test_annotates_ribosomal_genes(self, minimal_adata):
        """Test ribosomal gene detection."""
        adata = annotate_gene_types(minimal_adata.copy())

        assert "ribo" in adata.var.columns
        ribo_genes = adata.var[adata.var["ribo"]]["feature_name"].tolist()
        assert "RPS10" in ribo_genes
        assert "RPL15" in ribo_genes

    def test_annotates_hemoglobin_genes(self, minimal_adata):
        """Test hemoglobin gene detection."""
        adata = annotate_gene_types(minimal_adata.copy())

        assert "hb" in adata.var.columns
        hb_genes = adata.var[adata.var["hb"]]["feature_name"].tolist()
        assert "HBA1" in hb_genes
        assert "HBB" in hb_genes

    def test_regular_genes_not_annotated(self, minimal_adata):
        """Test that regular genes are not marked as special."""
        adata = annotate_gene_types(minimal_adata.copy())

        gene0 = adata.var[adata.var["feature_name"] == "GENE0"]
        assert not gene0["mt"].values[0]
        assert not gene0["ribo"].values[0]
        assert not gene0["hb"].values[0]


class TestCalculateQcMetrics:
    """Tests for QC metric calculation."""

    def test_calculates_total_counts(self, minimal_adata):
        """Test total count calculation."""
        adata = annotate_gene_types(minimal_adata.copy())
        adata = calculate_qc_metrics(adata)

        assert "total_counts" in adata.obs.columns
        assert adata.obs["total_counts"].min() > 0

    def test_calculates_genes_by_counts(self, minimal_adata):
        """Test genes per cell calculation."""
        adata = annotate_gene_types(minimal_adata.copy())
        adata = calculate_qc_metrics(adata)

        assert "n_genes_by_counts" in adata.obs.columns
        assert adata.obs["n_genes_by_counts"].max() <= adata.n_vars

    def test_calculates_mt_percentage(self, minimal_adata):
        """Test mitochondrial percentage calculation."""
        adata = annotate_gene_types(minimal_adata.copy())
        adata = calculate_qc_metrics(adata)

        assert "pct_counts_mt" in adata.obs.columns
        assert adata.obs["pct_counts_mt"].min() >= 0
        assert adata.obs["pct_counts_mt"].max() <= 100

    def test_calculates_hb_percentage(self, minimal_adata):
        """Test hemoglobin percentage calculation."""
        adata = annotate_gene_types(minimal_adata.copy())
        adata = calculate_qc_metrics(adata)

        assert "pct_counts_hb" in adata.obs.columns


class TestApplyQcFilters:
    """Tests for QC filtering."""

    def test_filters_low_gene_cells(self, minimal_adata_with_qc):
        """Test filtering cells with too few genes."""
        adata = minimal_adata_with_qc.copy()
        # Use 10th percentile - should keep ~90% of cells
        min_genes = int(adata.obs["n_genes_by_counts"].quantile(0.1))
        expected_pass = (adata.obs["n_genes_by_counts"] > min_genes).sum()

        filtered = apply_qc_filters(adata, min_genes=min_genes, max_genes=10000,
                                     min_counts=0, max_counts=100000, max_hb_pct=100)

        # With 10th percentile threshold, we expect most cells to pass
        assert filtered.n_obs > 0, (
            f"Expected non-empty result: {expected_pass} cells should have "
            f"n_genes > {min_genes}"
        )
        assert filtered.n_obs <= adata.n_obs
        assert filtered.obs["n_genes_by_counts"].min() > min_genes

    def test_filters_high_gene_cells(self, minimal_adata_with_qc):
        """Test filtering cells with too many genes (doublets)."""
        adata = minimal_adata_with_qc.copy()
        # Use 90th percentile - should keep ~90% of cells
        max_genes = int(adata.obs["n_genes_by_counts"].quantile(0.9))
        expected_pass = (adata.obs["n_genes_by_counts"] < max_genes).sum()

        filtered = apply_qc_filters(adata, min_genes=0, max_genes=max_genes,
                                     min_counts=0, max_counts=100000, max_hb_pct=100)

        # With 90th percentile threshold, we expect most cells to pass
        assert filtered.n_obs > 0, (
            f"Expected non-empty result: {expected_pass} cells should have "
            f"n_genes < {max_genes}"
        )
        assert filtered.obs["n_genes_by_counts"].max() < max_genes

    def test_filters_high_hb_cells(self, minimal_adata_with_qc):
        """Test filtering cells with high hemoglobin content."""
        adata = minimal_adata_with_qc.copy()
        # Use 90th percentile - should keep ~90% of cells
        max_hb = float(adata.obs["pct_counts_hb"].quantile(0.9))
        expected_pass = (adata.obs["pct_counts_hb"] < max_hb).sum()

        filtered = apply_qc_filters(adata, min_genes=0, max_genes=10000,
                                     min_counts=0, max_counts=100000, max_hb_pct=max_hb)

        # With 90th percentile threshold, we expect most cells to pass
        assert filtered.n_obs > 0, (
            f"Expected non-empty result: {expected_pass} cells should have "
            f"pct_counts_hb < {max_hb}"
        )
        assert filtered.obs["pct_counts_hb"].max() < max_hb

    def test_returns_copy(self, minimal_adata_with_qc):
        """Test that filtering returns a copy."""
        adata = minimal_adata_with_qc.copy()
        filtered = apply_qc_filters(adata)

        # Original should be unchanged
        assert adata.n_obs == minimal_adata_with_qc.n_obs


class TestFilterDoublets:
    """Tests for doublet filtering."""

    def test_filters_predicted_doublets(self, minimal_adata):
        """Test that predicted doublets are removed."""
        adata = minimal_adata.copy()
        # Add doublet predictions
        adata.obs["predicted_doublet"] = [i % 10 == 0 for i in range(adata.n_obs)]

        n_doublets = adata.obs["predicted_doublet"].sum()
        filtered = filter_doublets(adata)

        assert filtered.n_obs == adata.n_obs - n_doublets
        assert filtered.obs["predicted_doublet"].sum() == 0


class TestPlotFunctions:
    """Tests for plotting functions (smoke tests)."""

    def test_plot_qc_metrics_no_error(self, minimal_adata_with_qc, temp_output_dir):
        """Test that plot_qc_metrics runs without error."""
        # Just verify it doesn't crash
        plot_qc_metrics(minimal_adata_with_qc, temp_output_dir)
        assert (temp_output_dir / "qc_violin_plots.png").exists()

    def test_plot_hemoglobin_no_error(self, minimal_adata_with_qc, temp_output_dir):
        """Test that plot_hemoglobin_distribution runs without error."""
        plot_hemoglobin_distribution(minimal_adata_with_qc, temp_output_dir)
        assert (temp_output_dir / "hemoglobin_distribution.png").exists()

    def test_plot_doublets_without_umap(self, minimal_adata):
        """Test plot_doublets handles missing UMAP gracefully."""
        adata = minimal_adata.copy()
        adata.obs["doublet_score"] = 0.1
        adata.obs["predicted_doublet"] = False
        # Should not raise even without X_umap
        plot_doublets(adata)

    def test_plot_doublets_with_umap(self, minimal_adata, temp_output_dir):
        """Test plot_doublets with UMAP coordinates."""
        adata = minimal_adata.copy()
        adata.obs["doublet_score"] = np.random.random(adata.n_obs)
        adata.obs["predicted_doublet"] = adata.obs["doublet_score"] > 0.5
        adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 2)

        plot_doublets(adata, temp_output_dir)
        assert (temp_output_dir / "doublet_detection_umap.png").exists()
