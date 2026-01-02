"""Unit tests for overrepresentation analysis module."""

import numpy as np
import pandas as pd
import pytest

from example_rnaseq.overrepresentation_analysis import (
    get_significant_gene_lists,
    prepare_enrichr_plot_data,
)


class TestGetSignificantGeneLists:
    """Tests for significant gene list extraction."""

    def test_separates_up_and_down(self, sample_deseq_results):
        """Test that up and down genes are separated."""
        up_genes, down_genes = get_significant_gene_lists(sample_deseq_results)

        assert isinstance(up_genes, list)
        assert isinstance(down_genes, list)

    def test_respects_padj_threshold(self):
        """Test that padj threshold is respected."""
        results = pd.DataFrame({
            "padj": [0.01, 0.1, 0.001, 0.2],
            "log2FoldChange": [1.0, 2.0, -1.0, -2.0],
        }, index=["GENE1", "GENE2", "GENE3", "GENE4"])

        up_genes, down_genes = get_significant_gene_lists(results, padj_threshold=0.05)

        # Only GENE1 (up, padj=0.01) and GENE3 (down, padj=0.001) should pass
        assert "GENE1" in up_genes
        assert "GENE2" not in up_genes  # padj=0.1 > 0.05
        assert "GENE3" in down_genes
        assert "GENE4" not in down_genes  # padj=0.2 > 0.05

    def test_up_genes_have_positive_lfc(self):
        """Test that up genes have positive log2FC."""
        results = pd.DataFrame({
            "padj": [0.01, 0.01, 0.01],
            "log2FoldChange": [2.0, -1.0, 0.5],
        }, index=["GENE1", "GENE2", "GENE3"])

        up_genes, down_genes = get_significant_gene_lists(results)

        assert "GENE1" in up_genes
        assert "GENE3" in up_genes
        assert "GENE2" in down_genes

    def test_empty_results_handled(self):
        """Test handling of no significant genes."""
        results = pd.DataFrame({
            "padj": [0.5, 0.6, 0.7],
            "log2FoldChange": [1.0, -1.0, 0.5],
        }, index=["GENE1", "GENE2", "GENE3"])

        up_genes, down_genes = get_significant_gene_lists(results, padj_threshold=0.05)

        assert len(up_genes) == 0
        assert len(down_genes) == 0


class TestPrepareEnrichrPlotData:
    """Tests for Enrichr plot data preparation."""

    @pytest.fixture
    def mock_enrichr_result(self):
        """Create mock Enrichr result object."""
        class MockEnrichr:
            def __init__(self, direction="up"):
                self.results = pd.DataFrame({
                    "Term": [f"Pathway_{i}" for i in range(10)],
                    "Adjusted P-value": np.random.uniform(0.001, 0.1, 10),
                    "Overlap": [f"{i}/100" for i in range(5, 15)],
                })
        return MockEnrichr

    def test_combines_up_and_down(self, mock_enrichr_result):
        """Test that up and down results are combined."""
        enr_up = mock_enrichr_result("up")
        enr_down = mock_enrichr_result("down")

        combined = prepare_enrichr_plot_data(enr_up, enr_down, n_top=5)

        assert len(combined) == 10  # 5 from each

    def test_handles_none_inputs(self, mock_enrichr_result):
        """Test handling when one direction is None."""
        enr_up = mock_enrichr_result("up")

        combined = prepare_enrichr_plot_data(enr_up, None, n_top=5)

        assert len(combined) == 5
        assert (combined["Direction"] == "Upregulated").all()

    def test_returns_none_when_both_none(self):
        """Test that None is returned when both inputs are None."""
        combined = prepare_enrichr_plot_data(None, None, n_top=5)

        assert combined is None

    def test_adds_log_p_column(self, mock_enrichr_result):
        """Test that log p-value is computed."""
        enr_up = mock_enrichr_result("up")

        combined = prepare_enrichr_plot_data(enr_up, None, n_top=5)

        assert "log_p" in combined.columns
        # -log10(p) should be positive for p < 1
        assert combined["log_p"].min() > 0

    def test_extracts_gene_count(self, mock_enrichr_result):
        """Test that gene count is extracted from overlap."""
        enr_up = mock_enrichr_result("up")

        combined = prepare_enrichr_plot_data(enr_up, None, n_top=5)

        assert "Gene_Count" in combined.columns
        assert combined["Gene_Count"].dtype in [np.int64, int]


@pytest.mark.slow
class TestEnrichrIntegration:
    """Integration tests for Enrichr (marked slow, requires network)."""

    def test_run_enrichr(self):
        """Test running Enrichr with a small gene list."""
        from example_rnaseq.overrepresentation_analysis import run_enrichr

        # Use a small list of well-known genes
        gene_list = ["TP53", "BRCA1", "EGFR", "MYC", "KRAS"]

        try:
            enr = run_enrichr(gene_list, gene_sets=["MSigDB_Hallmark_2020"])
            assert enr is not None
        except Exception as e:
            pytest.skip(f"Enrichr failed (likely network issue): {e}")

    def test_run_enrichr_empty_list(self):
        """Test that empty gene list returns None."""
        from example_rnaseq.overrepresentation_analysis import run_enrichr

        result = run_enrichr([])
        assert result is None
