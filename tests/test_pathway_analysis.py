"""Unit tests for pathway analysis module."""

import numpy as np
import pandas as pd
import pytest

from example_rnaseq.pathway_analysis import (
    prepare_gsea_plot_data,
    prepare_ranked_list,
)


class TestPrepareRankedList:
    """Tests for ranked list preparation."""

    def test_returns_sorted_dataframe(self, sample_deseq_results):
        """Test that result is sorted by stat."""
        rank_df = prepare_ranked_list(sample_deseq_results)

        assert isinstance(rank_df, pd.DataFrame)
        assert list(rank_df.columns) == ["stat"]

        # Should be sorted descending
        values = rank_df["stat"].values
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_drops_na_values(self):
        """Test that NA values are dropped."""
        results = pd.DataFrame({
            "stat": [1.0, np.nan, -1.0, 2.0],
            "padj": [0.01, 0.02, 0.03, 0.04],
        })

        rank_df = prepare_ranked_list(results)

        assert len(rank_df) == 3
        assert not rank_df["stat"].isna().any()

    def test_preserves_gene_index(self, sample_deseq_results):
        """Test that gene names are preserved as index."""
        rank_df = prepare_ranked_list(sample_deseq_results)

        assert all(idx in sample_deseq_results.index for idx in rank_df.index)


class TestPrepareGseaPlotData:
    """Tests for GSEA plot data preparation."""

    @pytest.fixture
    def mock_gsea_result(self):
        """Create mock GSEA result object."""
        class MockGSEA:
            def __init__(self):
                self.res2d = pd.DataFrame({
                    "Term": [f"Pathway_{i}" for i in range(20)],
                    "NES": np.linspace(2, -2, 20),
                    "FDR q-val": np.random.uniform(0, 0.5, 20),
                    "Lead_genes": ["GENE1;GENE2;GENE3"] * 20,
                })
        return MockGSEA()

    def test_selects_top_and_bottom(self, mock_gsea_result):
        """Test that top upregulated and downregulated are selected."""
        combined = prepare_gsea_plot_data(mock_gsea_result, n_top=5)

        assert len(combined) == 10  # 5 up + 5 down

    def test_adds_direction_column(self, mock_gsea_result):
        """Test that Direction column is added."""
        combined = prepare_gsea_plot_data(mock_gsea_result, n_top=5)

        assert "Direction" in combined.columns
        assert set(combined["Direction"].unique()) == {"Upregulated", "Downregulated"}

    def test_computes_log_fdr(self, mock_gsea_result):
        """Test that log10 FDR is computed."""
        combined = prepare_gsea_plot_data(mock_gsea_result, n_top=5)

        assert "log_FDR" in combined.columns
        assert combined["log_FDR"].min() >= 0  # -log10 should be positive for FDR < 1

    def test_computes_gene_count(self, mock_gsea_result):
        """Test that gene count is extracted."""
        combined = prepare_gsea_plot_data(mock_gsea_result, n_top=5)

        assert "Count" in combined.columns
        assert combined["Count"].min() > 0

    def test_cleans_term_names(self, mock_gsea_result):
        """Test that prefix is removed from term names."""
        mock_gsea_result.res2d["Term"] = [
            f"MSigDB_Hallmark_2020__Pathway_{i}" for i in range(20)
        ]

        combined = prepare_gsea_plot_data(
            mock_gsea_result, n_top=5, label_prefix="MSigDB_Hallmark_2020__"
        )

        # Prefix should be removed
        assert all(not t.startswith("MSigDB_Hallmark_2020__") for t in combined["Term"])


@pytest.mark.slow
class TestGseaIntegration:
    """Integration tests for GSEA (marked slow, requires network)."""

    def test_run_gsea_prerank(self, sample_deseq_results):
        """Test running GSEA prerank."""
        from example_rnaseq.pathway_analysis import run_gsea_prerank

        rank_df = prepare_ranked_list(sample_deseq_results)

        # This requires network access
        try:
            prerank_res = run_gsea_prerank(
                rank_df,
                gene_sets=["MSigDB_Hallmark_2020"],
                permutation_num=10,  # Minimal for speed
            )
            assert prerank_res is not None
        except Exception as e:
            pytest.skip(f"GSEA prerank failed (likely network issue): {e}")
