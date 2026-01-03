"""Unit tests for pathway analysis module."""

import numpy as np
import pandas as pd
import pytest

from conftest import is_sorted
from example_rnaseq.pathway_analysis import (
    get_gsea_top_terms,
    plot_gsea_results,
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
        assert is_sorted(values, descending=True)

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

    def test_run_gsea_prerank(self, sample_deseq_results_real_genes):
        """Test running GSEA prerank with real gene symbols."""
        from example_rnaseq.pathway_analysis import run_gsea_prerank

        rank_df = prepare_ranked_list(sample_deseq_results_real_genes)

        # This requires network access
        try:
            prerank_res = run_gsea_prerank(
                rank_df,
                gene_sets=["MSigDB_Hallmark_2020"],
                permutation_num=10,  # Minimal for speed
                min_size=5,  # Lower threshold for test data
            )
            assert prerank_res is not None
            assert hasattr(prerank_res, "res2d")
            assert len(prerank_res.res2d) > 0
        except ConnectionError as e:
            pytest.skip(f"GSEA prerank failed due to network issue: {e}")
        except TimeoutError as e:
            pytest.skip(f"GSEA prerank timed out: {e}")


class TestGetGseaTopTerms:
    """Tests for GSEA top terms extraction."""

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

    def test_returns_top_terms(self, mock_gsea_result, capsys):
        """Test that top and bottom terms are returned."""
        top_up, top_down = get_gsea_top_terms(mock_gsea_result, n_top=5)

        assert len(top_up) == 5
        assert len(top_down) == 5

    def test_top_terms_are_sorted(self, mock_gsea_result, capsys):
        """Test that terms are sorted by NES."""
        top_up, top_down = get_gsea_top_terms(mock_gsea_result, n_top=5)

        # Top up should have highest NES
        assert top_up["NES"].max() == mock_gsea_result.res2d["NES"].max()
        # Top down should have lowest NES
        assert top_down["NES"].min() == mock_gsea_result.res2d["NES"].min()

    def test_prints_output(self, mock_gsea_result, capsys):
        """Test that function prints to stdout."""
        get_gsea_top_terms(mock_gsea_result, n_top=3)

        captured = capsys.readouterr()
        assert "Top Upregulated Pathways" in captured.out
        assert "Top Downregulated Pathways" in captured.out


class TestPlotGseaResults:
    """Tests for GSEA plot functions (smoke tests)."""

    @pytest.fixture
    def sample_gsea_combined(self):
        """Create sample combined GSEA data for plotting."""
        return pd.DataFrame({
            "Term": [f"Pathway_{i}" for i in range(10)],
            "NES": np.linspace(2, -2, 10),
            "FDR q-val": np.random.uniform(0.001, 0.25, 10),
            "Direction": ["Upregulated"] * 5 + ["Downregulated"] * 5,
            "log_FDR": np.random.uniform(0.5, 3, 10),
            "Count": list(range(10, 20)),
        })

    def test_plot_gsea_results_no_error(self, sample_gsea_combined, temp_output_dir):
        """Test that plot_gsea_results runs without error."""
        plot_gsea_results(sample_gsea_combined, temp_output_dir)
        assert (temp_output_dir / "gsea_pathways.png").exists()

    def test_plot_gsea_results_no_save(self, sample_gsea_combined):
        """Test that plot_gsea_results runs without saving."""
        # Should not raise even without figure_dir
        plot_gsea_results(sample_gsea_combined, figure_dir=None)
