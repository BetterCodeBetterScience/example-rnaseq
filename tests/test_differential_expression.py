"""Unit tests for differential expression module."""

import numpy as np
import pandas as pd
import pytest

from conftest import is_sorted
from example_rnaseq.differential_expression import (
    add_age_from_development_stage,
    get_significant_genes,
    prepare_deseq_inputs,
    subset_by_cell_type,
)


class TestAddAgeFromDevelopmentStage:
    """Tests for adding age column from development stage."""

    def test_adds_numeric_age(self, minimal_pseudobulk_adata):
        """Test that numeric age column is added."""
        adata = add_age_from_development_stage(minimal_pseudobulk_adata.copy())

        assert "age" in adata.obs.columns
        assert adata.obs["age"].notna().all()

    def test_handles_expected_format(self, minimal_pseudobulk_adata):
        """Test parsing of expected age format."""
        adata = minimal_pseudobulk_adata.copy()
        adata.obs["development_stage"] = "45-year-old human stage"

        adata = add_age_from_development_stage(adata)

        assert (adata.obs["age"] == 45).all()


class TestPrepareDeseqInputs:
    """Tests for DESeq2 input preparation."""

    def test_returns_counts_and_metadata(self, minimal_pseudobulk_adata):
        """Test that counts and metadata are returned."""
        adata = minimal_pseudobulk_adata.copy()
        adata = add_age_from_development_stage(adata)

        counts_df, metadata = prepare_deseq_inputs(adata)

        assert isinstance(counts_df, pd.DataFrame)
        assert isinstance(metadata, pd.DataFrame)
        assert len(counts_df) == adata.n_obs
        assert len(metadata) == adata.n_obs

    def test_counts_have_correct_shape(self, minimal_pseudobulk_adata):
        """Test counts DataFrame shape."""
        adata = minimal_pseudobulk_adata.copy()
        adata = add_age_from_development_stage(adata)

        counts_df, _ = prepare_deseq_inputs(adata)

        assert counts_df.shape[0] == adata.n_obs
        assert counts_df.shape[1] <= adata.n_vars  # May have duplicates removed

    def test_scales_age(self, minimal_pseudobulk_adata):
        """Test that age is scaled."""
        adata = minimal_pseudobulk_adata.copy()
        adata = add_age_from_development_stage(adata)

        _, metadata = prepare_deseq_inputs(adata)

        assert "age_scaled" in metadata.columns
        # Scaled age should have mean ~0
        assert abs(metadata["age_scaled"].mean()) < 0.5


class TestSubsetByCellType:
    """Tests for cell type subsetting."""

    def test_subsets_to_cell_type(self, minimal_pseudobulk_adata):
        """Test that data is subsetted to specified cell type."""
        adata = minimal_pseudobulk_adata.copy()
        adata = add_age_from_development_stage(adata)
        counts_df, metadata = prepare_deseq_inputs(adata)

        cell_type = adata.obs["cell_type"].unique()[0]
        pb_ct, counts_ct, meta_ct = subset_by_cell_type(
            adata, counts_df, metadata, cell_type
        )

        assert (pb_ct.obs["cell_type"] == cell_type).all()
        assert len(counts_ct) == pb_ct.n_obs
        assert len(meta_ct) == pb_ct.n_obs

    def test_maintains_index_consistency(self, minimal_pseudobulk_adata):
        """Test that indices are consistent across outputs."""
        adata = minimal_pseudobulk_adata.copy()
        adata = add_age_from_development_stage(adata)
        counts_df, metadata = prepare_deseq_inputs(adata)

        cell_type = adata.obs["cell_type"].unique()[0]
        pb_ct, counts_ct, meta_ct = subset_by_cell_type(
            adata, counts_df, metadata, cell_type
        )

        assert list(pb_ct.obs_names) == list(counts_ct.index)
        assert list(pb_ct.obs_names) == list(meta_ct.index)


class TestGetSignificantGenes:
    """Tests for significant gene extraction."""

    def test_filters_by_padj(self, sample_deseq_results):
        """Test filtering by adjusted p-value."""
        # Create a mock stat_res object
        class MockStatRes:
            def __init__(self, results):
                self.results_df = results

        mock_res = MockStatRes(sample_deseq_results)
        sig_genes = get_significant_genes(mock_res, padj_threshold=0.05)

        # All returned genes should have padj < threshold
        assert (sig_genes["padj"] < 0.05).all()

    def test_sorts_by_log2fc(self, sample_deseq_results):
        """Test that results are sorted by log2FC."""
        class MockStatRes:
            def __init__(self, results):
                self.results_df = results

        mock_res = MockStatRes(sample_deseq_results)
        sig_genes = get_significant_genes(mock_res, padj_threshold=1.0)

        # Should be sorted descending
        log2fc_values = sig_genes["log2FoldChange"].values
        assert is_sorted(log2fc_values, descending=True)

    def test_returns_dataframe(self, sample_deseq_results):
        """Test that result is a DataFrame."""
        class MockStatRes:
            def __init__(self, results):
                self.results_df = results

        mock_res = MockStatRes(sample_deseq_results)
        sig_genes = get_significant_genes(mock_res)

        assert isinstance(sig_genes, pd.DataFrame)


@pytest.mark.slow
class TestDeseq2Integration:
    """Integration tests for DESeq2 (marked slow)."""

    @pytest.mark.filterwarnings(
        "ignore:The dispersion trend curve fitting did not converge:UserWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:As the residual degrees of freedom is less than 3:UserWarning"
    )
    def test_run_deseq2_on_minimal_data(self, minimal_pseudobulk_adata):
        """Test running DESeq2 on minimal data.

        Note: The filterwarnings markers suppress expected warnings from pydeseq2
        that occur due to the small test dataset size (few samples, low degrees
        of freedom). These are not indicative of problems with the code.
        """
        from example_rnaseq.differential_expression import run_deseq2

        adata = minimal_pseudobulk_adata.copy()
        adata = add_age_from_development_stage(adata)
        counts_df, metadata = prepare_deseq_inputs(adata)

        # Subset to one cell type
        cell_type = adata.obs["cell_type"].unique()[0]
        _, counts_ct, meta_ct = subset_by_cell_type(
            adata, counts_df, metadata, cell_type
        )

        # Run DESeq2 (this is slow)
        dds = run_deseq2(counts_ct, meta_ct, ["age_scaled", "sex"], n_cpus=1)

        assert dds is not None
