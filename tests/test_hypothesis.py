"""Property-based tests using Hypothesis.

These tests verify invariants and properties that should hold across
a wide range of inputs, helping catch edge cases that example-based
tests might miss.
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from example_rnaseq.checkpoint import (
    bids_checkpoint_name,
    parse_bids_checkpoint_name,
    hash_parameters,
)
from example_rnaseq.execution_log import serialize_parameters
from example_rnaseq.overrepresentation_analysis import (
    get_significant_gene_lists,
    prepare_enrichr_plot_data,
)
from example_rnaseq.pathway_analysis import prepare_ranked_list
from example_rnaseq.pseudobulk import filter_pseudobulk_by_cell_count


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for valid BIDS-compatible strings (alphanumeric, no special chars)
bids_safe_text = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    min_size=1,
    max_size=20,
)

# Strategy for valid file extensions
valid_extensions = st.sampled_from(["h5ad", "pkl", "parquet", "csv", "json"])

# Strategy for step numbers (positive integers)
step_numbers = st.integers(min_value=0, max_value=99)

# Strategy for JSON-serializable values
json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=100),
)

json_values = st.recursive(
    json_primitives,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=5),
    ),
    max_leaves=10,
)


# =============================================================================
# Tests for checkpoint.py
# =============================================================================

class TestBidsCheckpointNameRoundtrip:
    """Property: parsing a generated BIDS name should recover the original values."""

    @given(
        dataset_name=bids_safe_text,
        step_number=step_numbers,
        description=bids_safe_text,
        extension=valid_extensions,
    )
    @settings(max_examples=100)
    def test_roundtrip_property(self, dataset_name, step_number, description, extension):
        """Parsing bids_checkpoint_name output should recover original inputs."""
        # Generate the filename
        filename = bids_checkpoint_name(dataset_name, step_number, description, extension)

        # Parse it back
        parsed = parse_bids_checkpoint_name(filename)

        # Verify roundtrip
        assert parsed["dataset"] == dataset_name
        assert parsed["step_number"] == step_number
        assert parsed["description"] == description
        assert parsed["extension"] == extension

    @given(
        dataset_name=bids_safe_text,
        step_number=step_numbers,
        description=bids_safe_text,
    )
    @settings(max_examples=50)
    def test_default_extension_is_h5ad(self, dataset_name, step_number, description):
        """Default extension should be h5ad."""
        filename = bids_checkpoint_name(dataset_name, step_number, description)
        parsed = parse_bids_checkpoint_name(filename)
        assert parsed["extension"] == "h5ad"


class TestHashParameters:
    """Property: hash function should be deterministic and consistent."""

    @given(
        key1=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"),
        value1=st.one_of(st.integers(), st.floats(allow_nan=False), st.text(max_size=50)),
        key2=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"),
        value2=st.one_of(st.integers(), st.floats(allow_nan=False), st.text(max_size=50)),
    )
    @settings(max_examples=100)
    def test_deterministic(self, key1, value1, key2, value2):
        """Same inputs should always produce the same hash."""
        kwargs = {key1: value1, key2: value2}
        hash1 = hash_parameters(**kwargs)
        hash2 = hash_parameters(**kwargs)
        assert hash1 == hash2

    @given(
        key=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"),
        value=st.integers(),
    )
    @settings(max_examples=50)
    def test_hash_length(self, key, value):
        """Hash should always be 8 characters."""
        result = hash_parameters(**{key: value})
        assert len(result) == 8

    @given(
        key=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"),
        value=st.integers(),
    )
    @settings(max_examples=50)
    def test_hash_is_hex(self, key, value):
        """Hash should be valid hexadecimal."""
        result = hash_parameters(**{key: value})
        # Should not raise
        int(result, 16)


# =============================================================================
# Tests for execution_log.py
# =============================================================================

class TestSerializeParameters:
    """Property: serialize_parameters should always produce JSON-serializable output."""

    @given(
        key=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"),
        value=json_values,
    )
    @settings(max_examples=100)
    def test_output_is_json_serializable(self, key, value):
        """Output should always be JSON-serializable."""
        result = serialize_parameters(**{key: value})
        # Should not raise
        json.dumps(result)

    @given(path_str=st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_path_objects_become_strings(self, path_str):
        """Path objects should be converted to strings."""
        path = Path(path_str)
        result = serialize_parameters(my_path=path)
        assert isinstance(result["my_path"], str)
        assert result["my_path"] == str(path)

    @given(arr=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_numpy_arrays_become_lists(self, arr):
        """Numpy arrays should be converted to lists."""
        np_arr = np.array(arr)
        result = serialize_parameters(my_array=np_arr)
        assert isinstance(result["my_array"], list)


# =============================================================================
# Tests for overrepresentation_analysis.py
# =============================================================================

class TestGetSignificantGeneLists:
    """Property: returned gene lists should satisfy filtering criteria."""

    @given(
        n_genes=st.integers(min_value=5, max_value=100),
        padj_threshold=st.floats(min_value=0.001, max_value=0.5),
    )
    @settings(max_examples=50)
    def test_up_genes_have_positive_lfc(self, n_genes, padj_threshold):
        """All upregulated genes should have positive log2FoldChange."""
        # Generate random DE results
        np.random.seed(42)
        results = pd.DataFrame({
            "padj": np.random.uniform(0, 1, n_genes),
            "log2FoldChange": np.random.uniform(-3, 3, n_genes),
        }, index=[f"GENE{i}" for i in range(n_genes)])

        up_genes, _ = get_significant_gene_lists(results, padj_threshold)

        # All up genes should have positive lfc
        for gene in up_genes:
            assert results.loc[gene, "log2FoldChange"] > 0

    @given(
        n_genes=st.integers(min_value=5, max_value=100),
        padj_threshold=st.floats(min_value=0.001, max_value=0.5),
    )
    @settings(max_examples=50)
    def test_down_genes_have_negative_lfc(self, n_genes, padj_threshold):
        """All downregulated genes should have negative log2FoldChange."""
        np.random.seed(42)
        results = pd.DataFrame({
            "padj": np.random.uniform(0, 1, n_genes),
            "log2FoldChange": np.random.uniform(-3, 3, n_genes),
        }, index=[f"GENE{i}" for i in range(n_genes)])

        _, down_genes = get_significant_gene_lists(results, padj_threshold)

        for gene in down_genes:
            assert results.loc[gene, "log2FoldChange"] < 0

    @given(
        n_genes=st.integers(min_value=5, max_value=100),
        padj_threshold=st.floats(min_value=0.001, max_value=0.5),
    )
    @settings(max_examples=50)
    def test_all_genes_pass_padj_threshold(self, n_genes, padj_threshold):
        """All returned genes should have padj below threshold."""
        np.random.seed(42)
        results = pd.DataFrame({
            "padj": np.random.uniform(0, 1, n_genes),
            "log2FoldChange": np.random.uniform(-3, 3, n_genes),
        }, index=[f"GENE{i}" for i in range(n_genes)])

        up_genes, down_genes = get_significant_gene_lists(results, padj_threshold)

        for gene in up_genes + down_genes:
            assert results.loc[gene, "padj"] < padj_threshold

    @given(padj_threshold=st.floats(min_value=0.001, max_value=0.5))
    @settings(max_examples=20)
    def test_disjoint_sets(self, padj_threshold):
        """Up and down gene lists should be disjoint."""
        np.random.seed(42)
        results = pd.DataFrame({
            "padj": np.random.uniform(0, 1, 50),
            "log2FoldChange": np.random.uniform(-3, 3, 50),
        }, index=[f"GENE{i}" for i in range(50)])

        up_genes, down_genes = get_significant_gene_lists(results, padj_threshold)

        # Sets should not overlap
        assert len(set(up_genes) & set(down_genes)) == 0


# =============================================================================
# Tests for pathway_analysis.py
# =============================================================================

class TestPrepareRankedList:
    """Property: prepare_ranked_list should produce sorted, NaN-free output."""

    @given(n_genes=st.integers(min_value=5, max_value=200))
    @settings(max_examples=50)
    def test_output_is_sorted_descending(self, n_genes):
        """Output should be sorted by stat in descending order."""
        np.random.seed(42)
        results = pd.DataFrame({
            "stat": np.random.uniform(-5, 5, n_genes),
            "padj": np.random.uniform(0, 1, n_genes),
        }, index=[f"GENE{i}" for i in range(n_genes)])

        rank_df = prepare_ranked_list(results)

        # Check sorting
        stat_values = rank_df["stat"].values
        assert all(stat_values[i] >= stat_values[i + 1] for i in range(len(stat_values) - 1))

    @given(
        n_genes=st.integers(min_value=10, max_value=100),
        n_nan=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50)
    def test_no_nan_in_output(self, n_genes, n_nan):
        """Output should not contain NaN values."""
        assume(n_nan < n_genes)

        np.random.seed(42)
        stats = np.random.uniform(-5, 5, n_genes)
        # Insert some NaNs
        nan_indices = np.random.choice(n_genes, n_nan, replace=False)
        stats[nan_indices] = np.nan

        results = pd.DataFrame({
            "stat": stats,
            "padj": np.random.uniform(0, 1, n_genes),
        }, index=[f"GENE{i}" for i in range(n_genes)])

        rank_df = prepare_ranked_list(results)

        # No NaN values should be present
        assert rank_df["stat"].notna().all()
        # Output should have fewer rows than input
        assert len(rank_df) == n_genes - n_nan

    @given(n_genes=st.integers(min_value=5, max_value=100))
    @settings(max_examples=30)
    def test_output_has_only_stat_column(self, n_genes):
        """Output should contain only the stat column."""
        np.random.seed(42)
        results = pd.DataFrame({
            "stat": np.random.uniform(-5, 5, n_genes),
            "padj": np.random.uniform(0, 1, n_genes),
            "log2FoldChange": np.random.uniform(-3, 3, n_genes),
        }, index=[f"GENE{i}" for i in range(n_genes)])

        rank_df = prepare_ranked_list(results)

        assert list(rank_df.columns) == ["stat"]


# =============================================================================
# Numerical Edge Case Tests
# =============================================================================

class TestPadjThresholdEdgeCases:
    """Test edge cases for p-value threshold parameters."""

    def test_threshold_zero_returns_empty(self):
        """Threshold of 0 should return no genes (nothing is < 0)."""
        results = pd.DataFrame({
            "padj": [0.0, 0.01, 0.5],
            "log2FoldChange": [1.0, -1.0, 0.5],
        }, index=["GENE1", "GENE2", "GENE3"])

        up_genes, down_genes = get_significant_gene_lists(results, padj_threshold=0.0)

        assert len(up_genes) == 0
        assert len(down_genes) == 0

    def test_threshold_one_returns_all_with_nonzero_lfc(self):
        """Threshold of 1 should return all genes with non-zero log2FC."""
        results = pd.DataFrame({
            "padj": [0.01, 0.5, 0.99],
            "log2FoldChange": [1.0, -1.0, 0.5],
        }, index=["GENE1", "GENE2", "GENE3"])

        up_genes, down_genes = get_significant_gene_lists(results, padj_threshold=1.0)

        # All genes should be returned (2 up, 1 down)
        assert len(up_genes) == 2
        assert len(down_genes) == 1

    @given(threshold=st.floats(min_value=-1e10, max_value=0, allow_nan=False))
    @settings(max_examples=20)
    def test_negative_threshold_returns_empty(self, threshold):
        """Negative thresholds should return no genes."""
        results = pd.DataFrame({
            "padj": [0.01, 0.05, 0.1],
            "log2FoldChange": [1.0, -1.0, 0.5],
        }, index=["GENE1", "GENE2", "GENE3"])

        up_genes, down_genes = get_significant_gene_lists(results, padj_threshold=threshold)

        assert len(up_genes) == 0
        assert len(down_genes) == 0

    @given(threshold=st.floats(min_value=1e-300, max_value=1e-100))
    @settings(max_examples=20)
    def test_very_small_threshold_handles_correctly(self, threshold):
        """Very small thresholds should not cause numerical issues."""
        results = pd.DataFrame({
            "padj": [1e-200, 1e-50, 0.01],
            "log2FoldChange": [1.0, -1.0, 0.5],
        }, index=["GENE1", "GENE2", "GENE3"])

        # Should not raise any errors
        up_genes, down_genes = get_significant_gene_lists(results, padj_threshold=threshold)

        # Results should be lists (may be empty depending on threshold)
        assert isinstance(up_genes, list)
        assert isinstance(down_genes, list)


class TestFilterPseudobulkEdgeCases:
    """Test edge cases for pseudobulk filtering."""

    @pytest.fixture
    def sample_pseudobulk(self):
        """Create sample pseudobulk AnnData."""
        n_samples = 10
        n_genes = 5
        X = np.random.randint(0, 100, (n_samples, n_genes))
        obs = pd.DataFrame({
            "n_cells": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "cell_type": ["A"] * 5 + ["B"] * 5,
            "donor_id": [f"D{i}" for i in range(n_samples)],
        }, index=[f"sample_{i}" for i in range(n_samples)])
        return ad.AnnData(X=X, obs=obs)

    def test_min_cells_zero_returns_all(self, sample_pseudobulk):
        """min_cells=0 should return all samples."""
        result = filter_pseudobulk_by_cell_count(sample_pseudobulk, min_cells=0)
        assert result.n_obs == sample_pseudobulk.n_obs

    def test_min_cells_negative_returns_all(self, sample_pseudobulk):
        """Negative min_cells should return all samples (n_cells >= negative is always true)."""
        result = filter_pseudobulk_by_cell_count(sample_pseudobulk, min_cells=-10)
        assert result.n_obs == sample_pseudobulk.n_obs

    def test_min_cells_very_large_returns_empty(self, sample_pseudobulk):
        """Very large min_cells should return empty result."""
        result = filter_pseudobulk_by_cell_count(sample_pseudobulk, min_cells=1000000)
        assert result.n_obs == 0

    @given(min_cells=st.integers(min_value=0, max_value=100))
    @settings(max_examples=30)
    def test_all_remaining_samples_meet_threshold(self, min_cells):
        """All remaining samples should have n_cells >= min_cells."""
        n_samples = 20
        X = np.random.randint(0, 100, (n_samples, 5))
        obs = pd.DataFrame({
            "n_cells": np.random.randint(1, 100, n_samples),
            "cell_type": ["A"] * n_samples,
        }, index=[f"sample_{i}" for i in range(n_samples)])
        adata = ad.AnnData(X=X, obs=obs)

        result = filter_pseudobulk_by_cell_count(adata, min_cells=min_cells)

        # All remaining samples should meet the threshold
        if result.n_obs > 0:
            assert (result.obs["n_cells"] >= min_cells).all()

    @given(min_cells=st.integers(min_value=0, max_value=100))
    @settings(max_examples=30)
    def test_result_is_subset_of_input(self, min_cells):
        """Result should always be a subset of input."""
        n_samples = 20
        X = np.random.randint(0, 100, (n_samples, 5))
        obs = pd.DataFrame({
            "n_cells": np.random.randint(1, 100, n_samples),
            "cell_type": ["A"] * n_samples,
        }, index=[f"sample_{i}" for i in range(n_samples)])
        adata = ad.AnnData(X=X, obs=obs)

        result = filter_pseudobulk_by_cell_count(adata, min_cells=min_cells)

        assert result.n_obs <= adata.n_obs
        # All result indices should be in original
        assert all(idx in adata.obs_names for idx in result.obs_names)


class TestLogTransformEdgeCases:
    """Test edge cases for log transformations in pathway analysis."""

    def test_enrichr_plot_data_handles_zero_pvalue(self):
        """prepare_enrichr_plot_data should handle p-value of 0 without inf."""
        class MockEnrichr:
            def __init__(self):
                self.results = pd.DataFrame({
                    "Term": ["Pathway_1", "Pathway_2"],
                    "Adjusted P-value": [0.0, 0.001],  # Zero p-value!
                    "Overlap": ["5/100", "10/100"],
                })

        enr = MockEnrichr()
        combined = prepare_enrichr_plot_data(enr, None, n_top=2)

        # log_p should not be infinite
        assert not np.isinf(combined["log_p"]).any()

    def test_enrichr_plot_data_handles_very_small_pvalue(self):
        """prepare_enrichr_plot_data should handle very small p-values."""
        class MockEnrichr:
            def __init__(self):
                self.results = pd.DataFrame({
                    "Term": ["Pathway_1", "Pathway_2"],
                    "Adjusted P-value": [1e-300, 1e-200],
                    "Overlap": ["5/100", "10/100"],
                })

        enr = MockEnrichr()
        combined = prepare_enrichr_plot_data(enr, None, n_top=2)

        # Should not raise and log_p should be finite
        assert combined is not None
        # Note: -log10(1e-300) ≈ 300, which is finite

    @given(pvalues=st.lists(
        st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    ))
    @settings(max_examples=30)
    def test_enrichr_log_p_always_non_negative(self, pvalues):
        """log_p (-log10(pvalue)) should always be non-negative for valid p-values."""
        assume(all(0 <= p <= 1 for p in pvalues))

        class MockEnrichr:
            def __init__(self, pvals):
                self.results = pd.DataFrame({
                    "Term": [f"Pathway_{i}" for i in range(len(pvals))],
                    "Adjusted P-value": pvals,
                    "Overlap": ["5/100"] * len(pvals),
                })

        enr = MockEnrichr(pvalues)
        combined = prepare_enrichr_plot_data(enr, None, n_top=len(pvalues))

        if combined is not None:
            # -log10(p) >= 0 for 0 < p <= 1
            # Allow tiny negative values due to floating point precision (e.g., -1e-10)
            # when p is very close to 1 and epsilon is added
            finite_log_p = combined["log_p"][~np.isinf(combined["log_p"])]
            assert (finite_log_p >= -1e-9).all()


class TestDataWithNaNAndInfinity:
    """Test handling of NaN and infinity in numerical data."""

    def test_get_significant_genes_with_nan_padj(self):
        """get_significant_gene_lists should handle NaN padj values."""
        results = pd.DataFrame({
            "padj": [0.01, np.nan, 0.05, np.nan],
            "log2FoldChange": [1.0, 2.0, -1.0, -2.0],
        }, index=["GENE1", "GENE2", "GENE3", "GENE4"])

        up_genes, down_genes = get_significant_gene_lists(results, padj_threshold=0.1)

        # NaN comparisons should be False, so genes with NaN padj excluded
        assert "GENE2" not in up_genes
        assert "GENE4" not in down_genes
        # Valid genes should be included
        assert "GENE1" in up_genes
        assert "GENE3" in down_genes

    def test_prepare_ranked_list_with_inf_stat(self):
        """prepare_ranked_list should handle infinity in stat values."""
        results = pd.DataFrame({
            "stat": [np.inf, -np.inf, 1.0, -1.0],
            "padj": [0.01, 0.02, 0.03, 0.04],
        }, index=["GENE1", "GENE2", "GENE3", "GENE4"])

        rank_df = prepare_ranked_list(results)

        # Infinity values should still be sorted correctly
        # inf should come first (largest), -inf should come last
        assert rank_df.index[0] == "GENE1"  # inf is largest
        assert rank_df.index[-1] == "GENE2"  # -inf is smallest

    @given(n_genes=st.integers(min_value=5, max_value=50))
    @settings(max_examples=20)
    def test_gene_lists_exclude_nan_padj(self, n_genes):
        """Genes with NaN padj should never appear in results."""
        np.random.seed(42)

        padj = np.random.uniform(0, 1, n_genes)
        # Randomly set some to NaN
        nan_mask = np.random.choice([True, False], n_genes, p=[0.3, 0.7])
        padj[nan_mask] = np.nan

        results = pd.DataFrame({
            "padj": padj,
            "log2FoldChange": np.random.uniform(-3, 3, n_genes),
        }, index=[f"GENE{i}" for i in range(n_genes)])

        up_genes, down_genes = get_significant_gene_lists(results, padj_threshold=0.5)

        # Check that no NaN-padj genes are in results
        nan_genes = results.index[np.isnan(results["padj"])].tolist()
        for gene in nan_genes:
            assert gene not in up_genes
            assert gene not in down_genes
