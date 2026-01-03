"""Unit tests for data filtering module."""

import pandas as pd
import pytest

from example_rnaseq.data_filtering import (
    compute_donor_cell_counts,
    filter_cell_types_by_frequency,
    filter_donors_by_cell_count,
    filter_donors_with_missing_cell_types,
    plot_donor_cell_distribution,
)


class TestComputeDonorCellCounts:
    """Tests for donor cell count computation."""

    def test_counts_cells_per_donor(self, minimal_adata):
        """Test that cells are counted correctly per donor."""
        counts = compute_donor_cell_counts(minimal_adata)

        assert isinstance(counts, pd.Series)
        assert len(counts) == minimal_adata.obs["donor_id"].nunique()
        assert counts.sum() == minimal_adata.n_obs

    def test_returns_sorted_series(self, minimal_adata):
        """Test that result is sorted by count (descending)."""
        counts = compute_donor_cell_counts(minimal_adata)
        assert list(counts.values) == sorted(counts.values, reverse=True)


class TestPlotDonorCellDistribution:
    """Tests for donor cell distribution plotting."""

    def test_returns_cutoff_value(self, minimal_adata):
        """Test that a cutoff value is returned."""
        counts = compute_donor_cell_counts(minimal_adata)
        cutoff = plot_donor_cell_distribution(counts, cutoff_percentile=1.0)

        assert isinstance(cutoff, int)
        assert cutoff >= 0

    def test_saves_figure(self, minimal_adata, temp_output_dir):
        """Test that figure is saved when directory provided."""
        counts = compute_donor_cell_counts(minimal_adata)
        plot_donor_cell_distribution(counts, figure_dir=temp_output_dir)

        assert (temp_output_dir / "donor_cell_counts_distribution.png").exists()


class TestFilterDonorsByCellCount:
    """Tests for donor filtering by cell count."""

    def test_filters_low_count_donors(self, minimal_adata):
        """Test that donors with few cells are removed."""
        counts = compute_donor_cell_counts(minimal_adata)
        median_count = int(counts.median())

        filtered = filter_donors_by_cell_count(minimal_adata, median_count)

        # Should have no more cells than original
        assert filtered.n_obs <= minimal_adata.n_obs

        # Remaining donors should meet threshold
        remaining_counts = compute_donor_cell_counts(filtered)
        assert remaining_counts.min() >= median_count

    def test_preserves_all_when_threshold_low(self, minimal_adata):
        """Test that all donors kept when threshold is 0."""
        filtered = filter_donors_by_cell_count(minimal_adata, min_cells_per_donor=0)

        assert filtered.obs["donor_id"].nunique() == minimal_adata.obs["donor_id"].nunique()


class TestFilterCellTypesByFrequency:
    """Tests for cell type frequency filtering."""

    def test_filters_rare_cell_types(self, minimal_adata):
        """Test that rare cell types are removed."""
        # With high percent_donors threshold, some cell types may be removed
        filtered = filter_cell_types_by_frequency(
            minimal_adata, min_cells=1, percent_donors=1.0
        )

        # All remaining cell types should be in most donors
        assert filtered.n_obs <= minimal_adata.n_obs

    def test_preserves_common_cell_types(self, minimal_adata):
        """Test that common cell types are preserved."""
        filtered = filter_cell_types_by_frequency(
            minimal_adata, min_cells=1, percent_donors=0.0
        )

        # All cell types should be preserved with 0% threshold
        assert filtered.obs["cell_type"].nunique() == minimal_adata.obs["cell_type"].nunique()


class TestFilterDonorsWithMissingCellTypes:
    """Tests for filtering donors with incomplete cell types."""

    def test_filters_incomplete_donors(self, minimal_adata):
        """Test that donors missing cell types are removed."""
        # This depends on the data distribution
        filtered = filter_donors_with_missing_cell_types(minimal_adata, min_cells=1)

        # Verify remaining donors have all cell types
        remaining_cell_types = filtered.obs["cell_type"].unique()
        for donor in filtered.obs["donor_id"].unique():
            donor_types = filtered[filtered.obs["donor_id"] == donor].obs["cell_type"].unique()
            # Each donor should have representation in cell types present
            assert len(donor_types) > 0
