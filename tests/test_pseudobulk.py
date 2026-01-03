"""Unit tests for pseudobulk module."""

import numpy as np
import pytest
import scipy.sparse as sp

from example_rnaseq.pseudobulk import (
    compute_pseudobulk_qc,
    create_pseudobulk,
    filter_pseudobulk_by_cell_count,
    plot_pseudobulk_qc,
)


@pytest.fixture
def adata_for_pseudobulk(minimal_adata):
    """Create AnnData suitable for pseudobulking."""
    adata = minimal_adata.copy()
    # Ensure counts layer exists (raw counts)
    adata.layers["counts"] = adata.X.copy()
    # Add metadata
    adata.obs["development_stage"] = [
        f"{25 + (i % 3) * 20}-year-old human stage"
        for i in range(adata.n_obs)
    ]
    adata.obs["sex"] = ["male" if i % 2 == 0 else "female" for i in range(adata.n_obs)]
    return adata


class TestCreatePseudobulk:
    """Tests for pseudobulk creation."""

    def test_aggregates_by_donor_celltype(self, adata_for_pseudobulk):
        """Test that cells are aggregated by donor and cell type."""
        pb = create_pseudobulk(
            adata_for_pseudobulk,
            group_col="cell_type",
            donor_col="donor_id",
        )

        # Number of samples should be <= donors x cell_types
        n_donors = adata_for_pseudobulk.obs["donor_id"].nunique()
        n_celltypes = adata_for_pseudobulk.obs["cell_type"].nunique()
        assert pb.n_obs <= n_donors * n_celltypes

    def test_preserves_gene_count(self, adata_for_pseudobulk):
        """Test that gene count is preserved."""
        pb = create_pseudobulk(
            adata_for_pseudobulk,
            group_col="cell_type",
            donor_col="donor_id",
        )

        assert pb.n_vars == adata_for_pseudobulk.n_vars

    def test_counts_are_summed(self, adata_for_pseudobulk):
        """Test that counts are summed correctly."""
        adata = adata_for_pseudobulk.copy()
        pb = create_pseudobulk(
            adata,
            group_col="cell_type",
            donor_col="donor_id",
        )

        # Total counts in pseudobulk should equal total in single-cell
        X_sc = adata.layers["counts"]
        if sp.issparse(X_sc):
            X_sc = X_sc.toarray()
        total_sc = X_sc.sum()

        X_pb = pb.X
        if sp.issparse(X_pb):
            X_pb = X_pb.toarray()
        total_pb = X_pb.sum()

        np.testing.assert_almost_equal(total_pb, total_sc)

    def test_includes_cell_counts(self, adata_for_pseudobulk):
        """Test that n_cells is included in obs."""
        pb = create_pseudobulk(
            adata_for_pseudobulk,
            group_col="cell_type",
            donor_col="donor_id",
        )

        assert "n_cells" in pb.obs.columns
        assert pb.obs["n_cells"].sum() == adata_for_pseudobulk.n_obs

    def test_includes_metadata(self, adata_for_pseudobulk):
        """Test that metadata columns are preserved and match source data."""
        adata = adata_for_pseudobulk
        pb = create_pseudobulk(
            adata,
            group_col="cell_type",
            donor_col="donor_id",
            metadata_cols=["development_stage", "sex"],
        )

        # Check columns exist
        assert "development_stage" in pb.obs.columns
        assert "sex" in pb.obs.columns

        # Verify metadata values match the source data for each donor
        for donor_id in pb.obs["donor_id"].unique():
            # Get expected values from source data
            source_mask = adata.obs["donor_id"] == donor_id
            expected_dev_stage = adata.obs.loc[source_mask, "development_stage"].iloc[0]
            expected_sex = adata.obs.loc[source_mask, "sex"].iloc[0]

            # Get values in pseudobulk for this donor
            pb_mask = pb.obs["donor_id"] == donor_id
            pb_dev_stages = pb.obs.loc[pb_mask, "development_stage"]
            pb_sexes = pb.obs.loc[pb_mask, "sex"]

            # All pseudobulk samples for this donor should have matching metadata
            assert (pb_dev_stages == expected_dev_stage).all(), (
                f"Development stage mismatch for donor {donor_id}: "
                f"expected {expected_dev_stage}, got {pb_dev_stages.unique()}"
            )
            assert (pb_sexes == expected_sex).all(), (
                f"Sex mismatch for donor {donor_id}: "
                f"expected {expected_sex}, got {pb_sexes.unique()}"
            )

    def test_returns_integer_counts(self, adata_for_pseudobulk):
        """Test that pseudobulk counts are integers."""
        pb = create_pseudobulk(
            adata_for_pseudobulk,
            group_col="cell_type",
            donor_col="donor_id",
        )

        X = pb.X
        if sp.issparse(X):
            X = X.toarray()

        assert np.allclose(X, X.astype(int))


class TestFilterPseudobulkByCellCount:
    """Tests for filtering pseudobulk by cell count."""

    def test_filters_low_count_samples(self, adata_for_pseudobulk):
        """Test that samples with few cells are removed."""
        pb = create_pseudobulk(
            adata_for_pseudobulk,
            group_col="cell_type",
            donor_col="donor_id",
        )

        # Use a threshold that will filter some samples
        median_cells = int(pb.obs["n_cells"].median())
        filtered = filter_pseudobulk_by_cell_count(pb, min_cells=median_cells)

        assert filtered.n_obs <= pb.n_obs
        assert filtered.obs["n_cells"].min() >= median_cells

    def test_preserves_all_with_low_threshold(self, adata_for_pseudobulk):
        """Test that all samples kept with threshold of 0."""
        pb = create_pseudobulk(
            adata_for_pseudobulk,
            group_col="cell_type",
            donor_col="donor_id",
        )

        filtered = filter_pseudobulk_by_cell_count(pb, min_cells=0)
        assert filtered.n_obs == pb.n_obs


class TestComputePseudobulkQc:
    """Tests for pseudobulk QC computation."""

    def test_computes_total_counts(self, adata_for_pseudobulk):
        """Test that total counts are computed."""
        pb = create_pseudobulk(
            adata_for_pseudobulk,
            group_col="cell_type",
            donor_col="donor_id",
        )

        pb = compute_pseudobulk_qc(pb)

        assert "total_counts" in pb.obs.columns
        assert pb.obs["total_counts"].min() > 0


class TestPlotPseudobulkQc:
    """Tests for pseudobulk QC plotting (smoke tests)."""

    def test_plot_no_error(self, adata_for_pseudobulk, temp_output_dir):
        """Test that QC plot runs without error."""
        pb = create_pseudobulk(
            adata_for_pseudobulk,
            group_col="cell_type",
            donor_col="donor_id",
        )
        pb = compute_pseudobulk_qc(pb)

        plot_pseudobulk_qc(pb, temp_output_dir)
        assert (temp_output_dir / "pseudobulk_violin.png").exists()
