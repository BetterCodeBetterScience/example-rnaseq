"""Integration tests for the RNA-seq workflow pipelines.

These tests run complete pipelines end-to-end using the test dataset.
They are marked as 'integration' and can be deselected with '-m "not integration"'.
"""

import pytest

from example_rnaseq.clustering import run_clustering_pipeline
from example_rnaseq.dimensionality_reduction import run_dimensionality_reduction_pipeline
from example_rnaseq.preprocessing import run_preprocessing_pipeline
from example_rnaseq.pseudobulk import run_pseudobulk_pipeline
from example_rnaseq.quality_control import run_qc_pipeline


@pytest.fixture
def adata_for_integration(test_adata):
    """Prepare test AnnData for integration tests."""
    adata = test_adata.copy()
    # Ensure we have the raw counts
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    return adata


@pytest.mark.integration
class TestQcPipelineIntegration:
    """Integration tests for QC pipeline."""

    def test_qc_pipeline_runs(self, adata_for_integration, temp_output_dir):
        """Test that QC pipeline runs to completion."""
        adata = adata_for_integration.copy()

        # Run QC pipeline with lenient thresholds for test data
        result = run_qc_pipeline(
            adata,
            min_genes=10,
            max_genes=10000,
            min_counts=50,
            max_counts=100000,
            max_hb_pct=50.0,
            expected_doublet_rate=0.06,
            figure_dir=temp_output_dir,
        )

        # Check outputs
        assert result is not None
        assert result.n_obs > 0
        assert "counts" in result.layers  # Raw counts preserved


@pytest.mark.integration
class TestPreprocessingPipelineIntegration:
    """Integration tests for preprocessing pipeline."""

    @pytest.fixture
    def adata_after_qc(self, adata_for_integration):
        """Run QC first to get data ready for preprocessing."""
        adata = adata_for_integration.copy()

        # Minimal QC
        from example_rnaseq.quality_control import (
            annotate_gene_types,
            calculate_qc_metrics,
        )

        adata = annotate_gene_types(adata)
        adata = calculate_qc_metrics(adata)
        adata.layers["counts"] = adata.X.copy()

        return adata

    def test_preprocessing_pipeline_runs(self, adata_after_qc):
        """Test that preprocessing pipeline runs to completion."""
        result = run_preprocessing_pipeline(
            adata_after_qc,
            target_sum=1e4,
            n_top_genes=100,  # Smaller for test data
            batch_key="donor_id",
        )

        # Check outputs
        assert result is not None
        assert "highly_variable" in result.var.columns
        assert "X_pca" in result.obsm


@pytest.mark.integration
class TestDimReductionPipelineIntegration:
    """Integration tests for dimensionality reduction pipeline."""

    @pytest.fixture
    def adata_after_preprocessing(self, adata_for_integration):
        """Run preprocessing to get data ready for dim reduction."""
        adata = adata_for_integration.copy()

        from example_rnaseq.quality_control import annotate_gene_types
        from example_rnaseq.preprocessing import (
            log_transform,
            normalize_counts,
            run_pca,
            select_highly_variable_genes,
        )

        adata = annotate_gene_types(adata)
        adata.layers["counts"] = adata.X.copy()
        normalize_counts(adata)
        log_transform(adata)
        select_highly_variable_genes(adata, n_top_genes=100)
        run_pca(adata)
        adata.obs["total_counts"] = adata.X.sum(axis=1)

        return adata

    def test_dim_reduction_pipeline_runs(self, adata_after_preprocessing, temp_output_dir):
        """Test that dim reduction pipeline runs to completion."""
        result = run_dimensionality_reduction_pipeline(
            adata_after_preprocessing,
            batch_key="donor_id",
            n_neighbors=10,
            n_pcs=20,
            figure_dir=temp_output_dir,
        )

        # Check outputs
        assert result is not None
        assert "X_umap" in result.obsm


@pytest.mark.integration
class TestClusteringPipelineIntegration:
    """Integration tests for clustering pipeline."""

    @pytest.fixture
    def adata_after_dimred(self, adata_for_integration):
        """Run dim reduction to get data ready for clustering."""
        adata = adata_for_integration.copy()

        from example_rnaseq.dimensionality_reduction import (
            compute_neighbors,
            compute_umap,
            run_harmony_integration,
        )
        from example_rnaseq.quality_control import annotate_gene_types
        from example_rnaseq.preprocessing import (
            log_transform,
            normalize_counts,
            run_pca,
            select_highly_variable_genes,
        )

        adata = annotate_gene_types(adata)
        adata.layers["counts"] = adata.X.copy()
        normalize_counts(adata)
        log_transform(adata)
        select_highly_variable_genes(adata, n_top_genes=100)
        run_pca(adata)
        adata.obs["total_counts"] = adata.X.sum(axis=1)
        adata, use_rep = run_harmony_integration(adata)
        compute_neighbors(adata, n_neighbors=10, n_pcs=20, use_rep=use_rep)
        compute_umap(adata, init_pos="spectral")

        return adata

    def test_clustering_pipeline_runs(self, adata_after_dimred, temp_output_dir):
        """Test that clustering pipeline runs to completion."""
        result = run_clustering_pipeline(
            adata_after_dimred,
            resolution=0.5,
            figure_dir=temp_output_dir,
        )

        # Check outputs
        assert result is not None
        assert "leiden_0.5" in result.obs.columns


@pytest.mark.integration
class TestPseudobulkPipelineIntegration:
    """Integration tests for pseudobulk pipeline."""

    def test_pseudobulk_pipeline_runs(self, adata_for_integration, temp_output_dir):
        """Test that pseudobulk pipeline runs and produces correct aggregation."""
        import scipy.sparse as sp

        adata = adata_for_integration.copy()

        # Ensure required columns exist
        if "development_stage" not in adata.obs.columns:
            adata.obs["development_stage"] = "45-year-old human stage"
        if "sex" not in adata.obs.columns:
            adata.obs["sex"] = "male"

        # Store input counts for verification
        if "counts" in adata.layers:
            input_X = adata.layers["counts"]
        else:
            input_X = adata.X
        if sp.issparse(input_X):
            input_total = input_X.toarray().sum()
        else:
            input_total = input_X.sum()

        n_donors = adata.obs["donor_id"].nunique()
        n_cell_types = adata.obs["cell_type"].nunique()
        n_input_cells = adata.n_obs

        result = run_pseudobulk_pipeline(
            adata,
            group_col="cell_type",
            donor_col="donor_id",
            metadata_cols=["development_stage", "sex"],
            min_cells=1,  # Low threshold for test data
            figure_dir=temp_output_dir,
        )

        # Check basic outputs
        assert result is not None
        assert result.n_obs > 0
        assert "n_cells" in result.obs.columns
        assert "cell_type" in result.obs.columns
        assert "donor_id" in result.obs.columns

        # Verify aggregation actually happened:
        # 1. Number of pseudobulk samples should be <= donors x cell_types
        assert result.n_obs <= n_donors * n_cell_types, (
            f"Pseudobulk samples ({result.n_obs}) should be <= donors x cell_types "
            f"({n_donors} x {n_cell_types} = {n_donors * n_cell_types})"
        )

        # 2. Total cell counts should match input
        assert result.obs["n_cells"].sum() == n_input_cells, (
            f"Sum of n_cells ({result.obs['n_cells'].sum()}) should equal "
            f"input cell count ({n_input_cells})"
        )

        # 3. Total counts should be preserved (sums should match within tolerance)
        # Note: Small discrepancies can occur due to floating-point rounding during
        # sparse matrix aggregation and the np.round().astype(int) step
        result_X = result.X
        if sp.issparse(result_X):
            result_total = result_X.toarray().sum()
        else:
            result_total = result_X.sum()
        relative_diff = abs(result_total - input_total) / input_total
        assert relative_diff < 1e-5, (
            f"Total counts should be preserved: input={input_total}, output={result_total}, "
            f"relative difference={relative_diff:.2e}"
        )

        # 4. Each pseudobulk sample should have at least 1 cell
        assert (result.obs["n_cells"] >= 1).all(), "All samples should have n_cells >= 1"


@pytest.mark.integration
@pytest.mark.slow
class TestFullWorkflowIntegration:
    """Full workflow integration test (marked slow)."""

    def test_full_pipeline_chain(self, adata_for_integration, temp_output_dir):
        """Test running the complete pipeline chain."""
        from example_rnaseq.clustering import run_leiden_clustering
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
        from example_rnaseq.quality_control import (
            annotate_gene_types,
            calculate_qc_metrics,
        )

        adata = adata_for_integration.copy()

        # Step 1: QC annotation
        adata = annotate_gene_types(adata)
        adata = calculate_qc_metrics(adata)
        adata.layers["counts"] = adata.X.copy()

        # Step 2: Preprocessing
        normalize_counts(adata)
        log_transform(adata)
        select_highly_variable_genes(adata, n_top_genes=100)
        run_pca(adata)
        adata.obs["total_counts"] = adata.X.sum(axis=1)

        # Step 3: Dim reduction
        adata, use_rep = run_harmony_integration(adata)
        compute_neighbors(adata, n_neighbors=10, n_pcs=20, use_rep=use_rep)
        compute_umap(adata, init_pos="spectral")

        # Step 4: Clustering
        run_leiden_clustering(adata, resolution=0.5, key_added="leiden_0.5")

        # Step 5: Pseudobulk
        if "development_stage" not in adata.obs.columns:
            adata.obs["development_stage"] = "45-year-old human stage"
        if "sex" not in adata.obs.columns:
            adata.obs["sex"] = "male"

        pb = run_pseudobulk_pipeline(
            adata,
            group_col="cell_type",
            donor_col="donor_id",
            min_cells=1,
        )

        # Verify final outputs
        assert adata.n_obs > 0
        assert pb.n_obs > 0
        assert "X_umap" in adata.obsm
        assert "leiden_0.5" in adata.obs.columns
