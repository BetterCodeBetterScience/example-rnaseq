"""Unit tests for preprocessing module."""

import numpy as np
import pytest
import scipy.sparse as sp

from example_rnaseq.preprocessing import (
    filter_nuisance_genes_from_hvg,
    identify_nuisance_genes,
    log_transform,
    normalize_counts,
    run_pca,
    select_highly_variable_genes,
)


class TestNormalizeCounts:
    """Tests for count normalization."""

    def test_normalizes_to_target_sum(self, minimal_adata):
        """Test that cells are normalized to target sum."""
        adata = minimal_adata.copy()
        target_sum = 1e4

        normalized = normalize_counts(adata, target_sum=target_sum)

        # Check row sums are close to target
        X = normalized.X.toarray() if sp.issparse(normalized.X) else normalized.X
        row_sums = X.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, target_sum, decimal=0)

    def test_different_target_sums(self, minimal_adata):
        """Test different target sum values."""
        for target in [100, 1000, 10000]:
            adata = minimal_adata.copy()
            normalized = normalize_counts(adata, target_sum=target)

            X = normalized.X.toarray() if sp.issparse(normalized.X) else normalized.X
            row_sums = X.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, target, decimal=0)


class TestLogTransform:
    """Tests for log transformation."""

    def test_applies_log1p(self, minimal_adata):
        """Test that log1p is applied correctly."""
        adata = minimal_adata.copy()
        # First normalize
        normalize_counts(adata, target_sum=1e4)

        # Store pre-transform values
        X_before = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()

        log_transform(adata)

        X_after = adata.X.toarray() if sp.issparse(adata.X) else adata.X

        # Check log1p was applied
        expected = np.log1p(X_before)
        np.testing.assert_array_almost_equal(X_after, expected)

    def test_zeros_remain_zero(self, minimal_adata):
        """Test that zeros remain zeros after log1p."""
        adata = minimal_adata.copy()
        X_before = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        zero_mask = X_before == 0

        log_transform(adata)

        X_after = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        np.testing.assert_array_equal(X_after[zero_mask], 0)


class TestIdentifyNuisanceGenes:
    """Tests for nuisance gene identification."""

    @pytest.fixture
    def adata_with_gene_names(self, minimal_adata):
        """Create AnnData with gene names as var_names (not ENSG IDs)."""
        adata = minimal_adata.copy()
        # Use feature_name as var_names
        adata.var_names = adata.var["feature_name"].values
        return adata

    def test_identifies_mt_genes(self, adata_with_gene_names):
        """Test mitochondrial gene identification."""
        nuisance = identify_nuisance_genes(adata_with_gene_names)

        # Check MT genes are identified
        mt_in_nuisance = [g for g in nuisance if g.startswith("MT-")]
        assert len(mt_in_nuisance) > 0

    def test_identifies_ribosomal_genes(self, adata_with_gene_names):
        """Test ribosomal gene identification."""
        nuisance = identify_nuisance_genes(adata_with_gene_names)

        ribo_in_nuisance = [g for g in nuisance if g.startswith(("RPS", "RPL"))]
        assert len(ribo_in_nuisance) > 0

    def test_returns_list(self, adata_with_gene_names):
        """Test that result is a list."""
        nuisance = identify_nuisance_genes(adata_with_gene_names)
        assert isinstance(nuisance, list)


class TestSelectHighlyVariableGenes:
    """Tests for HVG selection."""

    def test_selects_hvgs(self, minimal_adata):
        """Test that HVGs are selected."""
        adata = minimal_adata.copy()
        # Need counts layer
        adata.layers["counts"] = adata.X.copy()

        select_highly_variable_genes(adata, n_top_genes=10)

        assert "highly_variable" in adata.var.columns
        assert adata.var["highly_variable"].sum() > 0
        assert adata.var["highly_variable"].sum() <= 10

    def test_respects_n_top_genes(self, minimal_adata):
        """Test that n_top_genes parameter is respected."""
        adata = minimal_adata.copy()
        adata.layers["counts"] = adata.X.copy()

        n_hvg = 5
        select_highly_variable_genes(adata, n_top_genes=n_hvg)

        # Should select at most n_top_genes
        assert adata.var["highly_variable"].sum() <= n_hvg


class TestFilterNuisanceGenesFromHvg:
    """Tests for filtering nuisance genes from HVG list."""

    def test_removes_nuisance_from_hvg(self, minimal_adata):
        """Test that nuisance genes are removed from HVG list."""
        adata = minimal_adata.copy()
        adata.layers["counts"] = adata.X.copy()

        # Select HVGs first
        select_highly_variable_genes(adata, n_top_genes=30)

        # Get initial HVG count
        initial_hvg = adata.var["highly_variable"].sum()

        # Filter nuisance
        filter_nuisance_genes_from_hvg(adata)

        # Check MT/ribo genes are no longer HVG
        hvg_names = adata.var_names[adata.var["highly_variable"]]
        mt_hvg = [g for g in hvg_names if g.startswith("MT-")]
        ribo_hvg = [g for g in hvg_names if g.startswith(("RPS", "RPL"))]

        assert len(mt_hvg) == 0
        assert len(ribo_hvg) == 0


class TestRunPca:
    """Tests for PCA computation."""

    def test_creates_pca_embedding(self, minimal_adata):
        """Test that PCA creates embedding."""
        adata = minimal_adata.copy()
        adata.layers["counts"] = adata.X.copy()

        # Preprocess first
        normalize_counts(adata)
        log_transform(adata)
        select_highly_variable_genes(adata, n_top_genes=20)

        run_pca(adata)

        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape[0] == adata.n_obs

    def test_stores_variance_ratio(self, minimal_adata):
        """Test that variance ratio is stored."""
        adata = minimal_adata.copy()
        adata.layers["counts"] = adata.X.copy()

        normalize_counts(adata)
        log_transform(adata)
        select_highly_variable_genes(adata, n_top_genes=20)
        run_pca(adata)

        assert "pca" in adata.uns
        assert "variance_ratio" in adata.uns["pca"]
