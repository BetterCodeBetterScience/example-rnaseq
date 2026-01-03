"""Shared pytest fixtures for the example_rnaseq test suite."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    """Return the tests directory path."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def test_data_dir(tests_dir) -> Path:
    """Return the test data directory path."""
    return tests_dir / "data"


@pytest.fixture(scope="session")
def pathway_genes(test_data_dir) -> list[str]:
    """Load pathway gene list from JSON file."""
    pathway_file = test_data_dir / "HALLMARK_TNFA_SIGNALING_VIA_NFKB.v2025.1.Hs.json"
    with open(pathway_file) as f:
        data = json.load(f)
    return data["HALLMARK_TNFA_SIGNALING_VIA_NFKB"]["geneSymbols"]


@pytest.fixture(scope="session")
def test_adata(test_data_dir) -> ad.AnnData:
    """Load the test dataset (realistic data for integration tests).

    This fixture is session-scoped to avoid reloading the data for each test.
    """
    test_file = test_data_dir / "dataset-test_raw.h5ad"
    if not test_file.exists():
        pytest.skip("Test data not found. Run tests/create_test_data.py first.")
    return ad.read_h5ad(test_file)


@pytest.fixture
def minimal_adata() -> ad.AnnData:
    """Create a minimal synthetic AnnData for fast unit tests.

    Contains:
    - 100 cells across 3 donors and 2 cell types
    - 50 genes including MT, ribosomal, and hemoglobin markers
    - Sparse count matrix
    - Required obs columns: donor_id, cell_type, development_stage, sex
    - Required var columns: feature_name
    """
    np.random.seed(42)

    n_cells = 100
    n_genes = 50
    n_donors = 3
    n_cell_types = 2

    # Create sparse count matrix (negative binomial-like)
    counts = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))
    X = sp.csr_matrix(counts.astype(np.float32))

    # Create gene names with special markers
    gene_names = []
    # Mitochondrial genes
    gene_names.extend(["MT-CO1", "MT-CO2", "MT-ND1"])
    # Ribosomal genes
    gene_names.extend(["RPS10", "RPS15", "RPL10", "RPL15"])
    # Hemoglobin genes
    gene_names.extend(["HBA1", "HBA2", "HBB"])
    # Regular genes
    gene_names.extend([f"GENE{i}" for i in range(n_genes - len(gene_names))])

    # Create cell metadata
    donors = [f"donor_{i % n_donors}" for i in range(n_cells)]
    cell_types = [f"celltype_{i % n_cell_types}" for i in range(n_cells)]
    ages = [25 + (i % n_donors) * 20 for i in range(n_cells)]  # 25, 45, 65
    development_stages = [f"{age}-year-old human stage" for age in ages]
    sexes = ["male" if i % 2 == 0 else "female" for i in range(n_cells)]

    obs = pd.DataFrame({
        "donor_id": pd.Categorical(donors),
        "cell_type": pd.Categorical(cell_types),
        "development_stage": pd.Categorical(development_stages),
        "sex": pd.Categorical(sexes),
    })
    obs.index = [f"cell_{i}" for i in range(n_cells)]

    var = pd.DataFrame({
        "feature_name": gene_names,
    })
    var.index = [f"ENSG{i:05d}" for i in range(n_genes)]

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.fixture
def minimal_adata_with_qc(minimal_adata) -> ad.AnnData:
    """Minimal AnnData with QC metrics already calculated."""
    adata = minimal_adata.copy()

    # Add gene type annotations
    adata.var["mt"] = adata.var["feature_name"].str.startswith("MT-")
    adata.var["ribo"] = adata.var["feature_name"].str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var["feature_name"].str.contains("^HB[^(P)]", regex=True)

    # Add QC metrics to obs
    X_dense = adata.X.toarray() if sp.issparse(adata.X) else adata.X

    adata.obs["total_counts"] = X_dense.sum(axis=1)
    adata.obs["n_genes_by_counts"] = (X_dense > 0).sum(axis=1)

    # Calculate percentage metrics
    mt_mask = adata.var["mt"].values
    ribo_mask = adata.var["ribo"].values
    hb_mask = adata.var["hb"].values

    total = adata.obs["total_counts"].values
    adata.obs["pct_counts_mt"] = (X_dense[:, mt_mask].sum(axis=1) / total * 100)
    adata.obs["pct_counts_ribo"] = (X_dense[:, ribo_mask].sum(axis=1) / total * 100)
    adata.obs["pct_counts_hb"] = (X_dense[:, hb_mask].sum(axis=1) / total * 100)

    return adata


@pytest.fixture
def minimal_pseudobulk_adata() -> ad.AnnData:
    """Create a minimal pseudobulk AnnData for DE testing.

    Contains:
    - 20 samples (5 donors x 2 cell types x 2 replicates for variation)
    - Actually: 10 samples (5 donors x 2 cell types)
    - 30 genes
    - Integer counts suitable for DESeq2
    """
    np.random.seed(42)

    n_donors = 5
    n_cell_types = 2
    n_genes = 30

    # Create sample combinations
    samples = []
    for donor_idx in range(n_donors):
        for ct_idx in range(n_cell_types):
            samples.append({
                "donor_id": f"donor_{donor_idx}",
                "cell_type": f"celltype_{ct_idx}",
                "n_cells": np.random.randint(50, 200),
                "age": 25 + donor_idx * 15,  # 25, 40, 55, 70, 85
                "sex": "male" if donor_idx % 2 == 0 else "female",
            })

    n_samples = len(samples)
    obs = pd.DataFrame(samples)
    obs["development_stage"] = [f"{age}-year-old human stage" for age in obs["age"]]
    obs.index = [f"{row['cell_type']}::{row['donor_id']}" for _, row in obs.iterrows()]

    # Create count matrix with age-related pattern for some genes
    counts = np.random.negative_binomial(n=10, p=0.1, size=(n_samples, n_genes))

    # Add age effect to first 5 genes
    for i in range(5):
        age_effect = (obs["age"].values - 50) / 10  # Centered around 50
        counts[:, i] = counts[:, i] + (age_effect * 5).astype(int)
        counts[:, i] = np.maximum(counts[:, i], 0)

    gene_names = [f"GENE{i}" for i in range(n_genes)]
    var = pd.DataFrame({"feature_name": gene_names})
    var.index = gene_names

    adata = ad.AnnData(X=counts.astype(np.float32), obs=obs, var=var)
    return adata


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary output directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_checkpoint_dir(tmp_path) -> Path:
    """Create a temporary directory for checkpoint files."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def sample_deseq_results() -> pd.DataFrame:
    """Create sample DESeq2-like results for testing downstream analyses."""
    np.random.seed(42)
    n_genes = 100

    # Generate realistic-looking DE results
    log2fc = np.random.normal(0, 1, n_genes)
    pvalues = np.random.uniform(0, 1, n_genes)
    # Make some genes significant
    pvalues[:10] = np.random.uniform(0, 0.01, 10)

    # Adjust p-values (simple BH correction approximation)
    padj = np.minimum(pvalues * n_genes / np.arange(1, n_genes + 1), 1.0)

    results = pd.DataFrame({
        "baseMean": np.random.exponential(100, n_genes),
        "log2FoldChange": log2fc,
        "lfcSE": np.abs(np.random.normal(0.2, 0.1, n_genes)),
        "stat": log2fc / 0.2,
        "pvalue": pvalues,
        "padj": padj,
    })
    results.index = [f"GENE{i}" for i in range(n_genes)]

    return results


def is_sorted(values, descending: bool = True) -> bool:
    """Check if values are sorted.

    Parameters
    ----------
    values : array-like
        The values to check for sorting
    descending : bool
        If True (default), check for descending order (largest first).
        If False, check for ascending order (smallest first).

    Returns
    -------
    bool
        True if values are sorted in the specified order
    """
    if descending:
        return all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    else:
        return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


# Pytest markers for categorizing tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
