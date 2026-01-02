"""
Script to create a minimal test dataset for the RNA-seq workflow tests.

This script:
1. Loads the full dataset from the source h5ad file
2. Selects 30 donors with varied cell counts and age distribution
3. Selects ~500 genes:
   - 191 genes from TNF-alpha signaling pathway
   - ~200 highly variable genes
   - ~100 weakly variable genes
4. Saves the subset to tests/data/testdata.h5ad
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def load_pathway_genes(pathway_file: Path) -> list[str]:
    """Load gene symbols from the pathway JSON file."""
    with open(pathway_file) as f:
        data = json.load(f)
    return data["HALLMARK_TNFA_SIGNALING_VIA_NFKB"]["geneSymbols"]


def select_donors(adata: ad.AnnData, n_donors: int = 30) -> list[str]:
    """
    Select donors with varied cell counts and age distribution.

    Strategy:
    - Compute cell counts per donor
    - Divide donors into tertiles by cell count (low/medium/high)
    - Within each tertile, sample to cover age distribution
    """
    # Get donor info
    donor_info = adata.obs.groupby("donor_id").agg({
        "development_stage": "first",  # Age info
    }).reset_index()

    # Add cell counts
    cell_counts = adata.obs["donor_id"].value_counts()
    donor_info["cell_count"] = donor_info["donor_id"].map(cell_counts)

    # Extract numeric age from development_stage (e.g., "25-year-old human stage" -> 25)
    def extract_age(dev_stage):
        if pd.isna(dev_stage):
            return np.nan
        try:
            return int(dev_stage.split("-")[0])
        except (ValueError, IndexError):
            return np.nan

    donor_info["age"] = donor_info["development_stage"].apply(extract_age)

    # Remove donors with missing age
    donor_info = donor_info.dropna(subset=["age"])

    # Ensure age is numeric (not categorical)
    donor_info["age"] = pd.to_numeric(donor_info["age"])

    print(f"Total donors with valid age: {len(donor_info)}")
    print(f"Cell count range: {donor_info['cell_count'].min()} - {donor_info['cell_count'].max()}")
    print(f"Age range: {int(donor_info['age'].min())} - {int(donor_info['age'].max())}")

    # Divide into tertiles by cell count
    donor_info["cell_count_tertile"] = pd.qcut(
        donor_info["cell_count"],
        q=3,
        labels=["low", "medium", "high"]
    )

    # Within each tertile, sample to cover age distribution
    selected_donors = []
    donors_per_tertile = n_donors // 3

    for tertile in ["low", "medium", "high"]:
        tertile_donors = donor_info[donor_info["cell_count_tertile"] == tertile].copy()

        # Sort by age and sample evenly across age range
        tertile_donors = tertile_donors.sort_values("age")

        if len(tertile_donors) <= donors_per_tertile:
            selected = tertile_donors["donor_id"].tolist()
        else:
            # Sample evenly across indices (which are sorted by age)
            indices = np.linspace(0, len(tertile_donors) - 1, donors_per_tertile, dtype=int)
            selected = tertile_donors.iloc[indices]["donor_id"].tolist()

        selected_donors.extend(selected)
        print(f"Tertile {tertile}: selected {len(selected)} donors")

    # Verify selection
    selected_info = donor_info[donor_info["donor_id"].isin(selected_donors)]
    print(f"\nSelected {len(selected_donors)} donors:")
    print(f"  Cell count range: {selected_info['cell_count'].min()} - {selected_info['cell_count'].max()}")
    print(f"  Age range: {selected_info['age'].min()} - {selected_info['age'].max()}")

    return selected_donors


def select_genes(
    adata: ad.AnnData,
    pathway_genes: list[str],
    n_hvg: int = 200,
    n_low_var: int = 100,
) -> list[str]:
    """
    Select genes for the test dataset.

    Args:
        adata: Full AnnData object
        pathway_genes: List of pathway gene symbols to include
        n_hvg: Number of highly variable genes (excluding pathway genes)
        n_low_var: Number of weakly variable genes

    Returns:
        List of selected gene names
    """
    # Get gene names - check for feature_name column or use var index
    if "feature_name" in adata.var.columns:
        gene_names = adata.var["feature_name"].values
        gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    else:
        gene_names = adata.var_names.values
        gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}

    # Find pathway genes present in the dataset
    pathway_genes_present = [g for g in pathway_genes if g in gene_to_idx]
    print(f"Pathway genes in dataset: {len(pathway_genes_present)} / {len(pathway_genes)}")

    # Compute gene variance (on a sample if dataset is large)
    print("Computing gene variance...")
    if adata.n_obs > 10000:
        # Sample cells for variance computation
        sample_idx = np.random.choice(adata.n_obs, 10000, replace=False)
        X_sample = adata.X[sample_idx, :]
    else:
        X_sample = adata.X

    # Handle sparse matrices
    if hasattr(X_sample, "toarray"):
        gene_var = np.var(X_sample.toarray(), axis=0)
    else:
        gene_var = np.var(X_sample, axis=0)

    gene_var = np.asarray(gene_var).flatten()

    # Create DataFrame for gene selection
    gene_df = pd.DataFrame({
        "gene_name": gene_names,
        "variance": gene_var,
        "is_pathway": [g in pathway_genes_present for g in gene_names],
    })

    # Filter out pathway genes for HVG/low-var selection
    non_pathway = gene_df[~gene_df["is_pathway"]].copy()

    # Select highly variable genes (top variance, excluding pathway)
    hvg_candidates = non_pathway.nlargest(n_hvg * 2, "variance")  # Get more candidates
    hvg_selected = hvg_candidates.head(n_hvg)["gene_name"].tolist()
    print(f"Selected {len(hvg_selected)} highly variable genes")

    # Select weakly variable genes (low but non-zero variance)
    # Exclude genes with zero variance
    non_pathway_nonzero = non_pathway[non_pathway["variance"] > 0]
    # Exclude already selected HVG
    non_pathway_remaining = non_pathway_nonzero[~non_pathway_nonzero["gene_name"].isin(hvg_selected)]
    # Get lowest variance genes
    low_var_selected = non_pathway_remaining.nsmallest(n_low_var, "variance")["gene_name"].tolist()
    print(f"Selected {len(low_var_selected)} low variance genes")

    # Combine all selected genes
    all_selected = pathway_genes_present + hvg_selected + low_var_selected
    print(f"Total genes selected: {len(all_selected)}")

    return all_selected


def create_test_dataset(
    source_path: Path,
    output_path: Path,
    pathway_file: Path,
    n_donors: int = 30,
    n_hvg: int = 200,
    n_low_var: int = 100,
):
    """Create the test dataset."""
    print(f"Loading source data from {source_path}...")
    adata = ad.read_h5ad(source_path, backed="r")
    print(f"Source data shape: {adata.shape}")

    # Load pathway genes
    pathway_genes = load_pathway_genes(pathway_file)
    print(f"Loaded {len(pathway_genes)} pathway genes")

    # Select donors
    print("\n--- Selecting donors ---")
    selected_donors = select_donors(adata, n_donors)

    # Select genes
    print("\n--- Selecting genes ---")
    selected_genes = select_genes(adata, pathway_genes, n_hvg, n_low_var)

    # Subset the data
    print("\n--- Creating subset ---")
    # First filter by donors
    donor_mask = adata.obs["donor_id"].isin(selected_donors)

    # Get gene indices - handle feature_name column
    if "feature_name" in adata.var.columns:
        gene_mask = adata.var["feature_name"].isin(selected_genes)
    else:
        gene_mask = adata.var_names.isin(selected_genes)

    # Load into memory and subset
    print("Loading data into memory...")
    adata_memory = adata.to_memory()

    print("Subsetting...")
    adata_subset = adata_memory[donor_mask, gene_mask].copy()
    print(f"Subset shape: {adata_subset.shape}")

    # Verify the subset
    print("\n--- Verification ---")
    print(f"Number of cells: {adata_subset.n_obs}")
    print(f"Number of genes: {adata_subset.n_vars}")
    print(f"Number of donors: {adata_subset.obs['donor_id'].nunique()}")
    if "cell_type" in adata_subset.obs.columns:
        print(f"Number of cell types: {adata_subset.obs['cell_type'].nunique()}")

    # Save
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_subset.write_h5ad(output_path)
    print("Done!")

    return adata_subset


if __name__ == "__main__":
    # Paths
    source_path = Path("/Users/poldrack/data_unsynced/BCBS/immune_aging/dataset-OneK1K_subset-immune_raw.h5ad")
    tests_dir = Path(__file__).parent
    output_path = tests_dir / "data" / "testdata.h5ad"
    pathway_file = tests_dir / "data" / "HALLMARK_TNFA_SIGNALING_VIA_NFKB.v2025.1.Hs.json"

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create the test dataset
    create_test_dataset(
        source_path=source_path,
        output_path=output_path,
        pathway_file=pathway_file,
        n_donors=30,
        n_hvg=200,
        n_low_var=100,
    )
