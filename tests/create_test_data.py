"""
Script to create a minimal test dataset for the RNA-seq workflow tests.

This script:
1. Loads the full dataset from the source h5ad file
2. Filters to keep only the 2 most frequent cell types
3. Selects 30 donors: 28 with high cell counts, 2 with low cell counts
4. Selects ~500 genes:
   - 191 genes from TNF-alpha signaling pathway
   - ~200 highly variable genes
   - ~100 weakly variable genes
5. Saves the subset to tests/data/dataset-test_raw.h5ad

Requires DATADIR to be set in .env file pointing to the base data directory.
"""

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from dotenv import load_dotenv


def load_pathway_genes(pathway_file: Path) -> list[str]:
    """Load gene symbols from the pathway JSON file."""
    with open(pathway_file) as f:
        data = json.load(f)
    return data["HALLMARK_TNFA_SIGNALING_VIA_NFKB"]["geneSymbols"]


def filter_top_cell_types(adata: ad.AnnData, n_top: int = 2) -> list[str]:
    """
    Get the top N most frequent cell types.

    Parameters
    ----------
    adata : AnnData
        AnnData object with cell_type column in obs
    n_top : int
        Number of top cell types to keep

    Returns
    -------
    list[str]
        List of top cell type names
    """
    cell_type_counts = adata.obs["cell_type"].value_counts()
    top_cell_types = cell_type_counts.head(n_top).index.tolist()

    print(f"Top {n_top} cell types by frequency:")
    for ct in top_cell_types:
        print(f"  {ct}: {cell_type_counts[ct]} cells")

    return top_cell_types


def select_donors(
    adata: ad.AnnData,
    n_donors: int = 30,
    n_low_count: int = 2,
) -> list[str]:
    """
    Select donors with mostly high cell counts.

    Strategy:
    - Compute cell counts per donor
    - Select n_low_count donors from the bottom tertile (low cell counts)
    - Select remaining donors from top cell count donors
    - Within each group, sample to cover age distribution

    Parameters
    ----------
    adata : AnnData
        AnnData object
    n_donors : int
        Total number of donors to select
    n_low_count : int
        Number of donors with low cell counts to include

    Returns
    -------
    list[str]
        List of selected donor IDs
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

    # Sort by cell count
    donor_info_sorted = donor_info.sort_values("cell_count", ascending=True)

    # Select low cell count donors (from bottom third)
    n_bottom_third = len(donor_info) // 3
    low_count_candidates = donor_info_sorted.head(n_bottom_third).copy()
    # Sample evenly across age distribution
    low_count_candidates = low_count_candidates.sort_values("age")
    if len(low_count_candidates) <= n_low_count:
        low_count_donors = low_count_candidates["donor_id"].tolist()
    else:
        indices = np.linspace(0, len(low_count_candidates) - 1, n_low_count, dtype=int)
        low_count_donors = low_count_candidates.iloc[indices]["donor_id"].tolist()

    print(f"Selected {len(low_count_donors)} low cell count donors")

    # Select high cell count donors (from remaining, prioritizing high counts)
    n_high_count = n_donors - len(low_count_donors)
    remaining = donor_info[~donor_info["donor_id"].isin(low_count_donors)].copy()
    # Sort by cell count descending, then sample across age
    high_count_candidates = remaining.nlargest(n_high_count * 2, "cell_count")
    high_count_candidates = high_count_candidates.sort_values("age")

    if len(high_count_candidates) <= n_high_count:
        high_count_donors = high_count_candidates["donor_id"].tolist()
    else:
        indices = np.linspace(0, len(high_count_candidates) - 1, n_high_count, dtype=int)
        high_count_donors = high_count_candidates.iloc[indices]["donor_id"].tolist()

    print(f"Selected {len(high_count_donors)} high cell count donors")

    selected_donors = low_count_donors + high_count_donors

    # Verify selection
    selected_info = donor_info[donor_info["donor_id"].isin(selected_donors)]
    low_info = donor_info[donor_info["donor_id"].isin(low_count_donors)]
    high_info = donor_info[donor_info["donor_id"].isin(high_count_donors)]

    print(f"\nSelected {len(selected_donors)} donors total:")
    print(f"  Low count donors ({len(low_count_donors)}): "
          f"cell counts {low_info['cell_count'].min()} - {low_info['cell_count'].max()}")
    print(f"  High count donors ({len(high_count_donors)}): "
          f"cell counts {high_info['cell_count'].min()} - {high_info['cell_count'].max()}")
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
    n_low_count_donors: int = 2,
    n_top_cell_types: int = 2,
    n_hvg: int = 200,
    n_low_var: int = 100,
):
    """Create the test dataset.

    Parameters
    ----------
    source_path : Path
        Path to source h5ad file
    output_path : Path
        Path to save test dataset
    pathway_file : Path
        Path to pathway genes JSON file
    n_donors : int
        Total number of donors to select
    n_low_count_donors : int
        Number of donors with low cell counts to include
    n_top_cell_types : int
        Number of most frequent cell types to keep
    n_hvg : int
        Number of highly variable genes to select
    n_low_var : int
        Number of low variance genes to select
    """
    print(f"Loading source data from {source_path}...")
    adata = ad.read_h5ad(source_path, backed="r")
    print(f"Source data shape: {adata.shape}")

    # Load pathway genes
    pathway_genes = load_pathway_genes(pathway_file)
    print(f"Loaded {len(pathway_genes)} pathway genes")

    # Filter to top cell types first
    print("\n--- Filtering to top cell types ---")
    top_cell_types = filter_top_cell_types(adata, n_top_cell_types)
    cell_type_mask = adata.obs["cell_type"].isin(top_cell_types)

    # Load into memory for filtering
    print("Loading data into memory for cell type filtering...")
    adata_memory = adata.to_memory()
    adata_filtered = adata_memory[cell_type_mask].copy()
    print(f"After cell type filtering: {adata_filtered.shape}")

    # Select donors (based on filtered data)
    print("\n--- Selecting donors ---")
    selected_donors = select_donors(adata_filtered, n_donors, n_low_count_donors)

    # Select genes (based on filtered data)
    print("\n--- Selecting genes ---")
    selected_genes = select_genes(adata_filtered, pathway_genes, n_hvg, n_low_var)

    # Subset the data
    print("\n--- Creating final subset ---")
    donor_mask = adata_filtered.obs["donor_id"].isin(selected_donors)

    # Get gene indices - handle feature_name column
    if "feature_name" in adata_filtered.var.columns:
        gene_mask = adata_filtered.var["feature_name"].isin(selected_genes)
    else:
        gene_mask = adata_filtered.var_names.isin(selected_genes)

    print("Subsetting...")
    adata_subset = adata_filtered[donor_mask, gene_mask].copy()
    print(f"Subset shape: {adata_subset.shape}")

    # Verify the subset
    print("\n--- Verification ---")
    print(f"Number of cells: {adata_subset.n_obs}")
    print(f"Number of genes: {adata_subset.n_vars}")
    print(f"Number of donors: {adata_subset.obs['donor_id'].nunique()}")
    if "cell_type" in adata_subset.obs.columns:
        print(f"Cell types: {adata_subset.obs['cell_type'].unique().tolist()}")

    # Save with compression
    print(f"\nSaving to {output_path} (with gzip compression)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_subset.write_h5ad(output_path, compression="gzip")
    print("Done!")

    return adata_subset


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Get data directory from environment
    datadir = os.getenv("DATADIR")
    if not datadir:
        raise ValueError(
            "DATADIR environment variable not set. "
            "Please create a .env file with DATADIR=/path/to/your/data"
        )

    # Paths
    source_path = Path(datadir) / "immune_aging" / "dataset-OneK1K_subset-immune_raw.h5ad"
    if not source_path.exists():
        raise FileNotFoundError(f"Source data not found: {source_path}")

    tests_dir = Path(__file__).parent
    output_path = tests_dir / "data" / "dataset-test_raw.h5ad"
    pathway_file = tests_dir / "data" / "HALLMARK_TNFA_SIGNALING_VIA_NFKB.v2025.1.Hs.json"

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create the test dataset
    create_test_dataset(
        source_path=source_path,
        output_path=output_path,
        pathway_file=pathway_file,
        n_donors=30,
        n_low_count_donors=2,
        n_top_cell_types=2,
        n_hvg=200,
        n_low_var=100,
    )
