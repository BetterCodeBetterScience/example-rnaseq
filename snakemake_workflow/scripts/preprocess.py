"""Snakemake script for Step 4: Preprocessing."""
# ruff: noqa: F821

import logging
import sys
from pathlib import Path

from example_rnaseq.checkpoint import load_checkpoint, save_checkpoint
from example_rnaseq.preprocessing import (
    filter_nuisance_genes_from_hvg,
    log_transform,
    normalize_counts,
    run_pca,
    select_highly_variable_genes,
)

# Configure logging with immediate flush (unbuffered file handler)
file_handler = logging.FileHandler(snakemake.log[0])
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler],
)
logger = logging.getLogger(__name__)


def log_and_flush(message: str) -> None:
    """Log a message and flush handlers immediately."""
    logger.info(message)
    for handler in logger.handlers + logging.getLogger().handlers:
        handler.flush()


def main():
    """Run preprocessing pipeline with detailed logging."""
    input_file = Path(snakemake.input[0])
    output_file = Path(snakemake.output[0])

    # Get parameters
    target_sum = snakemake.params.target_sum
    n_top_genes = snakemake.params.n_top_genes
    batch_key = snakemake.params.batch_key

    log_and_flush(f"Loading data from: {input_file}")
    adata = load_checkpoint(input_file)
    log_and_flush(f"Loaded dataset: {adata.n_obs} cells x {adata.n_vars} genes")
    log_and_flush(f"Memory usage: {adata.X.nbytes / 1e9:.2f} GB for X matrix")

    # Step 1: Normalize
    log_and_flush("Step 1/5: Normalizing counts...")
    adata = normalize_counts(adata, target_sum)
    log_and_flush("Normalization complete")

    # Step 2: Log transform
    log_and_flush("Step 2/5: Log transforming...")
    adata = log_transform(adata)
    log_and_flush("Log transformation complete")

    # Step 3: HVG selection
    log_and_flush(f"Step 3/5: Selecting {n_top_genes} highly variable genes...")
    adata = select_highly_variable_genes(adata, n_top_genes, batch_key)
    log_and_flush(f"HVG selection complete: {adata.var['highly_variable'].sum()} genes")

    # Step 4: Filter nuisance genes
    log_and_flush("Step 4/5: Filtering nuisance genes from HVG list...")
    adata = filter_nuisance_genes_from_hvg(adata)
    log_and_flush("Nuisance gene filtering complete")

    # Step 5: PCA
    log_and_flush("Step 5/5: Running PCA (this may take a while)...")
    adata = run_pca(adata)
    log_and_flush("PCA complete")

    # Remove counts layer after preprocessing to save space
    if "counts" in adata.layers:
        del adata.layers["counts"]
        log_and_flush("Removed counts layer to save checkpoint space")

    # Save checkpoint
    log_and_flush(f"Saving checkpoint to: {output_file}")
    save_checkpoint(adata, output_file)
    log_and_flush(f"Saved checkpoint: {output_file}")


if __name__ == "__main__":
    main()
