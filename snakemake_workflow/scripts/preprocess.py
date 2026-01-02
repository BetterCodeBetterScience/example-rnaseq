"""Snakemake script for Step 4: Preprocessing."""
# ruff: noqa: F821

import logging
import sys
from pathlib import Path

from example_rnaseq.checkpoint import load_checkpoint, save_checkpoint
from example_rnaseq.preprocessing import run_preprocessing_pipeline

# Configure logging to write to both log file and stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(snakemake.log[0]),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Run preprocessing pipeline."""
    input_file = Path(snakemake.input[0])
    output_file = Path(snakemake.output[0])

    # Get parameters
    target_sum = snakemake.params.target_sum
    n_top_genes = snakemake.params.n_top_genes
    batch_key = snakemake.params.batch_key

    logger.info(f"Loading data from: {input_file}")
    adata = load_checkpoint(input_file)
    logger.info(f"Loaded dataset: {adata}")

    logger.info("Running preprocessing pipeline...")
    adata = run_preprocessing_pipeline(
        adata,
        target_sum=target_sum,
        n_top_genes=n_top_genes,
        batch_key=batch_key,
    )

    # Remove counts layer after preprocessing to save space
    if "counts" in adata.layers:
        del adata.layers["counts"]
        logger.info("Removed counts layer to save checkpoint space")

    # Save checkpoint
    save_checkpoint(adata, output_file)
    logger.info(f"Saved checkpoint: {output_file}")


if __name__ == "__main__":
    main()
