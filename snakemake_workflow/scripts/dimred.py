"""Snakemake script for Step 5: Dimensionality Reduction."""
# ruff: noqa: F821

import logging
import os
import sys
from pathlib import Path

# Set thread count for numba/pynndescent before importing scanpy
os.environ["NUMBA_NUM_THREADS"] = str(snakemake.threads)
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)

from example_rnaseq.checkpoint import load_checkpoint, save_checkpoint
from example_rnaseq.dimensionality_reduction import run_dimensionality_reduction_pipeline

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
    """Run dimensionality reduction pipeline."""
    logger.info(f"Running with {snakemake.threads} threads")

    input_file = Path(snakemake.input[0])
    output_file = Path(snakemake.output.checkpoint)

    # Get parameters
    batch_key = snakemake.params.batch_key
    n_neighbors = snakemake.params.n_neighbors
    n_pcs = snakemake.params.n_pcs
    figure_dir = (
        Path(snakemake.params.figure_dir) if snakemake.params.figure_dir else None
    )

    # Create output directories
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if figure_dir:
        figure_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from: {input_file}")
    adata = load_checkpoint(input_file)
    logger.info(f"Loaded dataset: {adata}")

    logger.info("Running dimensionality reduction pipeline...")
    adata = run_dimensionality_reduction_pipeline(
        adata,
        batch_key=batch_key,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        figure_dir=figure_dir,
    )

    # Save checkpoint
    save_checkpoint(adata, output_file)
    logger.info(f"Saved checkpoint: {output_file}")


if __name__ == "__main__":
    main()
