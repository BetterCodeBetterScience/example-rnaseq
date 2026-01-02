"""Snakemake script for Step 10: Overrepresentation Analysis (Enrichr)."""
# ruff: noqa: F821

import logging
import sys
from pathlib import Path

import pandas as pd

from example_rnaseq.checkpoint import save_checkpoint
from example_rnaseq.overrepresentation_analysis import run_overrepresentation_pipeline

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
    """Run Enrichr overrepresentation analysis for a cell type."""
    de_results_file = Path(snakemake.input.de_results)
    output_enr_up = Path(snakemake.output.enr_up)
    output_enr_down = Path(snakemake.output.enr_down)

    # Get parameters
    cell_type = snakemake.params.cell_type
    gene_sets = snakemake.params.gene_sets
    padj_threshold = snakemake.params.padj_threshold
    n_top = snakemake.params.n_top
    figure_dir = Path(snakemake.params.figure_dir)

    logger.info(f"Running Enrichr for cell type: {cell_type}")

    # Create output directories
    output_enr_up.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Load DE results
    de_results = pd.read_parquet(de_results_file)

    # Run overrepresentation analysis
    enr_up, enr_down = run_overrepresentation_pipeline(
        de_results,
        gene_sets=gene_sets,
        padj_threshold=padj_threshold,
        n_top=n_top,
        figure_dir=figure_dir,
    )

    # Save results
    save_checkpoint(enr_up, output_enr_up)
    save_checkpoint(enr_down, output_enr_down)
    logger.info("Enrichr results saved:")
    logger.info(f"  - enr_up: {output_enr_up}")
    logger.info(f"  - enr_down: {output_enr_down}")


if __name__ == "__main__":
    main()
