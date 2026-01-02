"""Snakemake script for Step 9: Pathway Analysis (GSEA)."""
# ruff: noqa: F821

import logging
import sys
from pathlib import Path

import pandas as pd

from example_rnaseq.checkpoint import save_checkpoint
from example_rnaseq.pathway_analysis import run_gsea_pipeline

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
    """Run GSEA pathway analysis for a cell type."""
    de_results_file = Path(snakemake.input.de_results)
    output_file = Path(snakemake.output.gsea_results)

    # Get parameters
    cell_type = snakemake.params.cell_type
    gene_sets = snakemake.params.gene_sets
    n_top = snakemake.params.n_top
    figure_dir = Path(snakemake.params.figure_dir)

    logger.info(f"Running GSEA for cell type: {cell_type}")

    # Create output directories
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Load DE results
    de_results = pd.read_parquet(de_results_file)

    # Run GSEA
    gsea_results = run_gsea_pipeline(
        de_results,
        gene_sets=gene_sets,
        n_top=n_top,
        figure_dir=figure_dir,
    )

    # Save results
    save_checkpoint(gsea_results, output_file)
    logger.info(f"GSEA results saved: {output_file}")


if __name__ == "__main__":
    main()
