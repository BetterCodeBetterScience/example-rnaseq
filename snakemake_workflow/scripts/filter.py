"""Snakemake script for Step 2: Data Filtering."""
# ruff: noqa: F821

import logging
import sys
from pathlib import Path

from example_rnaseq.checkpoint import save_checkpoint
from example_rnaseq.data_filtering import run_filtering_pipeline
from example_rnaseq.data_loading import load_lazy_anndata

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
    """Load data and run filtering pipeline."""
    input_file = Path(snakemake.input[0])
    output_file = Path(snakemake.output.checkpoint)

    # Get parameters
    cutoff_percentile = snakemake.params.cutoff_percentile
    min_cells_per_celltype = snakemake.params.min_cells_per_celltype
    percent_donors = snakemake.params.percent_donors
    figure_dir = (
        Path(snakemake.params.figure_dir) if snakemake.params.figure_dir else None
    )

    # Create output directories
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if figure_dir:
        figure_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from: {input_file}")
    adata = load_lazy_anndata(input_file)
    logger.info(f"Loaded dataset: {adata}")

    logger.info("Running filtering pipeline...")
    adata = run_filtering_pipeline(
        adata,
        cutoff_percentile=cutoff_percentile,
        min_cells_per_celltype=min_cells_per_celltype,
        percent_donors=percent_donors,
        figure_dir=figure_dir,
    )
    logger.info(f"Dataset after filtering: {adata}")

    # Save checkpoint
    save_checkpoint(adata, output_file)
    logger.info(f"Saved checkpoint: {output_file}")


if __name__ == "__main__":
    main()
