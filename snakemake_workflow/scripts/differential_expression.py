"""Snakemake script for Step 8: Differential Expression."""
# ruff: noqa: F821

import json
import logging
import sys
from pathlib import Path

from example_rnaseq.checkpoint import load_checkpoint, save_checkpoint
from example_rnaseq.differential_expression import run_differential_expression_pipeline

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


def unsanitize_cell_type(sanitized: str, cell_types_file: Path) -> str:
    """Convert sanitized cell type back to original name."""
    with open(cell_types_file) as f:
        data = json.load(f)
    # Reverse lookup
    for original, sanitized_name in data["sanitized_names"].items():
        if sanitized_name == sanitized:
            return original
    raise ValueError(f"Unknown sanitized cell type: {sanitized}")


def main():
    """Run differential expression for a cell type."""
    pseudobulk_file = Path(snakemake.input.pseudobulk)
    var_to_feature_file = Path(snakemake.input.var_to_feature)
    cell_types_file = Path(snakemake.input.cell_types)

    output_stat_res = Path(snakemake.output.stat_res)
    output_de_results = Path(snakemake.output.de_results)
    output_counts_df = Path(snakemake.output.counts_df)

    # Get parameters
    sanitized_cell_type = snakemake.params.cell_type
    design_factors = snakemake.params.design_factors
    n_cpus = snakemake.params.n_cpus

    # Get original cell type name
    cell_type = unsanitize_cell_type(sanitized_cell_type, cell_types_file)

    logger.info(f"Running DE for cell type: {cell_type}")
    logger.info(f"(sanitized: {sanitized_cell_type})")

    # Create output directory
    output_stat_res.parent.mkdir(parents=True, exist_ok=True)

    # Load pseudobulk data
    pb_adata = load_checkpoint(pseudobulk_file)

    # Load var_to_feature mapping
    with open(var_to_feature_file) as f:
        var_to_feature = json.load(f)

    # Run differential expression
    stat_res, de_results, counts_df = run_differential_expression_pipeline(
        pb_adata,
        cell_type=cell_type,
        design_factors=design_factors,
        var_to_feature=var_to_feature,
        n_cpus=n_cpus,
    )

    # Save outputs
    save_checkpoint(stat_res, output_stat_res)
    de_results.to_parquet(output_de_results)
    counts_df.to_parquet(output_counts_df)

    logger.info(f"DE results saved for: {cell_type}")
    logger.info(f"  - stat_res: {output_stat_res}")
    logger.info(f"  - de_results: {output_de_results}")
    logger.info(f"  - counts_df: {output_counts_df}")


if __name__ == "__main__":
    main()
