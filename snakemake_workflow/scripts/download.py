"""Snakemake script for Step 1: Data Download."""
# ruff: noqa: F821

import logging
import sys
from pathlib import Path

from example_rnaseq.data_loading import download_data

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
    """Download data file if it doesn't exist."""
    datafile = Path(snakemake.output[0])
    url = snakemake.params.url

    logger.info(f"Downloading data from: {url}")
    logger.info(f"Output file: {datafile}")

    download_data(datafile, url)

    logger.info(f"Download complete: {datafile}")


if __name__ == "__main__":
    main()
