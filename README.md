# Example RNA-seq Workflow

A single-cell RNA sequencing (scRNA-seq) analysis workflow for investigating gene expression changes associated with aging in immune cells. This project processes data from the OneK1K dataset through quality control, normalization, dimensionality reduction, and per-cell-type differential expression analysis.

## Overview

The workflow consists of 11 analysis steps:

**Global Steps (1-7)** - Process the entire dataset:
1. Data Download
2. Data Filtering
3. Quality Control
4. Preprocessing
5. Dimensionality Reduction
6. Clustering
7. Pseudobulking

**Per-Cell-Type Steps (8-11)** - Run for each cell type:
8. Differential Expression
9. Pathway Analysis (GSEA)
10. Overrepresentation Analysis (Enrichr)
11. Predictive Modeling

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd example-rnaseq

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

## Configuration

Create a `.env` file in the project root with your data directory:

```bash
# Copy the example and edit
cp .env.example .env

# Edit .env to set your data directory
DATADIR=/path/to/your/data
```

## Running the Workflow

The workflow is implemented using Snakemake. Run from the `snakemake_workflow` directory:

```bash
cd snakemake_workflow

# Run full workflow (adjust cores as needed)
snakemake --cores 8

# Dry run - see what would be executed
snakemake -n

# Run only preprocessing (steps 1-6)
snakemake --cores 8 preprocessing_only

# Run only through pseudobulking (steps 1-7)
snakemake --cores 8 pseudobulk_only

# Force re-run from a specific step
snakemake --cores 8 --forcerun preprocess
```

### Using a Custom Data File

You can specify a custom input file via the command line:

```bash
snakemake --cores 8 --config raw_data_file=/path/to/your/data.h5ad
```

### Configuration Options

Configuration is in `config/config.yaml`. Override parameters via command line:

```bash
snakemake --cores 8 --config dataset_name=MyDataset min_samples_per_cell_type=20
```

See `WORKFLOW_OVERVIEW.md` in the `snakemake_workflow` directory for detailed step documentation and all configuration options.

## Running Tests

### Prerequisites

First, create the test dataset:

```bash
# Make sure .env is configured with DATADIR
python tests/create_test_data.py

# Or specify a custom output directory
python tests/create_test_data.py -o /path/to/output/dir
```

### Running the Test Suite

```bash
# Run all tests (excluding slow/integration tests)
uv run pytest

# Run all tests including integration tests
uv run pytest -m ""

# Run only unit tests (fast)
uv run pytest -m "not integration and not slow"

# Run only integration tests
uv run pytest -m integration

# Run with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_preprocessing.py

# Run tests with coverage
uv run pytest --cov=example_rnaseq
```

### Test Categories

Tests are organized with pytest markers:

- **Unit tests** (default): Fast tests using synthetic data
- **Integration tests** (`-m integration`): Tests that use the real test dataset
- **Slow tests** (`-m slow`): Long-running tests

## Project Structure

```
example-rnaseq/
├── src/example_rnaseq/       # Core analysis modules
│   ├── checkpoint.py         # Checkpoint save/load utilities
│   ├── clustering.py         # Clustering functions
│   ├── data_filtering.py     # Data filtering functions
│   ├── data_loading.py       # Data I/O functions
│   ├── differential_expression.py
│   ├── dimensionality_reduction.py
│   ├── overrepresentation_analysis.py
│   ├── pathway_analysis.py
│   ├── predictive_modeling.py
│   ├── preprocessing.py
│   ├── pseudobulk.py
│   └── quality_control.py
├── snakemake_workflow/       # Snakemake workflow
│   ├── Snakefile             # Main workflow definition
│   ├── config/config.yaml    # Configuration
│   ├── rules/                # Modular rule files
│   ├── scripts/              # Step scripts
│   └── WORKFLOW_OVERVIEW.md  # Detailed documentation
├── tests/                    # Test suite
│   ├── conftest.py           # Shared fixtures
│   ├── create_test_data.py   # Test data generation
│   └── test_*.py             # Test modules
├── .env.example              # Environment template
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Output Structure

When running the workflow, outputs are organized as:

```
{working_directory}/
├── checkpoints/              # Intermediate h5ad files
├── figures/                  # Visualization outputs
├── results/
│   ├── per_cell_type/        # Per-cell-type results
│   │   └── {cell_type}/
│   │       ├── de_results.parquet
│   │       ├── gsea_results.pkl
│   │       ├── enrichr_up.pkl
│   │       ├── enrichr_down.pkl
│   │       └── prediction_results.pkl
│   └── workflow_complete.txt
└── logs/                     # Execution logs
```

## Troubleshooting

### Memory Issues

The full OneK1K dataset is large (~1.2M cells). If you encounter memory issues:

- Use a machine with at least 64GB RAM
- Reduce `--cores` to limit parallel jobs
- Use the test dataset for development

### Missing Output Files

If the workflow fails with missing output files, check the log files in `logs/` for detailed error messages.

### Environment Variable Not Set

If you see "DATADIR environment variable not set":

```bash
# Create .env file
echo "DATADIR=/path/to/your/data" > .env
```

## License

See LICENSE file for details.
