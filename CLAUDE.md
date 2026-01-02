## Example RNA-seq workflow

A single-cell RNA-seq analysis workflow implemented using Snakemake. The workflow processes scRNA-seq data through quality control, normalization, dimensionality reduction, and per-cell-type differential expression analysis.

## Project Structure

- `src/example_rnaseq/` - Core analysis modules
- `snakemake_workflow/` - Snakemake workflow definition
- `tests/` - Test suite with unit and integration tests

## Test Dataset

The test dataset is created from the source data using `tests/create_test_data.py`:
- Filters to top 2 most frequent cell types
- Selects 30 donors (28 high cell count, 2 low cell count)
- Selects ~500 genes (pathway genes + HVGs + low variance genes)
- Saved to `tests/data/dataset-test_raw.h5ad`

## Coding guidelines

## Notes for Development

- Think about the problem before generating code.
- Write code that is clean and modular. Prefer shorter functions/methods over longer ones.
- Prefer reliance on widely used packages (such as numpy, pandas, and scikit-learn); avoid unknown packages from Github.
- Do not include *any* code in `__init__.py` files.
- Use pytest for testing.
- Use functions rather than classes for tests. Use pytest fixtures to share resources between tests.