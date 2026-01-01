## Example RNA-seq workflow

This is a set of implementations of an analysis workflow for single-cell RNA-seq data analysis. These are meant to exemplify different ways of building a workflow.

The main goal of this development project is to develop a testing framework for this workflow.  This will involve two main steps:

- develop unit tests for the functions defined within src/example_rnaseq
- develop integration tests for the snakemake workflow defined in snakemake_workflow

Some of these tests can be performed using automatically generated data. However, the integration tests will require a test dataset. this should be based on the actual data that can be found at /Users/poldrack/data_unsynced/BCBS/immune_aging/dataset-OneK1K_subset-immune_raw.h5ad.  To minimize the size of this dataset, we should first select a subset of 30 donors from the dataset.  we should look for donors that vary in the number of cells, with some having high numbers and some having low numbers. The donors should also vary in age so that the subset covers the entire distribution of ages in the dataset. then we should select a subset of about 500 genes.  these should include:
- a set of genes from a pathway (TNF-alpha Signaling via NF-KB) known to be associated with aging, found in tests/data/HALLMARK_TNFA_SIGNALING_VIA_NFKB.v2025.1.Hs.json
- a set of about 200 other highly variable genes
- a set of about 100 weakly variable genes

This dataset should be saved to tests/data/testdata.h5ad.

## Coding guidelines

## Notes for Development

- Think about the problem before generating code.
- Write code that is clean and modular. Prefer shorter functions/methods over longer ones.
- Prefer reliance on widely used packages (such as numpy, pandas, and scikit-learn); avoid unknown packages from Github.
- Do not include *any* code in `__init__.py` files.
- Use pytest for testing.
- Use functions rather than classes for tests. Use pytest fixtures to share resources between tests.