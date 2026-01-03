## Problems to be fixed

Open problems marked with [ ]
Fixed problems marked with [x]
    - also annotate each fixed problem with a description of the soultion

[x] Currently it appears that files of unknown type are being treated as pickle files (e.g. see line 85 in test_checkpoint.py).  This is inappropriate - files of unknown type should raise an appropriate exception
    - Solution: Modified `get_file_type()` in checkpoint.py to only recognize .h5ad, .parquet, and .pkl extensions explicitly. Unknown extensions now raise a ValueError with a helpful message. Updated test to expect this exception.

[x] The function "extract_age_from_development_stage" seems to be misnamed - it actually returns a new dataset with the age variable added.  rename to better describe the actual intent of the function.
    - Solution: Renamed function to `add_age_from_development_stage()` and updated docstring to clarify it adds an 'age' column. Updated all references in differential_expression.py and test_differential_expression.py.

[x] Create a Makefile in the main directory to run the different test sets
    - Solution: Created Makefile with targets: test (default, unit tests only), test-unit, test-integration, test-slow, test-all, test-coverage, clean, and help.

[x] Add generation of coverage report using pytest-cov when full test suite is run
    - Solution: Added `test-coverage` target to Makefile that runs pytest with --cov=src/example_rnaseq --cov-report=term-missing --cov-report=html.

[x] It is not clear to me that TestRunHarmonyIntegration::test_runs_without_error is actually testing whether Harmony runs. it seems that it could return X_pca (i.e. harmony didn't run) and still pass.  It also seems like it needs to check whether harmonypy is actually installed.
    - Solution: Added fixture to check if harmonypy is installed, tests now skip if not available. Tests now verify that when Harmony is available: (1) use_rep is "X_pca_harmony", (2) harmony output differs from original PCA, (3) shapes are correct. Added test for fallback behavior when harmony is unavailable.

[x] It is not clear to me that TestPseudobulkPipelineIntegration::test_pseudobulk_pipeline_runs is actually testing the outputs sufficiently.  for example, would it fail if the input data were simply returned without pseudobulking?  If the outputs of run_pseudobulk_pipeline are being properly tested elsewhere then let me know.
    - Solution: Enhanced test to verify: (1) n_obs <= donors x cell_types, (2) sum of n_cells equals input cell count, (3) total counts are preserved between input and output, (4) all samples have n_cells >= 1.

[x] I think TestPrepareEnrichrPlotData::test_combines_up_and_down should be a bit more thorough.  e.g. it should test that the right number of up and down genes are included in the combined result.
    - Solution: Added assertions to verify exactly 5 upregulated and 5 downregulated pathways are in the combined result.

[x] In multiple places the tests check for proper sorting of a list using code like "all(values[i] >= values[i + 1] for i in range(len(values) - 1))".  I think this should be extracted into a utility function is_sorted() that takes the list/array as an input along with a flag for ascending/descending and returns a Boolean
    - Solution: Added `is_sorted(values, descending=True)` utility function to conftest.py. Updated test_pathway_analysis.py and test_differential_expression.py to use this function.

[x] TestCreatePseudobulk::test_includes_metadata should be more aggressive.  in particular, it shoudl check to make sure that the metadata fields (development_stage and sex) match for a given donor ID between adata_for_pseudobulk and pb
    - Solution: Enhanced test to verify for each donor_id that development_stage and sex values in the pseudobulk output match the corresponding values in the source data.

[x] Some of the tests in test_quality_control.py::TestApplyQcFilters seem problematic, because they allow a test to pass even if the filtering is done incorrectly.  for example, test_filters_high_gene_cells computes a max_genes value based on a quantile of the data - unless that quantile is below the minimum value for hb_pct, this filtering operation shoudl always return a dataset with at least one observation.  however, the test wraps its assertion in "if filtered.n_obs > 0:" which means that it would not actually catch cases where an empty filtered dataset was returned in error.
    - Solution: Removed the `if filtered.n_obs > 0:` guards and replaced them with explicit assertions that filtered.n_obs > 0 with informative error messages. Tests now fail if filtering incorrectly produces an empty result.

[x] The `make test-coverage` target skips integration tests because the test data file (tests/data/dataset-test_raw.h5ad) doesn't exist. Tests that require this file call pytest.skip() at runtime, resulting in incomplete coverage.
    - Solution: Updated Makefile to add the test data file as a prerequisite for test-integration, test-slow, test-all, and test-coverage targets. Added a `test-data` target that creates the test dataset by running `tests/create_test_data.py`. The script reads DATADIR from the .env file to locate the source data. Also added `clean-test-data` target to force regeneration of the test data. Make's dependency tracking ensures the test data is only created once (when the file doesn't exist).

[x] TestPseudobulkPipelineIntegration::test_pseudobulk_pipeline_runs fails with "Total counts should be preserved: input=93483552.0, output=93483547" - a difference of 5 counts out of ~93 million.
    - Solution: The absolute tolerance of `< 1` was too strict. Small floating-point discrepancies occur during sparse matrix aggregation and the `np.round().astype(int)` step in pseudobulk creation. Changed to a relative tolerance of `< 1e-5` (0.001%), which easily accommodates these rounding differences while still catching any real aggregation errors.

[x] Two tests are still being skipped:
    - tests/test_snakemake_workflow.py::TestSnakemakeDryRun::test_snakemake_dryrun
    - tests/test_pathway_analysis.py::TestGseaIntegration::test_run_gsea_prerank
Please determine why these are being skipped and ensure that they are included in the full test set.
    - Solution: The snakemake dryrun test was failing because Snakemake requires `--cores` flag when rules use thread calculations. Added `--cores 1` to the snakemake command. The GSEA prerank test was skipping because it used synthetic gene names ("GENE0", "GENE1") that don't match real gene sets in MSigDB. Created a new fixture `sample_deseq_results_real_genes` that uses actual human gene symbols from the HALLMARK_TNFA_SIGNALING_VIA_NFKB pathway plus common housekeeping genes, and updated the test to use this fixture with `min_size=5` to accommodate the smaller gene set.

[x] test coverage is only 69% at present for the modules in src/example_rnaseq.  Please examine the coverage report and identify where there are any important sections of code that are not currently being tested that should be added to the tests, and then add them.
    - Solution: Added comprehensive tests for previously untested modules:
      1. `execution_log.py` (0% → 98%): Added test_execution_log.py with 19 tests covering StepRecord, ExecutionLog, create_execution_log, and serialize_parameters functions.
      2. `checkpoint.py` (70% → 96%): Added tests for run_with_checkpoint_multi and checkpoint-with-logging integration (10 new tests).
      3. `overrepresentation_analysis.py` (51% → 75%): Added tests for plot_enrichr_results and plot_empty_enrichr_figure (4 new tests).
      4. `pathway_analysis.py` (48% → 84%): Added tests for get_gsea_top_terms and plot_gsea_results (5 new tests).
      Overall coverage improved from 69% to 88% with 38 new tests added (156 → 194 total tests).

[x] Two warnings occur during testing that appear potentially consequential, please investigate these and determine whether they require changes to the code.
    - tests/test_integration.py::TestQcPipelineIntegration::test_qc_pipeline_runs
  /Users/poldrack/Dropbox/code/BetterCodeBetterScience/example-rnaseq/.venv/lib/python3.13/site-packages/legacy_api_wrap/__init__.py:88: UserWarning: `flavor='seurat_v3'` expects raw count data, but non-integers were found.
    return fn(*args_all, **kw)

    - tests/test_integration.py::TestQcPipelineIntegration::test_qc_pipeline_runs
  /Users/poldrack/Dropbox/code/BetterCodeBetterScience/example-rnaseq/src/example_rnaseq/quality_control.py:369: ImplicitModificationWarning: Setting element `.layers['counts']` of view, initializing view as actual.
    adata.layers["counts"] = adata.X.copy()
    - Solution: Fixed both warnings:
      1. `seurat_v3` warning: In `compute_umap_for_qc()`, stored raw counts in a `counts` layer before normalization, then passed `layer="counts"` to `sc.pp.highly_variable_genes()` so seurat_v3 operates on raw counts as expected.
      2. `ImplicitModificationWarning`: Added `.copy()` after `filter_doublets()` to ensure we're working with an actual AnnData object rather than a view before modifying layers.
      3. Also updated both `quality_control.py` and `preprocessing.py` to use the new `mask_var` parameter instead of the deprecated `use_highly_variable` parameter in PCA calls.

[x] tests/test_differential_expression.py::TestDeseq2Integration::test_run_deseq2_on_minimal_data is giving warnings on a couple of tests:
    - tests/test_differential_expression.py::TestDeseq2Integration::test_run_deseq2_on_minimal_data
  /Users/poldrack/Dropbox/code/BetterCodeBetterScience/example-rnaseq/.venv/lib/python3.13/site-packages/pydeseq2/dds.py:820: UserWarning: The dispersion trend curve fitting did not converge. Switching to a mean-based dispersion trend.
    self._fit_parametric_dispersion_trend(vst)
- tests/test_differential_expression.py::TestDeseq2Integration::test_run_deseq2_on_minimal_data
  /Users/poldrack/Dropbox/code/BetterCodeBetterScience/example-rnaseq/.venv/lib/python3.13/site-packages/pydeseq2/dds.py:548: UserWarning: As the residual degrees of freedom is less than 3, the distribution of log dispersions is especially asymmetric and likely to be poorly estimated by the MAD.
    self.fit_dispersion_prior()

These are expected given the small test dataset size.  please capture these warnings so that they don't appear in the test summary.
    - Solution: Added `@pytest.mark.filterwarnings` decorators to the test method to suppress these expected warnings from pydeseq2. The warnings are documented in the test docstring explaining they occur due to the small test dataset size.