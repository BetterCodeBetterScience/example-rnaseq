## Problems to be fixed

Open problems marked with [ ]
Fixed problems marked with [x]
    - also annotate each fixed problem with a description of the soultion

[ ] Currently it appears that files of unknown type are being treated as pickle files (e.g. see line 85 in test_checkpoint.py).  This is inappropriate - files of unknown type should raise an appropriate exception

[ ] The function "extract_age_from_development_stage" seems to be misnamed - it actually returns a new dataset with the age variable added.  rename to better describe the actual intent of the function.

[ ] Create a Makefile in the main directory to run the different test sets

[ ] Add generation of coverage report using pytest-cov when full test suite is run

[ ] It is not clear to me that TestRunHarmonyIntegration::test_runs_without_error is actually testing whether Harmony runs. it seems that it could return X_pca (i.e. harmony didn't run) and still pass.  It also seems like it needs to check whether harmonypy is actually installed.

[ ] It is not clear to me that TestPseudobulkPipelineIntegration::test_pseudobulk_pipeline_runs is actually testing the outputs sufficiently.  for example, would it fail if the input data were simply returned without pseudobulking?  If the outputs of run_pseudobulk_pipeline are being properly tested elsewhere then let me know.

[ ] I think TestPrepareEnrichrPlotData::test_combines_up_and_down should be a bit more thorough.  e.g. it should test that the right number of up and down genes are included in the combined result.

[ ] In multiple places the tests check for proper sorting of a list using code like "all(values[i] >= values[i + 1] for i in range(len(values) - 1))".  I think this should be extracted into a utility function is_sorted() that takes the list/array as an input along with a flag for ascending/descending and returns a Boolean

[ ] TestCreatePseudobulk::test_includes_metadata should be more aggressive.  in particular, it shoudl check to make sure that the metadata fields (development_stage and sex) match for a given donor ID between adata_for_pseudobulk and pb

[ ] Some of the tests in test_quality_control.py::TestApplyQcFilters seem problematic, because they allow a test to pass even if the filtering is done incorrectly.  for example, test_filters_high_gene_cells computes a max_genes value based on a quantile of the data - unless that quantile is below the minimum value for hb_pct, this filtering operation shoudl always return a dataset with at least one observation.  however, the test wraps its assertion in "if filtered.n_obs > 0:" which means that it would not actually catch cases where an empty filtered dataset was returned in error.


