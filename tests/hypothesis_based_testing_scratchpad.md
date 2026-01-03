## Problems to be fixed

Open problems marked with [ ]
Fixed problems marked with [x]
    - also annotate each fixed problem with a description of the soultion

[x] I would like to implement property-based testing for the functions where this might be appropriate, using the `hypothesis` library.  Please examine the code in src/example_rnaseq and identify functions where property based testing would be appropriate to test for robustness of functions to input characteristics.
    - Solution: Created tests/test_hypothesis.py with 15 property-based tests covering:
      1. **checkpoint.py**:
         - `bids_checkpoint_name`/`parse_bids_checkpoint_name`: Roundtrip property (parsing generated names recovers original inputs)
         - `hash_parameters`: Determinism (same inputs = same hash), hash length (8 chars), hex validity
      2. **execution_log.py**:
         - `serialize_parameters`: Output is always JSON-serializable, Path objects become strings, numpy arrays become lists
      3. **overrepresentation_analysis.py**:
         - `get_significant_gene_lists`: Up genes have positive log2FC, down genes have negative log2FC, all genes pass padj threshold, up/down sets are disjoint
      4. **pathway_analysis.py**:
         - `prepare_ranked_list`: Output is sorted descending by stat, no NaN values in output, only stat column retained
    - These tests use Hypothesis strategies to generate diverse inputs and verify invariants that should hold regardless of input values.
    - **Additional edge case tests (15 more tests)**:
      5. **Numerical threshold edge cases**:
         - `padj_threshold=0`: returns empty (nothing < 0)
         - `padj_threshold=1`: returns all genes
         - Negative thresholds: returns empty
         - Very small thresholds (1e-300): no numerical issues
      6. **Pseudobulk filtering edge cases**:
         - `min_cells=0`: returns all samples
         - `min_cells<0`: returns all samples
         - Very large `min_cells`: returns empty
         - All remaining samples meet threshold invariant
         - Result is always a subset of input
      7. **Log transform edge cases**:
         - Zero p-value handling (found and fixed bug!)
         - Very small p-values
         - log_p always non-negative (within floating point tolerance)
      8. **NaN and Infinity handling**:
         - NaN padj values are excluded from gene lists
         - Infinity in stat values sorts correctly
    - **Bug found and fixed**: `prepare_enrichr_plot_data` was computing `-log10(0) = inf` for zero p-values. Fixed by adding epsilon: `+ 1e-10` (same approach as GSEA code).