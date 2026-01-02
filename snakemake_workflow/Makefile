include ../.env
export

OUTPUT_DIR := $(DATADIR)/immune_aging/wf_snakemake/
TEST_DIR := $(DATADIR)/test/wf_snakemake/

# get ncores - 1 on mac or linux
CORES := $(shell echo $$(( $$(nproc 2>/dev/null || sysctl -n hw.ncpu) - 1 )))

# Generate rule dependency graph
rulegraph:
	snakemake --rulegraph --config datadir=$(OUTPUT_DIR) --cores 2 | dot -Tpng > rulegraph.png

# Run the full workflow
run:
	uv run snakemake -d $(OUTPUT_DIR) --snakefile ./Snakefile --cores $(CORES) --config datadir=$(OUTPUT_DIR)

run-test:
	uv run snakemake -d $(TEST_DIR) --snakefile ./Snakefile --cores $(CORES) --config raw_data_file=/data1/bcbs_data/test/dataset-test_raw.h5ad --verbose --printshellcmds 2>&1 | tee snakemake.log

run-conda:
	uv run snakemake --cores 8 --sdm conda --config datadir=$(DATADIR)/immune_aging/wf_snakemake/
# Generate HTML report (run after workflow completes)
report:
	snakemake -c 1 --report $(OUTPUT_DIR)/report.html -d $(OUTPUT_DIR)


# Generate ZIP report (for larger reports with many figures)
report-zip:
	snakemake --report $(DATADIR)/immune_aging/workflow/wf_snakemake/report.zip --config datadir=$(DATADIR)/immune_aging/

# Dry run - show what would be executed
dry-run:
	uv run snakemake -n --cores 1 --config datadir=$(DATADIR)/immune_aging/wf_snakemake/

# Clean all outputs
clean:
	rm -rf $(DATADIR)/immune_aging/wf_snakemake
	
.PHONY: rulegraph run report report-zip dry-run clean
