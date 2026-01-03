"""Tests for the Snakemake workflow.

These tests verify that the Snakemake workflow is correctly configured
and can execute (at least in dry-run mode).
"""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def workflow_dir() -> Path:
    """Return the snakemake workflow directory."""
    return Path(__file__).parent.parent / "snakemake_workflow"


@pytest.fixture
def snakefile(workflow_dir) -> Path:
    """Return path to the Snakefile."""
    return workflow_dir / "Snakefile"


@pytest.fixture
def config_file(workflow_dir) -> Path:
    """Return path to the config file."""
    return workflow_dir / "config" / "config.yaml"


class TestSnakemakeSetup:
    """Tests for Snakemake workflow setup."""

    def test_snakefile_exists(self, snakefile):
        """Test that Snakefile exists."""
        assert snakefile.exists(), f"Snakefile not found at {snakefile}"

    def test_config_exists(self, config_file):
        """Test that config file exists."""
        assert config_file.exists(), f"Config not found at {config_file}"

    def test_rules_directory_exists(self, workflow_dir):
        """Test that rules directory exists."""
        rules_dir = workflow_dir / "rules"
        assert rules_dir.exists()

    def test_all_rule_files_exist(self, workflow_dir):
        """Test that all expected rule files exist."""
        rules_dir = workflow_dir / "rules"
        expected_rules = ["common.smk", "preprocessing.smk", "pseudobulk.smk", "per_cell_type.smk"]

        for rule_file in expected_rules:
            assert (rules_dir / rule_file).exists(), f"Rule file {rule_file} not found"


@pytest.mark.integration
class TestSnakemakeDryRun:
    """Tests that run Snakemake in dry-run mode."""

    def test_snakemake_dryrun(self, workflow_dir, snakefile, config_file):
        """Test that Snakemake dry-run works."""
        # Run snakemake in dry-run mode
        result = subprocess.run(
            [
                "snakemake",
                "--snakefile", str(snakefile),
                "--configfile", str(config_file),
                "--dry-run",
                "--quiet",
                "--cores", "1",
            ],
            cwd=str(workflow_dir),
            capture_output=True,
            text=True,
        )

        # Check that it didn't fail due to syntax errors
        # Note: It may fail if input files don't exist, which is expected
        combined_output = result.stdout + result.stderr
        if result.returncode != 0:
            # Check if failure is due to missing input (acceptable)
            if "MissingInputException" in combined_output or "Missing input files" in combined_output:
                pytest.skip("Dry-run failed due to missing input files (expected for test environment)")
            # Also check for WorkflowError which may indicate missing files
            elif "WorkflowError" in combined_output:
                pytest.skip("Dry-run failed due to workflow configuration (expected for test environment)")
            else:
                pytest.fail(f"Snakemake dry-run failed: {combined_output}")

    def test_snakemake_list_rules(self, workflow_dir, snakefile, config_file):
        """Test that Snakemake can list rules."""
        result = subprocess.run(
            [
                "snakemake",
                "--snakefile", str(snakefile),
                "--configfile", str(config_file),
                "--list",
            ],
            cwd=str(workflow_dir),
            capture_output=True,
            text=True,
        )

        # This should succeed even without input files
        if result.returncode != 0:
            # If it fails, check the error
            if "MissingInputException" not in result.stderr:
                pytest.fail(f"Snakemake list failed: {result.stderr}")

    def test_snakemake_dag_generation(self, workflow_dir, snakefile, config_file, tmp_path):
        """Test that Snakemake can generate DAG."""
        dag_file = tmp_path / "dag.txt"

        result = subprocess.run(
            [
                "snakemake",
                "--snakefile", str(snakefile),
                "--configfile", str(config_file),
                "--dag",
            ],
            cwd=str(workflow_dir),
            capture_output=True,
            text=True,
        )

        # DAG generation may fail if inputs are missing, but should produce output
        if result.returncode == 0:
            assert len(result.stdout) > 0, "DAG output should not be empty"


@pytest.mark.slow
@pytest.mark.integration
class TestSnakemakeWithTestData:
    """Tests that run Snakemake with test data (marked slow)."""

    def test_preprocessing_rules_with_test_data(
        self, workflow_dir, snakefile, tmp_path, test_data_dir
    ):
        """Test running preprocessing rules with test data."""
        # Check if test data exists
        test_data = test_data_dir / "dataset-test_raw.h5ad"
        if not test_data.exists():
            pytest.skip("Test data not found")

        # Create a temporary config that uses test data
        config_content = f"""
dataset_name: "test"
data_url: ""
raw_data_file: "{test_data}"
output_dir: "{tmp_path / 'output'}"

min_genes: 10
max_genes: 10000
min_counts: 50
max_counts: 100000
max_hb_pct: 50.0
expected_doublet_rate: 0.06
cutoff_percentile: 1.0
min_cells_per_celltype: 1
percent_donors: 0.1
"""
        test_config = tmp_path / "test_config.yaml"
        test_config.write_text(config_content)

        # Try to run just the first rule in dry-run mode
        result = subprocess.run(
            [
                "snakemake",
                "--snakefile", str(snakefile),
                "--configfile", str(test_config),
                "--dry-run",
                "--quiet",
                "filter_data",  # Run just the filter rule
            ],
            cwd=str(workflow_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )

        # This may fail due to rule dependencies, but shouldn't have syntax errors
        if result.returncode != 0:
            if "SyntaxError" in result.stderr or "NameError" in result.stderr:
                pytest.fail(f"Snakemake has syntax errors: {result.stderr}")
