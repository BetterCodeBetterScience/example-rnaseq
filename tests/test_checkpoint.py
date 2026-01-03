"""Unit tests for checkpoint module."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from example_rnaseq.checkpoint import (
    bids_checkpoint_name,
    clear_checkpoints,
    clear_checkpoints_from_step,
    get_file_type,
    hash_parameters,
    load_checkpoint,
    parse_bids_checkpoint_name,
    run_with_checkpoint,
    run_with_checkpoint_multi,
    save_checkpoint,
)
from example_rnaseq.execution_log import create_execution_log


class TestBidsCheckpointName:
    """Tests for BIDS checkpoint naming functions."""

    def test_generates_correct_format(self):
        """Test basic BIDS name generation."""
        name = bids_checkpoint_name("OneK1K", 2, "filtered", "h5ad")
        assert name == "dataset-OneK1K_step-02_desc-filtered.h5ad"

    def test_zero_pads_step_number(self):
        """Test step numbers are zero-padded."""
        name = bids_checkpoint_name("test", 1, "qc", "h5ad")
        assert "step-01" in name

        name = bids_checkpoint_name("test", 10, "qc", "h5ad")
        assert "step-10" in name

    def test_supports_different_extensions(self):
        """Test different file extensions."""
        assert bids_checkpoint_name("ds", 1, "d", "pkl").endswith(".pkl")
        assert bids_checkpoint_name("ds", 1, "d", "parquet").endswith(".parquet")


class TestParseBidsCheckpointName:
    """Tests for parsing BIDS checkpoint names."""

    def test_parses_valid_name(self):
        """Test parsing a valid BIDS name."""
        parsed = parse_bids_checkpoint_name("dataset-OneK1K_step-02_desc-filtered.h5ad")
        assert parsed["dataset"] == "OneK1K"
        assert parsed["step_number"] == 2
        assert parsed["description"] == "filtered"
        assert parsed["extension"] == "h5ad"

    def test_returns_empty_for_invalid(self):
        """Test that invalid names return empty dict."""
        assert parse_bids_checkpoint_name("invalid_name.txt") == {}
        assert parse_bids_checkpoint_name("") == {}

    def test_roundtrip(self):
        """Test that generate and parse are inverses."""
        original_name = bids_checkpoint_name("TestData", 5, "preprocessed", "parquet")
        parsed = parse_bids_checkpoint_name(original_name)
        assert parsed["dataset"] == "TestData"
        assert parsed["step_number"] == 5
        assert parsed["description"] == "preprocessed"


class TestGetFileType:
    """Tests for file type detection."""

    def test_h5ad_extension(self):
        """Test h5ad file detection."""
        assert get_file_type(Path("test.h5ad")) == "h5ad"
        assert get_file_type(Path("test.H5AD")) == "h5ad"

    def test_parquet_extension(self):
        """Test parquet file detection."""
        assert get_file_type(Path("test.parquet")) == "parquet"

    def test_pickle_extension(self):
        """Test pickle file detection."""
        assert get_file_type(Path("test.pkl")) == "pickle"

    def test_unknown_extension_raises(self):
        """Test that unknown extensions raise ValueError."""
        with pytest.raises(ValueError, match="Unknown file extension"):
            get_file_type(Path("test.unknown"))


class TestSaveLoadCheckpoint:
    """Tests for saving and loading checkpoints."""

    def test_save_load_anndata(self, minimal_adata, temp_checkpoint_dir):
        """Test saving and loading AnnData."""
        filepath = temp_checkpoint_dir / "test.h5ad"
        save_checkpoint(minimal_adata, filepath)

        loaded = load_checkpoint(filepath)
        assert isinstance(loaded, ad.AnnData)
        assert loaded.shape == minimal_adata.shape

    def test_save_load_dataframe(self, temp_checkpoint_dir):
        """Test saving and loading DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        filepath = temp_checkpoint_dir / "test.parquet"

        save_checkpoint(df, filepath)
        loaded = load_checkpoint(filepath)

        assert isinstance(loaded, pd.DataFrame)
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_load_pickle(self, temp_checkpoint_dir):
        """Test saving and loading arbitrary objects."""
        data = {"key": [1, 2, 3], "nested": {"a": 1}}
        filepath = temp_checkpoint_dir / "test.pkl"

        save_checkpoint(data, filepath)
        loaded = load_checkpoint(filepath)

        assert loaded == data

    def test_creates_parent_directories(self, temp_checkpoint_dir):
        """Test that parent directories are created."""
        filepath = temp_checkpoint_dir / "nested" / "deep" / "test.pkl"
        save_checkpoint({"a": 1}, filepath)
        assert filepath.exists()

    def test_raises_on_wrong_type_h5ad(self, temp_checkpoint_dir):
        """Test that saving non-AnnData to h5ad raises TypeError."""
        filepath = temp_checkpoint_dir / "test.h5ad"
        with pytest.raises(TypeError, match="Expected AnnData"):
            save_checkpoint({"not": "anndata"}, filepath)

    def test_raises_on_wrong_type_parquet(self, temp_checkpoint_dir):
        """Test that saving non-DataFrame to parquet raises TypeError."""
        filepath = temp_checkpoint_dir / "test.parquet"
        with pytest.raises(TypeError, match="Expected DataFrame"):
            save_checkpoint({"not": "dataframe"}, filepath)


class TestHashParameters:
    """Tests for parameter hashing."""

    def test_consistent_hash(self):
        """Test that same parameters produce same hash."""
        hash1 = hash_parameters(a=1, b="test")
        hash2 = hash_parameters(a=1, b="test")
        assert hash1 == hash2

    def test_different_params_different_hash(self):
        """Test that different parameters produce different hash."""
        hash1 = hash_parameters(a=1)
        hash2 = hash_parameters(a=2)
        assert hash1 != hash2

    def test_order_independent(self):
        """Test that parameter order doesn't affect hash."""
        hash1 = hash_parameters(a=1, b=2)
        hash2 = hash_parameters(b=2, a=1)
        assert hash1 == hash2

    def test_returns_8_chars(self):
        """Test hash length."""
        h = hash_parameters(x=100)
        assert len(h) == 8


class TestRunWithCheckpoint:
    """Tests for checkpoint-based execution."""

    def test_executes_and_saves(self, temp_checkpoint_dir):
        """Test that function is executed and result saved."""
        filepath = temp_checkpoint_dir / "result.pkl"
        call_count = [0]

        def my_func(x):
            call_count[0] += 1
            return x * 2

        result = run_with_checkpoint("test", filepath, my_func, 5)

        assert result == 10
        assert call_count[0] == 1
        assert filepath.exists()

    def test_loads_from_cache(self, temp_checkpoint_dir):
        """Test that cached result is loaded."""
        filepath = temp_checkpoint_dir / "cached.pkl"
        call_count = [0]

        def my_func():
            call_count[0] += 1
            return "result"

        # First call - executes
        run_with_checkpoint("test", filepath, my_func)
        assert call_count[0] == 1

        # Second call - loads from cache
        result = run_with_checkpoint("test", filepath, my_func)
        assert result == "result"
        assert call_count[0] == 1  # Not called again

    def test_force_rerun(self, temp_checkpoint_dir):
        """Test that force=True re-executes."""
        filepath = temp_checkpoint_dir / "force.pkl"
        call_count = [0]

        def my_func():
            call_count[0] += 1
            return call_count[0]

        run_with_checkpoint("test", filepath, my_func)
        result = run_with_checkpoint("test", filepath, my_func, force=True)

        assert call_count[0] == 2
        assert result == 2

    def test_skip_save(self, temp_checkpoint_dir):
        """Test that skip_save=True doesn't save."""
        filepath = temp_checkpoint_dir / "no_save.pkl"

        result = run_with_checkpoint(
            "test", filepath, lambda: "result", skip_save=True
        )

        assert result == "result"
        assert not filepath.exists()

    def test_passes_args_and_kwargs(self, temp_checkpoint_dir):
        """Test that args and kwargs are passed correctly."""
        filepath = temp_checkpoint_dir / "args.pkl"

        def my_func(a, b, c=10):
            return a + b + c

        result = run_with_checkpoint("test", filepath, my_func, 1, 2, c=3)
        assert result == 6


class TestClearCheckpoints:
    """Tests for checkpoint clearing functions."""

    def test_clear_by_pattern(self, temp_checkpoint_dir):
        """Test clearing checkpoints by pattern."""
        # Create some files
        (temp_checkpoint_dir / "step01_test.h5ad").touch()
        (temp_checkpoint_dir / "step02_test.h5ad").touch()
        (temp_checkpoint_dir / "other_file.txt").touch()

        removed = clear_checkpoints(temp_checkpoint_dir, "step*")

        assert len(removed) == 2
        assert not (temp_checkpoint_dir / "step01_test.h5ad").exists()
        assert (temp_checkpoint_dir / "other_file.txt").exists()

    def test_clear_from_step_legacy_format(self, temp_checkpoint_dir):
        """Test clearing from a step (legacy format)."""
        (temp_checkpoint_dir / "step01_data.h5ad").touch()
        (temp_checkpoint_dir / "step02_data.h5ad").touch()
        (temp_checkpoint_dir / "step03_data.h5ad").touch()

        removed = clear_checkpoints_from_step(temp_checkpoint_dir, from_step=2)

        assert len(removed) == 2
        assert (temp_checkpoint_dir / "step01_data.h5ad").exists()
        assert not (temp_checkpoint_dir / "step02_data.h5ad").exists()
        assert not (temp_checkpoint_dir / "step03_data.h5ad").exists()

    def test_clear_from_step_bids_format(self, temp_checkpoint_dir):
        """Test clearing from a step (BIDS format)."""
        (temp_checkpoint_dir / "dataset-test_step-01_desc-a.h5ad").touch()
        (temp_checkpoint_dir / "dataset-test_step-02_desc-b.h5ad").touch()
        (temp_checkpoint_dir / "dataset-test_step-03_desc-c.h5ad").touch()

        removed = clear_checkpoints_from_step(temp_checkpoint_dir, from_step=2)

        assert len(removed) == 2
        assert (temp_checkpoint_dir / "dataset-test_step-01_desc-a.h5ad").exists()


class TestRunWithCheckpointMulti:
    """Tests for multi-checkpoint execution."""

    def test_executes_and_saves_multiple(self, temp_checkpoint_dir):
        """Test that function is executed and multiple results saved."""
        files = {
            "result1": temp_checkpoint_dir / "result1.pkl",
            "result2": temp_checkpoint_dir / "result2.pkl",
        }
        call_count = [0]

        def my_func(x):
            call_count[0] += 1
            return {"result1": x * 2, "result2": x * 3}

        result = run_with_checkpoint_multi("test", files, my_func, 5)

        assert result == {"result1": 10, "result2": 15}
        assert call_count[0] == 1
        assert files["result1"].exists()
        assert files["result2"].exists()

    def test_loads_from_cache(self, temp_checkpoint_dir):
        """Test that cached results are loaded."""
        files = {
            "a": temp_checkpoint_dir / "a.pkl",
            "b": temp_checkpoint_dir / "b.pkl",
        }
        call_count = [0]

        def my_func():
            call_count[0] += 1
            return {"a": "value_a", "b": "value_b"}

        # First call - executes
        run_with_checkpoint_multi("test", files, my_func)
        assert call_count[0] == 1

        # Second call - loads from cache
        result = run_with_checkpoint_multi("test", files, my_func)
        assert result == {"a": "value_a", "b": "value_b"}
        assert call_count[0] == 1  # Not called again

    def test_force_rerun(self, temp_checkpoint_dir):
        """Test that force=True re-executes."""
        files = {
            "x": temp_checkpoint_dir / "x.pkl",
        }
        call_count = [0]

        def my_func():
            call_count[0] += 1
            return {"x": call_count[0]}

        run_with_checkpoint_multi("test", files, my_func)
        result = run_with_checkpoint_multi("test", files, my_func, force=True)

        assert call_count[0] == 2
        assert result == {"x": 2}

    def test_skip_save(self, temp_checkpoint_dir):
        """Test that skip_save=True doesn't save."""
        files = {"out": temp_checkpoint_dir / "no_save.pkl"}

        result = run_with_checkpoint_multi(
            "test", files, lambda: {"out": "data"}, skip_save=True
        )

        assert result == {"out": "data"}
        assert not files["out"].exists()

    def test_raises_if_return_not_dict(self, temp_checkpoint_dir):
        """Test that non-dict return raises TypeError."""
        files = {"out": temp_checkpoint_dir / "out.pkl"}

        def bad_func():
            return "not a dict"

        with pytest.raises(TypeError, match="must return dict"):
            run_with_checkpoint_multi("test", files, bad_func)

    def test_raises_if_missing_key(self, temp_checkpoint_dir):
        """Test that missing keys raise KeyError."""
        files = {
            "a": temp_checkpoint_dir / "a.pkl",
            "b": temp_checkpoint_dir / "b.pkl",
        }

        def missing_key_func():
            return {"a": 1}  # Missing 'b'

        with pytest.raises(KeyError, match="missing key"):
            run_with_checkpoint_multi("test", files, missing_key_func)


class TestCheckpointWithExecutionLog:
    """Tests for checkpoint execution with logging."""

    def test_logs_step_on_execution(self, temp_checkpoint_dir):
        """Test that execution log records step."""
        log = create_execution_log("test_workflow")
        filepath = temp_checkpoint_dir / "logged.pkl"

        run_with_checkpoint(
            "test_step",
            filepath,
            lambda: "result",
            execution_log=log,
            step_number=1,
        )

        assert len(log.steps) == 1
        assert log.steps[0].step_name == "test_step"
        assert log.steps[0].status == "completed"
        assert log.steps[0].from_cache is False

    def test_logs_cache_hit(self, temp_checkpoint_dir):
        """Test that cache hit is recorded in log."""
        log = create_execution_log("test_workflow")
        filepath = temp_checkpoint_dir / "cached.pkl"

        # First call - creates cache
        run_with_checkpoint(
            "step1", filepath, lambda: "data",
            execution_log=log, step_number=1
        )

        # Second call with new log - should record cache hit
        log2 = create_execution_log("test_workflow2")
        run_with_checkpoint(
            "step1", filepath, lambda: "new_data",
            execution_log=log2, step_number=1
        )

        assert log2.steps[0].from_cache is True

    def test_logs_parameters(self, temp_checkpoint_dir):
        """Test that parameters are logged."""
        log = create_execution_log("test_workflow")
        filepath = temp_checkpoint_dir / "params.pkl"

        run_with_checkpoint(
            "param_step",
            filepath,
            lambda: "result",
            execution_log=log,
            step_number=1,
            log_parameters={"n_genes": 100, "threshold": 0.05},
        )

        assert log.steps[0].parameters == {"n_genes": 100, "threshold": 0.05}

    def test_multi_checkpoint_logs_step(self, temp_checkpoint_dir):
        """Test that multi-checkpoint logs step."""
        log = create_execution_log("test_workflow")
        files = {
            "out1": temp_checkpoint_dir / "out1.pkl",
            "out2": temp_checkpoint_dir / "out2.pkl",
        }

        run_with_checkpoint_multi(
            "multi_step",
            files,
            lambda: {"out1": 1, "out2": 2},
            execution_log=log,
            step_number=1,
        )

        assert len(log.steps) == 1
        assert log.steps[0].step_name == "multi_step"
        assert log.steps[0].status == "completed"
