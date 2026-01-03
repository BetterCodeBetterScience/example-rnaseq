"""Unit tests for execution logging module."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from example_rnaseq.execution_log import (
    ExecutionLog,
    StepRecord,
    create_execution_log,
    serialize_parameters,
)


class TestStepRecord:
    """Tests for StepRecord dataclass."""

    def test_creates_step_with_required_fields(self):
        """Test that step record can be created with required fields."""
        record = StepRecord(
            step_number=1,
            step_name="test_step",
            start_time=datetime.now().isoformat(),
        )
        assert record.step_number == 1
        assert record.step_name == "test_step"
        assert record.status == "running"
        assert record.from_cache is False

    def test_default_values(self):
        """Test that optional fields have correct defaults."""
        record = StepRecord(
            step_number=1,
            step_name="test_step",
            start_time=datetime.now().isoformat(),
        )
        assert record.end_time is None
        assert record.duration_seconds is None
        assert record.parameters == {}
        assert record.checkpoint_file is None
        assert record.error_message is None


class TestExecutionLog:
    """Tests for ExecutionLog class."""

    def test_creates_log_with_required_fields(self):
        """Test that execution log can be created."""
        log = ExecutionLog(
            workflow_name="test_workflow",
            run_id="test_123",
            start_time=datetime.now().isoformat(),
        )
        assert log.workflow_name == "test_workflow"
        assert log.run_id == "test_123"
        assert log.status == "running"
        assert log.steps == []

    def test_add_step(self):
        """Test adding a step to the log."""
        log = ExecutionLog(
            workflow_name="test",
            run_id="123",
            start_time=datetime.now().isoformat(),
        )
        record = log.add_step(
            step_number=1,
            step_name="preprocess",
            parameters={"n_genes": 100},
            checkpoint_file="step1.h5ad",
        )
        assert len(log.steps) == 1
        assert record.step_name == "preprocess"
        assert record.parameters == {"n_genes": 100}
        assert record.checkpoint_file == "step1.h5ad"

    def test_complete_step(self):
        """Test completing a step."""
        log = ExecutionLog(
            workflow_name="test",
            run_id="123",
            start_time=datetime.now().isoformat(),
        )
        record = log.add_step(step_number=1, step_name="test")
        log.complete_step(record, from_cache=False)

        assert record.status == "completed"
        assert record.end_time is not None
        assert record.duration_seconds is not None
        assert record.duration_seconds >= 0
        assert record.from_cache is False

    def test_complete_step_from_cache(self):
        """Test completing a cached step."""
        log = ExecutionLog(
            workflow_name="test",
            run_id="123",
            start_time=datetime.now().isoformat(),
        )
        record = log.add_step(step_number=1, step_name="cached")
        log.complete_step(record, from_cache=True)

        assert record.from_cache is True
        assert record.status == "completed"

    def test_complete_step_with_error(self):
        """Test completing a step with an error."""
        log = ExecutionLog(
            workflow_name="test",
            run_id="123",
            start_time=datetime.now().isoformat(),
        )
        record = log.add_step(step_number=1, step_name="failed")
        log.complete_step(record, error_message="Something went wrong")

        assert record.status == "failed"
        assert record.error_message == "Something went wrong"

    def test_complete_workflow(self):
        """Test completing the workflow."""
        log = ExecutionLog(
            workflow_name="test",
            run_id="123",
            start_time=datetime.now().isoformat(),
        )
        log.complete()

        assert log.status == "completed"
        assert log.end_time is not None
        assert log.total_duration_seconds is not None
        assert log.total_duration_seconds >= 0

    def test_complete_workflow_with_error(self):
        """Test completing the workflow with an error."""
        log = ExecutionLog(
            workflow_name="test",
            run_id="123",
            start_time=datetime.now().isoformat(),
        )
        log.complete(error_message="Workflow failed")

        assert log.status == "failed"

    def test_to_dict(self):
        """Test converting log to dictionary."""
        log = ExecutionLog(
            workflow_name="test",
            run_id="123",
            start_time=datetime.now().isoformat(),
            workflow_parameters={"param1": "value1"},
        )
        log.add_step(step_number=1, step_name="step1")

        result = log.to_dict()

        assert isinstance(result, dict)
        assert result["workflow_name"] == "test"
        assert result["run_id"] == "123"
        assert result["workflow_parameters"] == {"param1": "value1"}
        assert len(result["steps"]) == 1
        assert result["steps"][0]["step_name"] == "step1"

    def test_save(self, tmp_path):
        """Test saving log to file."""
        log = ExecutionLog(
            workflow_name="test_workflow",
            run_id="test_run",
            start_time=datetime.now().isoformat(),
        )
        log.add_step(step_number=1, step_name="test")
        log.complete()

        log_file = log.save(tmp_path / "logs")

        assert log_file.exists()
        assert log_file.suffix == ".json"
        assert "execution_log" in log_file.name

        with open(log_file) as f:
            saved_data = json.load(f)
        assert saved_data["workflow_name"] == "test_workflow"

    def test_print_summary(self, capsys):
        """Test printing summary."""
        log = ExecutionLog(
            workflow_name="test",
            run_id="123",
            start_time=datetime.now().isoformat(),
        )
        record = log.add_step(step_number=1, step_name="step1")
        log.complete_step(record)
        log.complete()

        log.print_summary()

        captured = capsys.readouterr()
        assert "EXECUTION SUMMARY" in captured.out
        assert "test" in captured.out
        assert "step1" in captured.out


class TestCreateExecutionLog:
    """Tests for create_execution_log function."""

    def test_creates_log_with_name(self):
        """Test creating a log with workflow name."""
        log = create_execution_log("my_workflow")

        assert log.workflow_name == "my_workflow"
        assert log.run_id is not None
        assert log.start_time is not None
        assert log.status == "running"

    def test_creates_log_with_parameters(self):
        """Test creating a log with workflow parameters."""
        params = {"input_file": "data.h5ad", "n_genes": 1000}
        log = create_execution_log("workflow", workflow_parameters=params)

        assert log.workflow_parameters == params


class TestSerializeParameters:
    """Tests for serialize_parameters function."""

    def test_serializes_simple_values(self):
        """Test serializing simple types."""
        result = serialize_parameters(
            name="test",
            count=42,
            ratio=0.5,
            flag=True,
        )

        assert result["name"] == "test"
        assert result["count"] == 42
        assert result["ratio"] == 0.5
        assert result["flag"] is True

    def test_serializes_path(self):
        """Test serializing Path objects."""
        result = serialize_parameters(file_path=Path("/tmp/test.h5ad"))

        assert result["file_path"] == "/tmp/test.h5ad"

    def test_serializes_numpy_array(self):
        """Test serializing numpy arrays."""
        arr = np.array([1, 2, 3])
        result = serialize_parameters(data=arr)

        assert result["data"] == [1, 2, 3]

    def test_serializes_complex_objects(self):
        """Test serializing objects with __dict__."""
        class MyObject:
            pass

        result = serialize_parameters(obj=MyObject())

        assert result["obj"] == "MyObject"

    def test_serializes_non_serializable(self):
        """Test serializing non-JSON-serializable values."""
        # Using a lambda as example of non-serializable
        result = serialize_parameters(func=lambda x: x)

        assert "function" in result["func"]
