"""Unit tests for predictive modeling module."""

import numpy as np
import pandas as pd
import pytest

from example_rnaseq.predictive_modeling import (
    compare_models,
    plot_predictions,
    prepare_baseline_features,
    prepare_features,
    print_cv_results,
    run_cross_validation,
    train_evaluate_fold,
)


@pytest.fixture
def modeling_data():
    """Create sample data for modeling tests."""
    np.random.seed(42)
    n_samples = 50
    n_genes = 20

    counts_df = pd.DataFrame(
        np.random.negative_binomial(10, 0.1, (n_samples, n_genes)),
        columns=[f"GENE{i}" for i in range(n_genes)],
        index=[f"sample_{i}" for i in range(n_samples)],
    )

    metadata = pd.DataFrame({
        "age": np.random.uniform(20, 80, n_samples),
        "sex": np.random.choice(["male", "female"], n_samples),
    }, index=counts_df.index)

    return counts_df, metadata


class TestPrepareFeatures:
    """Tests for feature preparation."""

    def test_returns_features_and_target(self, modeling_data):
        """Test that features and target are returned."""
        counts_df, metadata = modeling_data
        X, y = prepare_features(counts_df, metadata)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)

    def test_includes_gene_features(self, modeling_data):
        """Test that gene features are included."""
        counts_df, metadata = modeling_data
        X, y = prepare_features(counts_df, metadata)

        # All gene columns should be in X
        for gene in counts_df.columns:
            assert gene in X.columns

    def test_includes_sex_feature(self, modeling_data):
        """Test that sex is encoded as feature."""
        counts_df, metadata = modeling_data
        X, y = prepare_features(counts_df, metadata)

        # Should have more columns than just genes (sex added)
        assert X.shape[1] >= counts_df.shape[1]

    def test_target_is_age(self, modeling_data):
        """Test that target is age."""
        counts_df, metadata = modeling_data
        X, y = prepare_features(counts_df, metadata)

        np.testing.assert_array_equal(y, metadata["age"].values)


class TestPrepareBaselineFeatures:
    """Tests for baseline feature preparation."""

    def test_returns_sex_encoded(self, modeling_data):
        """Test that sex is one-hot encoded."""
        _, metadata = modeling_data
        X_baseline = prepare_baseline_features(metadata)

        assert isinstance(X_baseline, pd.DataFrame)
        assert X_baseline.shape[0] == len(metadata)
        # With drop_first=True, should have 1 column for binary
        assert X_baseline.shape[1] == 1


class TestTrainEvaluateFold:
    """Tests for single fold training."""

    def test_returns_metrics_and_predictions(self, modeling_data):
        """Test that r2, mae, and predictions are returned."""
        counts_df, metadata = modeling_data
        X, y = prepare_features(counts_df, metadata)

        # Split manually
        train_idx = range(0, 40)
        test_idx = range(40, 50)

        r2, mae, y_pred = train_evaluate_fold(
            X.iloc[train_idx].values,
            X.iloc[test_idx].values,
            y[list(train_idx)],
            y[list(test_idx)],
        )

        assert isinstance(r2, float)
        assert isinstance(mae, float)
        assert len(y_pred) == len(test_idx)

    def test_predictions_are_finite(self, modeling_data):
        """Test that predictions are finite numbers."""
        counts_df, metadata = modeling_data
        X, y = prepare_features(counts_df, metadata)

        train_idx = range(0, 40)
        test_idx = range(40, 50)

        _, _, y_pred = train_evaluate_fold(
            X.iloc[train_idx].values,
            X.iloc[test_idx].values,
            y[list(train_idx)],
            y[list(test_idx)],
        )

        assert np.all(np.isfinite(y_pred))


class TestRunCrossValidation:
    """Tests for cross-validation."""

    def test_returns_correct_number_of_folds(self, modeling_data):
        """Test that correct number of folds is run."""
        counts_df, metadata = modeling_data
        X, y = prepare_features(counts_df, metadata)

        n_splits = 3
        r2_scores, mae_scores, _, _ = run_cross_validation(X, y, n_splits=n_splits)

        assert len(r2_scores) == n_splits
        assert len(mae_scores) == n_splits

    def test_predictions_match_test_samples(self, modeling_data):
        """Test that number of predictions matches test samples."""
        counts_df, metadata = modeling_data
        X, y = prepare_features(counts_df, metadata)

        n_splits = 3
        test_size = 0.2
        _, _, predictions, actuals = run_cross_validation(
            X, y, n_splits=n_splits, test_size=test_size
        )

        # Each fold tests 20% of samples
        expected_total = int(len(X) * test_size) * n_splits
        # Allow some tolerance due to rounding
        assert abs(len(predictions) - expected_total) <= n_splits
        assert len(predictions) == len(actuals)


class TestPrintCvResults:
    """Tests for results printing."""

    def test_prints_without_error(self, capsys):
        """Test that results are printed."""
        r2_scores = [0.5, 0.6, 0.7]
        mae_scores = [5.0, 4.5, 4.0]

        print_cv_results(r2_scores, mae_scores, "Test Model")

        captured = capsys.readouterr()
        assert "Test Model" in captured.out
        assert "R2 Score" in captured.out
        assert "MAE" in captured.out


class TestPlotPredictions:
    """Tests for prediction plotting."""

    def test_saves_figure(self, temp_output_dir):
        """Test that figure is saved."""
        actual = [20, 30, 40, 50, 60]
        predicted = [22, 28, 42, 48, 58]
        r2_scores = [0.9]
        mae_scores = [2.5]

        plot_predictions(actual, predicted, r2_scores, mae_scores, temp_output_dir)

        assert (temp_output_dir / "age_prediction_performance.png").exists()


class TestCompareModels:
    """Tests for model comparison."""

    def test_prints_comparison(self, capsys):
        """Test that comparison is printed."""
        full_r2 = [0.7, 0.75, 0.72]
        full_mae = [4.0, 3.5, 4.2]
        baseline_r2 = [0.1, 0.05, 0.08]
        baseline_mae = [10.0, 11.0, 10.5]

        compare_models(full_r2, full_mae, baseline_r2, baseline_mae)

        captured = capsys.readouterr()
        assert "Full Model" in captured.out
        assert "Baseline" in captured.out
        assert "Improvement" in captured.out
        assert "Delta R2" in captured.out
