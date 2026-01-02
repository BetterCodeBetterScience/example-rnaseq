"""Unit tests for data loading module."""

from pathlib import Path

import anndata as ad
import pytest

from example_rnaseq.data_loading import (
    load_anndata,
    load_lazy_anndata,
    save_anndata,
)


def test_save_and_load_anndata(minimal_adata, temp_output_dir):
    """Test saving and loading AnnData round-trip."""
    filepath = temp_output_dir / "test_adata.h5ad"

    save_anndata(minimal_adata, filepath)
    assert filepath.exists()

    loaded = load_anndata(filepath)
    assert isinstance(loaded, ad.AnnData)
    assert loaded.shape == minimal_adata.shape
    assert list(loaded.obs.columns) == list(minimal_adata.obs.columns)
    assert list(loaded.var.columns) == list(minimal_adata.var.columns)


def test_load_lazy_anndata(minimal_adata, temp_output_dir):
    """Test lazy loading of AnnData."""
    filepath = temp_output_dir / "test_lazy.h5ad"
    save_anndata(minimal_adata, filepath)

    # Lazy load
    lazy_adata = load_lazy_anndata(filepath)
    assert lazy_adata.shape == minimal_adata.shape


def test_load_anndata_preserves_data(minimal_adata, temp_output_dir):
    """Test that data values are preserved."""
    filepath = temp_output_dir / "test_data.h5ad"
    save_anndata(minimal_adata, filepath)

    loaded = load_anndata(filepath)

    # Check obs data
    assert (loaded.obs["donor_id"] == minimal_adata.obs["donor_id"]).all()
    assert (loaded.obs["cell_type"] == minimal_adata.obs["cell_type"]).all()

    # Check var data
    assert (loaded.var["feature_name"] == minimal_adata.var["feature_name"]).all()


def test_load_anndata_nonexistent_file():
    """Test that loading nonexistent file raises error."""
    with pytest.raises(Exception):
        load_anndata(Path("/nonexistent/path/file.h5ad"))


@pytest.mark.integration
def test_load_test_dataset(test_data_dir):
    """Test loading the actual test dataset."""
    test_file = test_data_dir / "dataset-test_raw.h5ad"
    if not test_file.exists():
        pytest.skip("Test data file not found")

    adata = load_anndata(test_file)

    # Verify expected structure
    assert adata.n_obs > 0
    assert adata.n_vars > 0
    assert "donor_id" in adata.obs.columns
    assert "cell_type" in adata.obs.columns
    assert "feature_name" in adata.var.columns
