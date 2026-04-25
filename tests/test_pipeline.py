"""Layer 3 tests: End-to-end pipeline integration tests.

These tests require RadonPy and RDKit installed.
Marked as integration so they can be skipped with -m "not integration".
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

# Ensure RadonPy is importable
_radonpy_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "RadonPy")
_radonpy_abs = os.path.abspath(_radonpy_path)
if _radonpy_abs not in sys.path and os.path.isdir(_radonpy_abs):
    sys.path.insert(0, _radonpy_abs)

from gaff2kme.pipeline import compute, compute_batch, compute_from_csv, desc_names


@pytest.mark.integration
class TestCompute:
    def test_compute_methane(self):
        d = compute("C")
        assert d.shape == (190,)
        assert not np.any(np.isnan(d))

    def test_compute_ethane(self):
        d = compute("CC")
        assert d.shape == (190,)
        assert not np.any(np.isnan(d))

    def test_compute_benzene(self):
        d = compute("c1ccccc1")
        assert d.shape == (190,)
        assert not np.any(np.isnan(d))

    def test_compute_invalid(self):
        d = compute("INVALID")
        assert d.shape == (190,)
        assert np.all(np.isnan(d))

    def test_compute_polar_false(self):
        d = compute("C", polar=False)
        assert d.shape == (170,)

    def test_compute_cyclic_polymer(self):
        d = compute("*CC*", cyclic=10, backend="rdkit_lite")
        assert d.shape == (190,)
        assert not np.any(np.isnan(d))

    def test_compute_gaff2(self):
        d = compute("C", ff_name="gaff2")
        assert d.shape == (190,)
        assert not np.any(np.isnan(d))

    def test_compute_deterministic(self):
        d1 = compute("C")
        d2 = compute("C")
        np.testing.assert_array_equal(d1, d2)

    def test_compute_mass_sum(self):
        d = compute("C")
        mass_sum = np.sum(d[:10])
        assert pytest.approx(mass_sum, abs=1e-10) == 1.0


@pytest.mark.integration
class TestComputeBatch:
    def test_batch_basic(self):
        df = compute_batch(["C", "CC"])
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 190)

    def test_batch_with_failure(self):
        df = compute_batch(["C", "INVALID", "CC"])
        assert df.shape[0] == 3
        assert df.shape[1] == 190
        assert not np.any(np.isnan(df.iloc[0].values))
        assert np.all(np.isnan(df.iloc[1].values))

    def test_batch_polar_false(self):
        df = compute_batch(["C"], polar=False)
        assert df.shape[1] == 170

    def test_batch_column_names(self):
        df = compute_batch(["C"])
        names = list(df.columns)
        assert names[0] == "mass_H"
        assert names[-1] == "k_dih_19"


@pytest.mark.integration
class TestComputeFromCsv:
    def test_csv_roundtrip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("SMILES\nC\nCC\n")
            input_path = f.name
        output_path = input_path.replace(".csv", "_out.csv")
        try:
            df = compute_from_csv(input_path, output=output_path)
            assert df.shape == (2, 190)
            assert os.path.exists(output_path)
            df_read = pd.read_csv(output_path)
            assert df_read.shape == (2, 190)
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_csv_custom_column(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("smi\nC\n")
            input_path = f.name
        try:
            df = compute_from_csv(input_path, smiles_col="smi")
            assert df.shape == (1, 190)
        finally:
            os.unlink(input_path)


@pytest.mark.integration
class TestDescNames:
    def test_desc_names_count(self):
        names = desc_names()
        assert len(names) == 190

    def test_desc_names_polar_false(self):
        names = desc_names(polar=False)
        assert len(names) == 170

    def test_desc_names_format(self):
        names = desc_names()
        assert names[:10] == ["mass_H", "mass_C", "mass_N", "mass_O",
                              "mass_F", "mass_P", "mass_S", "mass_Cl",
                              "mass_Br", "mass_I"]
        assert names[10] == "charge_0"
        assert names[-1] == "k_dih_19"
