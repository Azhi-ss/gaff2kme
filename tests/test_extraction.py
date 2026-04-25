"""Layer 2 tests: Parameter extraction with mocked RadonPy/RDKit."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from gaff2kme.math import MinMaxScale


# ---------------------------------------------------------------------------
# Helper: build a mock RDKit Mol with FF-assigned properties
# ---------------------------------------------------------------------------

def _make_mock_atom(symbol: str, mass: float, charge: float, epsilon: float, sigma: float):
    atom = MagicMock()
    atom.GetSymbol.return_value = symbol
    atom.GetMass.return_value = mass
    atom.GetDoubleProp.side_effect = lambda k: {
        "AtomicCharge": charge,
        "ff_epsilon": epsilon,
        "ff_sigma": sigma,
    }[k]
    atom.GetTotalNumHs.return_value = 0
    atom.GetTotalDegree.return_value = 1
    return atom


def _make_mock_bond(k: float, r0: float, begin_charge: float, end_charge: float):
    bond = MagicMock()
    bond.GetDoubleProp.side_effect = lambda p: {
        "ff_k": k, "ff_r0": r0,
    }[p]
    ba = MagicMock()
    ba.GetDoubleProp.side_effect = lambda p: begin_charge if p == "AtomicCharge" else 0.0
    ea = MagicMock()
    ea.GetDoubleProp.side_effect = lambda p: end_charge if p == "AtomicCharge" else 0.0
    bond.GetBeginAtom.return_value = ba
    bond.GetEndAtom.return_value = ea
    ba.GetSymbol.return_value = "C"
    ea.GetSymbol.return_value = "C"
    return bond


def _make_mock_angle(k: float, theta0: float):
    ff = MagicMock()
    ff.k = k
    ff.theta0 = theta0
    ang = MagicMock()
    ang.ff = ff
    ang.a = 0
    ang.b = 1
    ang.c = 2
    return ang


def _make_mock_dihedral(k_list: list[float]):
    ff = MagicMock()
    ff.k = k_list
    dih = MagicMock()
    dih.ff = ff
    dih.a = 0
    dih.b = 1
    dih.c = 2
    dih.d = 3
    return dih


def _make_assigned_mol():
    """Create a mock mol with 2 atoms (C, H), 1 bond, 1 angle, 1 dihedral."""
    mol = MagicMock()
    c_atom = _make_mock_atom("C", 12.011, -0.1, 0.1094, 3.39967)
    h_atom = _make_mock_atom("H", 1.008, 0.1, 0.0157, 2.59967)
    mol.GetAtoms.return_value = [c_atom, h_atom]
    mol.GetAtomWithIdx.side_effect = lambda i: [c_atom, h_atom][i]

    bond = _make_mock_bond(340.0, 1.09, -0.1, 0.1)
    mol.GetBonds.return_value = [bond]

    ang = _make_mock_angle(50.0, 109.5)
    mol.angles = {"0-1-2": ang}

    dih = _make_mock_dihedral([0.15, 0.0, 0.25])
    mol.dihedrals = {"0-1-2-3": dih}

    return mol


# ---------------------------------------------------------------------------
# Tests: smiles_to_assigned_mol
# ---------------------------------------------------------------------------

class TestSmilesToAssignedMol:
    def test_simple_smiles(self):
        import gaff2kme.extraction as ext
        mock_chem = MagicMock()
        mock_mol = MagicMock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        ff = MagicMock()
        ff.ff_assign.return_value = True

        original_chem = ext.Chem
        original_poly = ext.poly
        try:
            ext.Chem = mock_chem
            ext.poly = None
            result = ext.smiles_to_assigned_mol("C", ff, cyclic=0)
            assert result is mock_mol
        finally:
            ext.Chem = original_chem
            ext.poly = original_poly

    def test_cyclic_polymer(self):
        import gaff2kme.extraction as ext
        mock_chem = MagicMock()
        mock_poly = MagicMock()
        mock_mol = MagicMock()
        mock_poly.make_cyclicpolymer.return_value = mock_mol
        ff = MagicMock()
        ff.ff_assign.return_value = True

        original_chem = ext.Chem
        original_poly = ext.poly
        try:
            ext.Chem = mock_chem
            ext.poly = mock_poly
            result = ext.smiles_to_assigned_mol("*CC*", ff, cyclic=10)
            mock_poly.make_cyclicpolymer.assert_called_once()
            assert result is mock_mol
        finally:
            ext.Chem = original_chem
            ext.poly = original_poly

    def test_cyclic_disabled(self):
        import gaff2kme.extraction as ext
        mock_chem = MagicMock()
        mock_mol = MagicMock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        ff = MagicMock()
        ff.ff_assign.return_value = True

        original_chem = ext.Chem
        original_poly = ext.poly
        try:
            ext.Chem = mock_chem
            ext.poly = None
            ext.smiles_to_assigned_mol("*CC*", ff, cyclic=0)
            mock_chem.MolFromSmiles.assert_called_once()
        finally:
            ext.Chem = original_chem
            ext.poly = original_poly

    def test_invalid_smiles(self):
        import gaff2kme.extraction as ext
        mock_chem = MagicMock()
        mock_chem.MolFromSmiles.return_value = None
        ff = MagicMock()

        original_chem = ext.Chem
        original_poly = ext.poly
        try:
            ext.Chem = mock_chem
            ext.poly = None
            result = ext.smiles_to_assigned_mol("INVALID", ff, cyclic=0)
            assert result is None
        finally:
            ext.Chem = original_chem
            ext.poly = original_poly

    def test_ff_assign_fails(self):
        import gaff2kme.extraction as ext
        mock_chem = MagicMock()
        mock_mol = MagicMock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        ff = MagicMock()
        ff.ff_assign.return_value = False

        original_chem = ext.Chem
        original_poly = ext.poly
        try:
            ext.Chem = mock_chem
            ext.poly = None
            result = ext.smiles_to_assigned_mol("C", ff, cyclic=0)
            assert result is None
        finally:
            ext.Chem = original_chem
            ext.poly = original_poly


# ---------------------------------------------------------------------------
# Tests: extract_params
# ---------------------------------------------------------------------------

class TestExtractParams:
    def test_keys_polar_true(self):
        from gaff2kme.extraction import extract_params
        mol = _make_assigned_mol()
        params = extract_params(mol, polar=True)
        expected = {"mass", "charge", "epsilon", "sigma", "k_bond", "r0",
                    "polar", "k_angle", "theta0", "k_dih"}
        assert set(params.keys()) == expected

    def test_keys_polar_false(self):
        from gaff2kme.extraction import extract_params
        mol = _make_assigned_mol()
        params = extract_params(mol, polar=False)
        assert "polar" not in params

    def test_arrays_are_numpy(self):
        from gaff2kme.extraction import extract_params
        mol = _make_assigned_mol()
        params = extract_params(mol, polar=True)
        for v in params.values():
            assert isinstance(v, np.ndarray)

    def test_mass_values(self):
        from gaff2kme.extraction import extract_params
        mol = _make_assigned_mol()
        params = extract_params(mol, polar=True)
        np.testing.assert_allclose(params["mass"], [12.011, 1.008])

    def test_charge_values(self):
        from gaff2kme.extraction import extract_params
        mol = _make_assigned_mol()
        params = extract_params(mol, polar=True)
        np.testing.assert_allclose(params["charge"], [-0.1, 0.1])

    def test_polar_values(self):
        from gaff2kme.extraction import extract_params
        mol = _make_assigned_mol()
        params = extract_params(mol, polar=True)
        np.testing.assert_allclose(params["polar"], [0.2])

    def test_k_bond_values(self):
        from gaff2kme.extraction import extract_params
        mol = _make_assigned_mol()
        params = extract_params(mol, polar=True)
        np.testing.assert_allclose(params["k_bond"], [340.0])

    def test_k_dih_takes_max(self):
        from gaff2kme.extraction import extract_params
        mol = _make_assigned_mol()
        params = extract_params(mol, polar=True)
        np.testing.assert_allclose(params["k_dih"], [0.25])

    def test_ignore_h(self):
        from gaff2kme.extraction import extract_params
        mol = _make_assigned_mol()
        params = extract_params(mol, polar=True, ignoreH=True)
        assert len(params["mass"]) == 1
        assert len(params["charge"]) == 1


# ---------------------------------------------------------------------------
# Tests: build_scaling_factors
# ---------------------------------------------------------------------------

class TestBuildScalingFactors:
    def _make_mock_ff(self):
        ff = MagicMock()
        pt = {}
        for tag, eps, sig in [("c3", 0.1094, 3.39967), ("h1", 0.0157, 2.59967),
                               ("c", 0.086, 3.39967), ("os", 0.170, 3.0)]:
            entry = MagicMock()
            entry.epsilon = eps
            entry.sigma = sig
            pt[tag] = entry
        ff.param.pt = pt

        bt = {}
        for tag, k, r0 in [("c3-c3", 316.55, 1.526), ("c3-h1", 340.0, 1.09)]:
            entry = MagicMock()
            entry.k = k
            entry.r0 = r0
            bt[tag] = entry
        ff.param.bt = bt

        at = {}
        for tag, k, th in [("c3-c3-c3", 62.51, 109.50)]:
            entry = MagicMock()
            entry.k = k
            entry.theta0 = th
            at[tag] = entry
        ff.param.at = at

        dt = {}
        for tag in ["X-c3-c3-X"]:
            entry = MagicMock()
            entry.k = [0.157, 0.0, 0.25]
            dt[tag] = entry
        ff.param.dt = dt

        return ff

    def test_keys_polar_true(self):
        from gaff2kme.extraction import build_scaling_factors
        ff = self._make_mock_ff()
        scaling = build_scaling_factors(ff, polar=True)
        expected = {"mass", "charge", "epsilon", "sigma", "k_bond", "r0",
                    "polar", "k_angle", "theta0", "k_dih"}
        assert set(scaling.keys()) == expected

    def test_keys_polar_false(self):
        from gaff2kme.extraction import build_scaling_factors
        ff = self._make_mock_ff()
        scaling = build_scaling_factors(ff, polar=False)
        assert "polar" not in scaling

    def test_scaling_values_are_minmaxscale(self):
        from gaff2kme.extraction import build_scaling_factors
        ff = self._make_mock_ff()
        scaling = build_scaling_factors(ff, polar=True)
        for v in scaling.values():
            assert isinstance(v, MinMaxScale)

    def test_mass_range(self):
        from gaff2kme.extraction import build_scaling_factors
        ff = self._make_mock_ff()
        scaling = build_scaling_factors(ff, polar=True)
        mass_scaler = scaling["mass"]
        assert mass_scaler.dataMin == pytest.approx(1.008)
        unscaled_max = mass_scaler.unscale(1.0)
        assert unscaled_max == pytest.approx(126.904)


# ---------------------------------------------------------------------------
# Tests: create_force_field
# ---------------------------------------------------------------------------

class TestCreateForceField:
    def test_invalid_raises(self):
        from gaff2kme.extraction import create_force_field
        with pytest.raises(ValueError, match="Unknown force field"):
            create_force_field("invalid")

    @patch("gaff2kme.extraction.importlib.import_module")
    def test_gaff2_mod(self, mock_import):
        from gaff2kme.extraction import create_force_field
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.GAFF2_mod = mock_cls
        mock_import.return_value = mock_module
        ff = create_force_field("gaff2_mod")
        mock_import.assert_called_with("radonpy.ff.gaff2_mod")
        mock_cls.assert_called_once()

    @patch("gaff2kme.extraction.importlib.import_module")
    def test_gaff2(self, mock_import):
        from gaff2kme.extraction import create_force_field
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.GAFF2 = mock_cls
        mock_import.return_value = mock_module
        ff = create_force_field("gaff2")
        mock_import.assert_called_with("radonpy.ff.gaff2")
        mock_cls.assert_called_once()
