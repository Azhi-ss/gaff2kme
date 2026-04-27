"""Tests for backend abstraction and rdkit_lite backend."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from gaff2kme.backends import get_backend
from gaff2kme.pipeline import compute, compute_batch, desc_names


# ---------------------------------------------------------------------------
# Backend selection tests
# ---------------------------------------------------------------------------

class TestGetBackend:
    def test_default_is_rdkit_lite(self):
        from gaff2kme.backends.rdkit_lite import RdkitLiteBackend
        be = get_backend()
        assert isinstance(be, RdkitLiteBackend)

    def test_rdkit_lite_explicit(self):
        from gaff2kme.backends.rdkit_lite import RdkitLiteBackend
        be = get_backend("rdkit_lite")
        assert isinstance(be, RdkitLiteBackend)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid")

    def test_rdkit_lite_name(self):
        be = get_backend("rdkit_lite", ff_name="gaff2_mod")
        assert be.name == "gaff2_mod"

    def test_rdkit_lite_gaff2(self):
        be = get_backend("rdkit_lite", ff_name="gaff2")
        assert be.name == "gaff2"

    def test_rdkit_lite_missing_ff(self):
        with pytest.raises(ValueError, match="not found"):
            get_backend("rdkit_lite", ff_name="nonexistent")


# ---------------------------------------------------------------------------
# rdkit_lite backend: atom typing
# ---------------------------------------------------------------------------

class TestRdkitLiteTyping:
    def _assign(self, smiles: str):
        from gaff2kme.backends.rdkit_lite import RdkitLiteBackend
        from rdkit import Chem
        be = RdkitLiteBackend()
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        result = be.assign(mol)
        return mol, result

    def test_methane_types(self):
        mol, ok = self._assign("C")
        assert ok
        types = [a.GetProp("ff_type") for a in mol.GetAtoms()]
        assert "c3" in types
        assert "h1" in types or "hc" in types

    def test_methane_epsilon_sigma(self):
        mol, ok = self._assign("C")
        assert ok
        for a in mol.GetAtoms():
            assert a.HasProp("ff_epsilon")
            assert a.HasProp("ff_sigma")

    def test_ethane_bond_params(self):
        mol, ok = self._assign("CC")
        assert ok
        for b in mol.GetBonds():
            assert b.HasProp("ff_k")
            assert b.HasProp("ff_r0")

    def test_benzene_aromatic(self):
        mol, ok = self._assign("c1ccccc1")
        assert ok
        c_types = [a.GetProp("ff_type") for a in mol.GetAtoms()
                   if a.GetSymbol() == "C"]
        assert "ca" in c_types

    def test_angles_created(self):
        mol, ok = self._assign("CC")
        assert ok
        assert len(mol.angles) > 0
        for ang in mol.angles.values():
            assert hasattr(ang, "ff")
            assert hasattr(ang.ff, "k")
            assert hasattr(ang.ff, "theta0")

    def test_dihedrals_created(self):
        mol, ok = self._assign("CCC")
        assert ok
        assert len(mol.dihedrals) > 0
        for dih in mol.dihedrals.values():
            assert hasattr(dih, "ff")
            assert hasattr(dih.ff, "k")

    def test_gasteiger_charges(self):
        mol, ok = self._assign("C")
        assert ok
        for a in mol.GetAtoms():
            assert a.HasProp("AtomicCharge")


# ---------------------------------------------------------------------------
# rdkit_lite backend: extract_params works
# ---------------------------------------------------------------------------

class TestRdkitLiteExtractParams:
    def test_extract_from_assigned_mol(self):
        from gaff2kme.backends.rdkit_lite import RdkitLiteBackend
        from gaff2kme.extraction import extract_params
        from rdkit import Chem
        be = RdkitLiteBackend()
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        be.assign(mol)
        params = extract_params(mol, polar=True)
        assert set(params.keys()) == {"mass", "charge", "epsilon", "sigma",
                                       "k_bond", "r0", "polar", "k_angle",
                                       "theta0", "k_dih"}
        for v in params.values():
            assert isinstance(v, np.ndarray)


# ---------------------------------------------------------------------------
# Pipeline with backend parameter
# ---------------------------------------------------------------------------

class TestPipelineBackend:
    def test_compute_rdkit_lite(self):
        d = compute("C", backend="rdkit_lite")
        assert d.shape == (190,)
        assert not np.any(np.isnan(d))

    def test_compute_default_backend(self):
        d = compute("CC")
        assert d.shape == (190,)
        assert not np.any(np.isnan(d))

    def test_compute_batch_rdkit_lite(self):
        df = compute_batch(["C", "CC"], backend="rdkit_lite")
        assert df.shape == (2, 190)

    def test_compute_benzene_rdkit_lite(self):
        d = compute("c1ccccc1", backend="rdkit_lite")
        assert d.shape == (190,)
        assert not np.any(np.isnan(d))

    def test_compute_invalid_rdkit_lite(self):
        d = compute("INVALID", backend="rdkit_lite")
        assert d.shape == (190,)
        assert np.all(np.isnan(d))

    def test_compute_cyclic_rdkit_lite(self):
        d = compute("*CC*", backend="rdkit_lite", cyclic=10)
        assert d.shape == (190,)
        assert not np.any(np.isnan(d))

    def test_mass_sum_rdkit_lite(self):
        d = compute("C", backend="rdkit_lite")
        mass_sum = np.sum(d[:10])
        assert pytest.approx(mass_sum, abs=1e-10) == 1.0


# ---------------------------------------------------------------------------
# pSMILES cyclic polymer generation
# ---------------------------------------------------------------------------

class TestPSMILESCyclicPolymer:
    """Tests for pSMILES cyclic polymer generation in rdkit_lite."""

    def _make_polymer(self, smiles, n=10):
        from gaff2kme.backends.rdkit_lite import RdkitLiteBackend
        be = RdkitLiteBackend()
        return be.smiles_to_mol(smiles, cyclic=n)

    def test_pe_no_wildcards_in_result(self):
        """Cyclic polymer must not contain wildcard atoms."""
        mol = self._make_polymer("*CC*")
        assert mol is not None
        assert all(a.GetSymbol() != "*" for a in mol.GetAtoms())

    def test_pe_atom_count(self):
        """PE n=10: 20 C + 40 H = 60 atoms."""
        mol = self._make_polymer("*CC*", n=10)
        assert mol is not None
        assert mol.GetNumAtoms() == 60

    def test_pe_assign_succeeds(self):
        """Cyclic PE must pass force field assignment."""
        from gaff2kme.backends.rdkit_lite import RdkitLiteBackend
        be = RdkitLiteBackend()
        mol = be.smiles_to_mol("*CC*", cyclic=10)
        assert mol is not None
        assert be.assign(mol) is True

    def test_asymmetric_monomer(self):
        """Asymmetric monomer produces valid cyclic polymer."""
        mol = self._make_polymer("*C(=O)OCC*", n=3)
        assert mol is not None
        assert all(a.GetSymbol() != "*" for a in mol.GetAtoms())

    def test_aromatic_monomer(self):
        """Aromatic pSMILES produces valid cyclic polymer."""
        mol = self._make_polymer("*c1ccccc1*", n=3)
        assert mol is not None
        assert all(a.GetSymbol() != "*" for a in mol.GetAtoms())

    def test_double_bond_wildcard(self):
        """pSMILES with double bond at connection point is handled."""
        mol = self._make_polymer("*C=CC*", n=3)
        assert mol is not None
        assert all(a.GetSymbol() != "*" for a in mol.GetAtoms())

    def test_degenerate_single_atom_core_handled(self):
        """Degenerate *=C=* (single-atom core) is now handled by _make_cyclic_polymer_degenerate."""
        mol = self._make_polymer("*=C=*", n=3)
        # With the degenerate core handler, this now succeeds
        assert mol is not None

    def test_non_psmiles_passthrough(self):
        """Non-pSMILES (no wildcards) is not affected by cyclic parameter."""
        from gaff2kme.backends.rdkit_lite import RdkitLiteBackend
        be = RdkitLiteBackend()
        mol = be.smiles_to_mol("CC", cyclic=10)
        assert mol is not None

    def test_pe_full_pipeline(self):
        """Full pipeline compute for pSMILES produces valid 190-dim descriptor."""
        d = compute("*CC*", backend="rdkit_lite", cyclic=10)
        assert d.shape == (190,)
        assert not np.any(np.isnan(d))
