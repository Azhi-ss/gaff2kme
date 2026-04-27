"""Tests for bond type fallback mechanism in rdkit_lite backend.

TDD: These tests define the EXPECTED behavior before implementation.
Currently will FAIL because n7/n8/n3 bond types are missing from the force field JSON.
"""
import pytest
import numpy as np
from rdkit import Chem
from gaff2kme.backends.rdkit_lite import RdkitLiteBackend


@pytest.fixture
def be():
    return RdkitLiteBackend(ff_name="gaff2_mod")


def _build_mol_with_ff_types(smiles, be, cyclic=10):
    """Helper: parse SMILES, build cyclic polymer, assign ptypes, return mol."""
    mol = be.smiles_to_mol(smiles, cyclic=cyclic)
    if mol is None:
        pytest.skip(f"Could not build mol from {smiles}")
    Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)
    Chem.rdmolops.SetAromaticity(
        mol, model=Chem.rdmolops.AromaticityModel.AROMATICITY_MDL
    )
    for atom in mol.GetAtoms():
        atom.SetProp("ff_name", be._ff_name)
    ptype_ok = be._assign_ptypes(mol)
    if not ptype_ok:
        pytest.skip(f"ptype assignment failed for {smiles}")
    return mol


# ---- Test 1: n7-c3 bond should fall back to n3-c3 ----
def test_n7_bond_falls_back(be):
    # SMILES that produces n7 atom type (verified from dataset)
    smi = "[*]CCCCCCCCOc1cccc(NC([*])=O)c1"
    mol = _build_mol_with_ff_types(smi, be)

    n7_atoms = [a for a in mol.GetAtoms()
                if a.HasProp("ff_type") and a.GetProp("ff_type") == "n7"]
    if not n7_atoms:
        pytest.skip("No n7 atom found in this molecule")

    result = be._assign_btypes(mol)
    assert result is True, "_assign_btypes should succeed with n7 fallback"

    # Verify n7 bonds got parameters assigned
    for n7 in n7_atoms:
        for bond in n7.GetBonds():
            assert bond.HasProp("ff_k"), "Bond from n7 should have ff_k via fallback"
            assert bond.HasProp("ff_r0"), "Bond from n7 should have ff_r0 via fallback"


# ---- Test 2: _assign_btypes returns True for real polymer SMILES ----
def test_assign_btypes_returns_true_with_fallback(be):
    smi = "[*]CCCCCCCCOc1cccc(NC([*])=O)c1"
    mol = _build_mol_with_ff_types(smi, be)

    result = be._assign_btypes(mol)
    assert result is True, "Should succeed with bond type fallback"


# ---- Test 3: Original keys still work unchanged ----
def test_original_bond_types_still_work(be):
    smi = "[*]CCCCCC[*]"
    mol = _build_mol_with_ff_types(smi, be)

    result = be._assign_btypes(mol)
    assert result is True, "Simple molecule should still work"

    for bond in mol.GetBonds():
        if bond.HasProp("ff_k"):
            assert bond.GetDoubleProp("ff_k") > 0, "ff_k should be positive"
            assert bond.GetDoubleProp("ff_r0") > 0, "ff_r0 should be positive"


# ---- Test 4: Full assign() pipeline works with fallback ----
def test_full_assign_with_fallback(be):
    smi = "[*]C(=O)N([*])C"
    mol = be.smiles_to_mol(smi, cyclic=10)
    if mol is None:
        pytest.skip("Could not build mol")

    result = be.assign(mol, charge="gasteiger")
    assert result is True, "Full assign should succeed with bond type fallback"


# ---- Test 5: Pipeline compute() no longer returns NaN for n7 molecules ----
def test_compute_no_nan_for_n7_molecule():
    from gaff2kme import compute
    smi = "[*]CCCCCCCCOc1cccc(NC([*])=O)c1"
    result = compute(smi, cyclic=10)
    assert not np.any(np.isnan(result)), "compute() should not return NaN for n7 molecule"


# ---- Test 6: n3 missing SP2 carbon partners (c,n3 ca,n3 cc,n3) fall back to n ----
def test_n3_missing_sp2_partners_fallback(be):
    smi = "[*]C(=O)NC([*])"
    mol = _build_mol_with_ff_types(smi, be)

    result = be._assign_btypes(mol)
    assert result is True, "n3 bonds with SP2 carbons should fall back to n"


# ---- Test 7: n3 with SP3 carbon partners still uses native n3 entries ----
def test_n3_sp3_partners_use_native(be):
    smi = "[*]CCNCC[*]"
    mol = _build_mol_with_ff_types(smi, be)

    n3_atoms = [a for a in mol.GetAtoms()
                if a.HasProp("ff_type") and a.GetProp("ff_type") == "n3"]
    if not n3_atoms:
        pytest.skip("No n3 atom found")

    result = be._assign_btypes(mol)
    assert result is True, "n3 with SP3 carbons should use native entries"
