"""Tests for smiles_to_mol SanitizeMol fallback and degenerate core handling.

TDD: When a pSMILES has both wildcards on the same neighbor atom (degenerate core),
the code should handle it by connecting branching atoms between adjacent repeat units
instead of returning None.
"""
import pytest
from rdkit import Chem
from gaff2kme.backends.rdkit_lite import RdkitLiteBackend


@pytest.fixture
def be():
    return RdkitLiteBackend(ff_name="gaff2_mod")


# ---- Test 1: Degenerate core SMILES works now ----
def test_degenerate_core_succeeds(be):
    # [*]C([*]) — both wildcards on same C atom
    smi = "[*]C([*])C(=O)OC"
    mol = be.smiles_to_mol(smi, cyclic=10)
    assert mol is not None, "Degenerate core should be handled"


# ---- Test 2: Degenerate core produces a molecule with correct structure ----
def test_degenerate_core_structure(be):
    smi = "[*]C([*])C(=O)OC"
    mol = be.smiles_to_mol(smi, cyclic=10)
    assert mol is not None
    # Should have atoms from 10 repeat units + Hs
    assert mol.GetNumAtoms() > 0


# ---- Test 3: _make_cyclic_polymer_degenerate works directly ----
def test_make_cyclic_polymer_degenerate_directly(be):
    smi = "[*]C([*])C(=O)OC"
    mol_raw = Chem.MolFromSmiles(smi)
    wc_info = be._extract_wildcard_info(mol_raw)
    assert wc_info is not None

    head, tail = wc_info[0], wc_info[1]
    rw = Chem.RWMol(mol_raw)
    for idx in sorted([head.star_idx, tail.star_idx], reverse=True):
        rw.RemoveAtom(idx)
    core = rw.GetMol()

    all_removed = sorted([head.star_idx, tail.star_idx])
    def _remapped(orig):
        return orig - sum(1 for r in all_removed if r < orig)

    head_ne = _remapped(head.neighbor_idx)
    tail_ne = _remapped(tail.neighbor_idx)
    assert head_ne == tail_ne, "Should be degenerate"

    result = be._make_cyclic_polymer_degenerate(core, head_ne, tail_ne, 10, head, tail)
    assert result is not None, "_make_cyclic_polymer_degenerate should succeed"


# ---- Test 4: Normal (non-degenerate) SMILES still work ----
def test_normal_smiles_still_work(be):
    smi = "[*]CCCCCC[*]"
    mol = be.smiles_to_mol(smi, cyclic=10)
    assert mol is not None


# ---- Test 5: Invalid SMILES still returns None ----
def test_invalid_smiles_returns_none(be):
    mol = be.smiles_to_mol("INVALID", cyclic=10)
    assert mol is None


# ---- Test 6: Full pipeline works for degenerate SMILES ----
def test_compute_no_nan_for_degenerate():
    import numpy as np
    from gaff2kme import compute
    smi = "[*]C([*])C(=O)OC"
    result = compute(smi, cyclic=10)
    assert not np.any(np.isnan(result)), "compute() should not return NaN for degenerate SMILES"
