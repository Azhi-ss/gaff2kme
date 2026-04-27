"""Regression tests for alkyne polymer SMILES bug (GitHub Issue #1).

When processing polymer SMILES with alkyne/triple bonds (e.g., [*]#CNCc1cccc(C#[*])c1),
the rdkit_lite backend previously failed because:

1. Bond type `cg,n7` (sp carbon bonded to sp3 nitrogen) had no entry in GAFF2_mod
   force field bond_types.
2. No BOND_TYPE_FALLBACK existed for `cg` atom type.
3. _assign_btypes left some bonds without ff_type, causing downstream KeyError.

Fix: Added `cg: ["c1"]` to BOND_TYPE_FALLBACK, cross-product fallback logic,
HasProp guards, and fixed _empirical_angle element extraction.
"""

import numpy as np
import pytest
from rdkit import Chem

from gaff2kme import compute
from gaff2kme.backends.rdkit_lite import BOND_TYPE_FALLBACK, RdkitLiteBackend


ALKYNE_SMILES_1 = "[*]#CNCc1cccc(C#[*])c1"
ALKYNE_SMILES_2 = "[*]CNC#Cc1cc([*])c(N)c(C)c1"


@pytest.fixture
def be():
    return RdkitLiteBackend(ff_name="gaff2_mod")


# ---- Test 1: assign() must not raise KeyError for alkyne polymer SMILES ----

def test_alkyne_smiles_no_keyerror(be):
    """Calling be.assign(mol) on alkyne polymer SMILES must not raise KeyError.

    Even if assign() returns False (indicating incomplete parameterization),
    it must not crash with an unhandled KeyError on missing 'ff_type' props.
    """
    mol = be.smiles_to_mol(ALKYNE_SMILES_1, cyclic=10)
    assert mol is not None, "Cyclic polymer molecule should be created"

    # The bug manifests as KeyError: 'ff_type' inside _assign_atypes
    # when _assign_btypes has failed to set ff_type on some bonds.
    # After the fix, assign() should return a bool (True or False) without
    # raising KeyError.
    try:
        result = be.assign(mol)
        assert isinstance(result, bool), "assign() should return a bool"
    except KeyError as exc:
        pytest.fail(f"assign() raised KeyError: {exc}")


# ---- Test 2: compute() must not return all-NaN for alkyne polymer SMILES ----

def test_alkyne_smiles_compute_not_all_nan():
    """compute() for alkyne pSMILES must not return an all-NaN vector.

    The descriptor may have partial NaN values (for parameters that genuinely
    cannot be resolved), but returning a completely NaN vector indicates an
    unhandled failure that should be fixed.
    """
    result = compute(ALKYNE_SMILES_1, cyclic=10)
    assert result.shape == (190,), "Descriptor must be 190-dimensional"
    assert not np.all(np.isnan(result)), (
        "compute() should not return all-NaN for alkyne polymer SMILES; "
        "partial NaN is acceptable but all NaN indicates a bug"
    )


# ---- Test 3: assign() survives missing bond type lookups ----

def test_alkyne_smiles_assign_survives_missing_bond_type(be):
    """When a bond type lookup fails, the code must not crash.

    The bond `cg,n7` is not in the force field bond_types. The assign()
    method should handle this gracefully -- either via fallback or by
    setting an empirical/default bond parameter -- without raising an
    exception.
    """
    mol = be.smiles_to_mol(ALKYNE_SMILES_1, cyclic=10)
    assert mol is not None

    # assign() must complete without raising any exception
    result = be.assign(mol)
    assert isinstance(result, bool), "assign() must return a bool, not raise"

    # After fix: at minimum, all atoms should have ff_type set
    for atom in mol.GetAtoms():
        assert atom.HasProp("ff_type"), (
            f"Atom {atom.GetIdx()} ({atom.GetSymbol()}) missing ff_type"
        )


# ---- Test 4: Second alkyne SMILES also works ----

def test_second_alkyne_smiles(be):
    """Same alkyne bug test for a different pSMILES structure.

    [*]CNC#Cc1cc([*])c(N)c(C)c1 has the alkyne in a different position
    but the same cg,n7 bond type mismatch.
    """
    mol = be.smiles_to_mol(ALKYNE_SMILES_2, cyclic=10)
    assert mol is not None, "Cyclic polymer molecule should be created"

    # Must not raise KeyError
    try:
        result = be.assign(mol)
        assert isinstance(result, bool), "assign() should return a bool"
    except KeyError as exc:
        pytest.fail(f"assign() raised KeyError for second alkyne: {exc}")

    # Pipeline should not return all-NaN
    desc = compute(ALKYNE_SMILES_2, cyclic=10)
    assert desc.shape == (190,)
    assert not np.all(np.isnan(desc)), (
        "compute() should not return all-NaN for second alkyne SMILES"
    )


# ---- Test 5: sp carbon atom types (cg, c1) exist in force field ----

def test_sp_carbon_types_exist_in_ff(be):
    """Verify that cg and c1 atom types exist in the force field param.pt.

    These are the GAFF2 sp carbon types assigned by _type_carbon() for
    SP-hybridized carbons (cg for conjugated, c1 for non-conjugated).
    They must exist in particle_types for epsilon/sigma assignment.
    """
    assert "cg" in be.param.pt, (
        "cg (sp conjugated carbon) must exist in force field particle types"
    )
    assert "c1" in be.param.pt, (
        "c1 (sp non-conjugated carbon) must exist in force field particle types"
    )


# ---- Test 6: BOND_TYPE_FALLBACK has entry for cg ----

def test_cg_bond_fallback_exists():
    """BOND_TYPE_FALLBACK must have an entry for the cg atom type.

    Without this fallback, bonds like cg,n7 fail lookup and leave bonds
    without ff_type, causing downstream KeyError crashes.
    """
    assert "cg" in BOND_TYPE_FALLBACK, (
        "cg must be in BOND_TYPE_FALLBACK to resolve missing bond types "
        "like cg,n7; suggested fallback: cg -> ['c1'] (sp carbon analog)"
    )
