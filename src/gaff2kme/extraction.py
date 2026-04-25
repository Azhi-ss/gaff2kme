"""Layer 2: Parameter extraction from SMILES via RadonPy force field assignment.

Parses SMILES, assigns GAFF2/GAFF2_mod force field, and extracts parameter
arrays for KME computation. External imports (rdkit, radonpy) are resolved
lazily so the module can be imported without them; they are patched in tests.
"""

from __future__ import annotations

import importlib
import numpy as np

from .math import FF_MASS, MinMaxScale

IGNORE_PT = frozenset({"hw", "n4", "nx", "ny", "nz", "n+"})

FF_REGISTRY: dict[str, str] = {
    "gaff2_mod": "radonpy.ff.gaff2_mod:GAFF2_mod",
    "gaff2": "radonpy.ff.gaff2:GAFF2",
    "gaff": "radonpy.ff.gaff:GAFF",
}

# Lazily populated; patchable in tests
Chem = None
poly = None


def _ensure_imports():
    global Chem, poly
    if Chem is None:
        try:
            from rdkit import Chem as _Chem
            Chem = _Chem
        except ImportError:
            pass
    if poly is None:
        try:
            from radonpy.core import poly as _poly
            poly = _poly
        except ImportError:
            pass


def create_force_field(ff_name: str = "gaff2_mod"):
    """Create a force field object by name."""
    if ff_name not in FF_REGISTRY:
        raise ValueError(f"Unknown force field: {ff_name}. Valid: {list(FF_REGISTRY)}")
    module_path, cls_name = FF_REGISTRY[ff_name].rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)()


def smiles_to_assigned_mol(smiles: str, ff, cyclic: int = 10):
    """Parse SMILES and assign force field. Returns None on failure."""
    _ensure_imports()
    mol = None
    if cyclic > 0 and smiles.count("*") == 2:
        try:
            mol = poly.make_cyclicpolymer(smiles, n=cyclic, return_mol=True, removeHs=False)
        except Exception:
            return None
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)

    if mol is None:
        return None

    try:
        result = ff.ff_assign(mol, charge="gasteiger")
        if not result:
            return None
    except Exception:
        return None

    return mol


def extract_params(mol, polar: bool = True, ignoreH: bool = False) -> dict:
    """Extract force field parameter arrays from an assigned molecule."""
    mass = []
    charge = []
    epsilon = []
    sigma = []
    for atom in mol.GetAtoms():
        if ignoreH and atom.GetSymbol() == "H":
            continue
        mass.append(atom.GetMass())
        charge.append(atom.GetDoubleProp("AtomicCharge"))
        epsilon.append(atom.GetDoubleProp("ff_epsilon"))
        sigma.append(atom.GetDoubleProp("ff_sigma"))

    k_bond = []
    r0 = []
    polar_arr = []
    for b in mol.GetBonds():
        if ignoreH and (b.GetBeginAtom().GetSymbol() == "H" or b.GetEndAtom().GetSymbol() == "H"):
            continue
        k_bond.append(b.GetDoubleProp("ff_k"))
        r0.append(b.GetDoubleProp("ff_r0"))
        if polar:
            polar_arr.append(
                abs(b.GetBeginAtom().GetDoubleProp("AtomicCharge")
                    - b.GetEndAtom().GetDoubleProp("AtomicCharge"))
            )

    k_angle = []
    theta0 = []
    for ang in mol.angles.values():
        if ignoreH and (
            mol.GetAtomWithIdx(ang.a).GetSymbol() == "H"
            or mol.GetAtomWithIdx(ang.b).GetSymbol() == "H"
            or mol.GetAtomWithIdx(ang.c).GetSymbol() == "H"
        ):
            continue
        k_angle.append(ang.ff.k)
        theta0.append(ang.ff.theta0)

    k_dih = []
    for dih in mol.dihedrals.values():
        if ignoreH and (
            mol.GetAtomWithIdx(dih.a).GetSymbol() == "H"
            or mol.GetAtomWithIdx(dih.b).GetSymbol() == "H"
            or mol.GetAtomWithIdx(dih.c).GetSymbol() == "H"
            or mol.GetAtomWithIdx(dih.d).GetSymbol() == "H"
        ):
            continue
        idx = int(np.argmax(dih.ff.k))
        k_dih.append(dih.ff.k[idx])

    result = {
        "mass": np.array(mass, dtype=float),
        "charge": np.array(charge, dtype=float),
        "epsilon": np.array(epsilon, dtype=float),
        "sigma": np.array(sigma, dtype=float),
        "k_bond": np.array(k_bond, dtype=float),
        "r0": np.array(r0, dtype=float),
        "k_angle": np.array(k_angle, dtype=float),
        "theta0": np.array(theta0, dtype=float),
        "k_dih": np.array(k_dih, dtype=float),
    }
    if polar:
        result["polar"] = np.array(polar_arr, dtype=float)
    return result


def build_scaling_factors(ff, polar: bool = True) -> dict:
    """Build MinMaxScale objects from force field parameter database."""
    ff_epsilon = []
    ff_sigma = []
    for key, val in ff.param.pt.items():
        if key in IGNORE_PT:
            continue
        ff_epsilon.append(val.epsilon)
        ff_sigma.append(val.sigma)

    ff_k_bond = []
    ff_r0 = []
    for val in ff.param.bt.values():
        if val.k < 20:
            continue
        ff_k_bond.append(val.k)
        ff_r0.append(val.r0)

    ff_k_angle = []
    ff_theta0 = []
    for val in ff.param.at.values():
        if val.k < 20:
            continue
        ff_k_angle.append(val.k)
        ff_theta0.append(val.theta0)

    ff_k_dih = []
    for val in ff.param.dt.values():
        idx = int(np.argmax(val.k))
        ff_k_dih.append(val.k[idx])

    scaling = {
        "mass": MinMaxScale(FF_MASS),
        "charge": MinMaxScale([-1.0, 1.0]),
        "epsilon": MinMaxScale(ff_epsilon),
        "sigma": MinMaxScale(ff_sigma),
        "k_bond": MinMaxScale(ff_k_bond),
        "r0": MinMaxScale(ff_r0),
        "k_angle": MinMaxScale(ff_k_angle),
        "theta0": MinMaxScale(ff_theta0),
        "k_dih": MinMaxScale(ff_k_dih),
    }
    if polar:
        scaling["polar"] = MinMaxScale([0.0, 2.0])
    return scaling
