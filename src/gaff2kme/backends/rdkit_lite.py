"""rdkit_lite backend: pure RDKit + bundled JSON, no RadonPy dependency.

Replicates RadonPy's GAFF/GAFF2/GAFF2_mod atom typing, bond/angle/dihedral
parameter assignment, and Gasteiger charge calculation using only RDKit and
the bundled force field JSON data files.
"""

from __future__ import annotations

import json
import os
from itertools import permutations
from typing import Any, NamedTuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


_FF_DAT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "ff_dat"
)

IGNORE_PT = frozenset({"hw", "n4", "nx", "ny", "nz", "n+"})

ELCTRWD_ELEMENTS = frozenset({"N", "O", "F", "Cl", "Br", "I"})

ALT_PTYPE_BASE: dict[str, str] = {
    "cc": "c2", "cd": "c2", "ce": "c2", "cf": "c2", "cg": "c1", "ch": "c1",
    "cp": "ca", "cq": "ca", "cu": "c2", "cv": "c2", "cx": "c3", "cy": "c3",
    "h1": "hc", "h2": "hc", "h3": "hc", "h4": "ha", "h5": "ha",
    "nb": "nc", "nc": "n2", "nd": "n2", "ne": "n2", "nf": "n2",
    "pb": "pc", "pc": "p2", "pd": "p2", "pe": "p2", "pf": "p2",
    "p4": "px", "px": "p4", "p5": "py", "py": "p5",
    "s4": "sx", "sx": "s4", "s6": "sy", "sy": "s6",
}

ALT_PTYPE_GAFF2_MOD: dict[str, str] = {
    "c3f": "c3", "ci": "c3", "ng": "n3",
}


class _WildcardInfo(NamedTuple):
    """Connection point info from a pSMILES wildcard atom."""
    star_idx: int
    neighbor_idx: int
    bond_type_as_double: float


class _ParamContainer:
    """Simple container for force field parameters loaded from JSON."""

    def __init__(self) -> None:
        self.pt: dict[str, Any] = {}
        self.bt: dict[str, Any] = {}
        self.at: dict[str, Any] = {}
        self.dt: dict[str, Any] = {}
        self.it: dict[str, Any] = {}


class _PtEntry:
    __slots__ = ("tag", "name", "elem", "mass", "epsilon", "sigma")

    def __init__(self, tag: str, name: str, elem: str, mass: float,
                 epsilon: float, sigma: float) -> None:
        self.tag = tag
        self.name = name
        self.elem = elem
        self.mass = mass
        self.epsilon = epsilon
        self.sigma = sigma


class _BtEntry:
    __slots__ = ("tag", "name", "rname", "k", "r0")

    def __init__(self, tag: str, name: str, rname: str, k: float, r0: float) -> None:
        self.tag = tag
        self.name = name
        self.rname = rname
        self.k = k
        self.r0 = r0


class _AtEntry:
    __slots__ = ("tag", "name", "rname", "k", "theta0")

    def __init__(self, tag: str, name: str, rname: str, k: float, theta0: float) -> None:
        self.tag = tag
        self.name = name
        self.rname = rname
        self.k = k
        self.theta0 = theta0


class _DtEntry:
    __slots__ = ("tag", "name", "rname", "k", "d", "m", "n")

    def __init__(self, tag: str, name: str, rname: str,
                 k: list[float], d: list[float], m: int, n: list[int]) -> None:
        self.tag = tag
        self.name = name
        self.rname = rname
        self.k = k
        self.d = d
        self.m = m
        self.n = n


class _AngleObj:
    __slots__ = ("a", "b", "c", "ff")

    def __init__(self, a: int, b: int, c: int, ff: Any) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.ff = ff


class _AngleFF:
    __slots__ = ("type", "k", "theta0")

    def __init__(self, type: str, k: float, theta0: float) -> None:
        self.type = type
        self.k = k
        self.theta0 = theta0


class _DihedralObj:
    __slots__ = ("a", "b", "c", "d", "ff")

    def __init__(self, a: int, b: int, c: int, d: int, ff: Any) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.ff = ff


class _DihedralFF:
    __slots__ = ("type", "k", "d", "m", "n")

    def __init__(self, type: str, k: list[float], d: list[float],
                 m: int, n: list[int]) -> None:
        self.type = type
        self.k = k
        self.d = d
        self.m = m
        self.n = n


def _load_ff_json(json_path: str) -> _ParamContainer:
    """Load force field parameters from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    param = _ParamContainer()

    for pt in data.get("particle_types", []):
        entry = _PtEntry(
            tag=pt["tag"], name=pt["name"], elem=pt.get("elem", ""),
            mass=pt.get("mass", 0.0), epsilon=pt["epsilon"], sigma=pt["sigma"],
        )
        param.pt[pt["name"]] = entry

    for bt in data.get("bond_types", []):
        entry = _BtEntry(
            tag=bt["tag"], name=bt["name"], rname=bt["rname"],
            k=bt["k"], r0=bt["r0"],
        )
        param.bt[bt["name"]] = entry
        param.bt[bt["rname"]] = entry

    for at in data.get("angle_types", []):
        entry = _AtEntry(
            tag=at["tag"], name=at["name"], rname=at["rname"],
            k=at["k"], theta0=at["theta0"],
        )
        param.at[at["name"]] = entry
        param.at[at["rname"]] = entry

    for dt in data.get("dihedral_types", []):
        entry = _DtEntry(
            tag=dt["tag"], name=dt["name"], rname=dt["rname"],
            k=dt["k"], d=dt["d"], m=dt["m"], n=dt["n"],
        )
        param.dt[dt["name"]] = entry
        param.dt[dt["rname"]] = entry

    for it in data.get("improper_types", []):
        param.it[it["name"]] = it

    return param


class RdkitLiteBackend:
    """Pure RDKit + bundled JSON force field backend. No RadonPy required."""

    def __init__(self, ff_name: str = "gaff2_mod") -> None:
        ff_file = os.path.join(_FF_DAT_DIR, f"{ff_name}.json")
        if not os.path.isfile(ff_file):
            raise ValueError(f"Force field data not found: {ff_file}")
        self._ff_name = ff_name
        self.param = _load_ff_json(ff_file)
        self.alt_ptype = dict(ALT_PTYPE_BASE)
        if ff_name == "gaff2_mod":
            self.alt_ptype.update(ALT_PTYPE_GAFF2_MOD)
        self.max_ring_size = 14

    @property
    def name(self) -> str:
        return self._ff_name

    def get_param_db(self) -> _ParamContainer:
        return self.param

    def smiles_to_mol(self, smiles: str, cyclic: int = 10):
        """Parse SMILES, optionally create cyclic polymer. Returns None on failure."""
        if cyclic > 0 and smiles.count("*") == 2:
            mol = self._make_cyclic_polymer(smiles, cyclic)
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
        return mol

    def _make_cyclic_polymer(self, smiles: str, n: int):
        """Create a cyclic n-mer from pSMILES by polymerizing repeat units and closing the ring.

        Algorithm: parse pSMILES → find 2 wildcard atoms → delete them to get repeat core
        → copy n cores via InsertMol → add inter-core bonds → close ring → SanitizeMol → AddHs.
        Returns None on any failure.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        wc_info = self._extract_wildcard_info(mol)
        if wc_info is None or len(wc_info) != 2:
            return None

        head, tail = wc_info[0], wc_info[1]

        # Build repeat core by deleting wildcard atoms (high idx first)
        rw = Chem.RWMol(mol)
        for idx in sorted([head.star_idx, tail.star_idx], reverse=True):
            rw.RemoveAtom(idx)
        core = rw.GetMol()

        # Remap neighbor indices after atom removal
        all_removed = sorted([head.star_idx, tail.star_idx])
        def _remapped(orig: int) -> int:
            return orig - sum(1 for r in all_removed if r < orig)

        head_ne = _remapped(head.neighbor_idx)
        tail_ne = _remapped(tail.neighbor_idx)

        if head_ne == tail_ne:
            return None  # degenerate single-atom core

        # Bond type resolution: max of head/tail connection bond orders
        head_bt, tail_bt = head.bond_type_as_double, tail.bond_type_as_double
        if head_bt == 2.0 or tail_bt == 2.0:
            new_bond_type = Chem.rdchem.BondType.DOUBLE
        elif head_bt == 3.0 or tail_bt == 3.0:
            new_bond_type = Chem.rdchem.BondType.TRIPLE
        else:
            new_bond_type = Chem.rdchem.BondType.SINGLE

        # Assemble n-mer by inserting n copies of the repeat core
        combined = Chem.RWMol()
        offsets = []
        for _ in range(n):
            offsets.append(combined.GetNumAtoms())
            combined.InsertMol(core)

        # Add inter-core bonds (cyclic: last core's tail connects to first core's head)
        for i in range(n):
            next_i = (i + 1) % n
            tail_global = offsets[i] + tail_ne
            head_global = offsets[next_i] + head_ne
            combined.AddBond(tail_global, head_global, new_bond_type)

        mol_out = combined.GetMol()
        try:
            Chem.SanitizeMol(mol_out)
            mol_out = Chem.AddHs(mol_out)
        except Exception:
            return None

        return mol_out

    @staticmethod
    def _extract_wildcard_info(mol) -> list[_WildcardInfo] | None:
        """Extract connection point info from wildcard atoms in a pSMILES mol."""
        info: list[_WildcardInfo] = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 0:
                continue
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                return None  # wildcard must have exactly 1 neighbor
            ne = neighbors[0]
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), ne.GetIdx())
            info.append(_WildcardInfo(
                star_idx=atom.GetIdx(),
                neighbor_idx=ne.GetIdx(),
                bond_type_as_double=bond.GetBondTypeAsDouble(),
            ))
        return info

    def assign(self, mol, charge: str | None = "gasteiger") -> bool:
        """Assign force field parameters to an RDKit mol object."""
        Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)
        Chem.rdmolops.SetAromaticity(mol, model=Chem.rdmolops.AromaticityModel.AROMATICITY_MDL)

        mol.SetProp("ff_name", self._ff_name)
        mol.SetProp("ff_class", "1")
        mol.SetProp("pair_style", "lj")
        mol.SetProp("bond_style", "harmonic")
        mol.SetProp("angle_style", "harmonic")
        mol.SetProp("dihedral_style", "fourier")
        mol.SetProp("improper_style", "cvff")

        result = self._assign_ptypes(mol)
        if result:
            result = self._assign_btypes(mol)
        if result:
            result = self._assign_atypes(mol)
        if result:
            result = self._assign_dtypes(mol)
        if result and charge is not None:
            result = self._assign_charges(mol, charge)
        return result

    def _set_ptype(self, atom, pt: str) -> None:
        atom.SetProp("ff_type", pt)
        if pt in self.param.pt:
            atom.SetDoubleProp("ff_epsilon", self.param.pt[pt].epsilon)
            atom.SetDoubleProp("ff_sigma", self.param.pt[pt].sigma)

    def _assign_ptypes(self, mol) -> bool:
        result_flag = True
        for atom in mol.GetAtoms():
            if not self._assign_ptype_atom(atom):
                result_flag = False
        return result_flag

    def _assign_ptype_atom(self, p) -> bool:
        symbol = p.GetSymbol()
        if symbol == "*":
            p.SetProp("ff_type", "*")
            p.SetDoubleProp("ff_epsilon", 0.0)
            p.SetDoubleProp("ff_sigma", 0.0)
            return True
        if symbol == "H":
            return self._type_hydrogen(p)
        if symbol == "C":
            return self._type_carbon(p)
        if symbol == "N":
            return self._type_nitrogen(p)
        if symbol == "O":
            return self._type_oxygen(p)
        if symbol in ("F", "Cl", "Br", "I"):
            p.SetProp("ff_type", symbol.lower())
            self._set_ptype(p, symbol.lower())
            return True
        if symbol == "P":
            return self._type_phosphorus(p)
        if symbol == "S":
            return self._type_sulfur(p)
        if symbol == "Si":
            p.SetProp("ff_type", "si")
            self._set_ptype(p, "si")
            return True
        return False

    def _type_hydrogen(self, p) -> bool:
        neighbors = p.GetNeighbors()
        if not neighbors:
            return False
        nb = neighbors[0]
        nb_sym = nb.GetSymbol()

        if nb_sym == "O":
            for pb in neighbors:
                if pb.GetSymbol() == "O" and pb.GetTotalNumHs(includeNeighbors=True) == 2:
                    self._set_ptype(p, "hw")
                    return True
            self._set_ptype(p, "ho")
            return True
        if nb_sym == "N":
            self._set_ptype(p, "hn")
            return True
        if nb_sym == "P":
            self._set_ptype(p, "hp")
            return True
        if nb_sym == "S":
            self._set_ptype(p, "hs")
            return True
        if nb_sym == "Si":
            self._set_ptype(p, "hi")
            return True
        if nb_sym == "C":
            for pb in neighbors:
                if pb.GetSymbol() == "C":
                    elctrwd = 0
                    degree = pb.GetTotalDegree()
                    for pbb in pb.GetNeighbors():
                        if pbb.GetSymbol() in ELCTRWD_ELEMENTS and pbb.GetTotalDegree() < 4:
                            elctrwd += 1
                    if elctrwd == 0:
                        hyb = str(pb.GetHybridization())
                        self._set_ptype(p, "ha" if hyb in ("SP2", "SP") else "hc")
                    elif degree == 4 and elctrwd == 1:
                        self._set_ptype(p, "h1")
                    elif degree == 4 and elctrwd == 2:
                        self._set_ptype(p, "h2")
                    elif degree == 4 and elctrwd == 3:
                        self._set_ptype(p, "h3")
                    elif degree == 3 and elctrwd == 1:
                        self._set_ptype(p, "h4")
                    elif degree == 3 and elctrwd == 2:
                        self._set_ptype(p, "h5")
                    else:
                        self._set_ptype(p, "hc")
                    return True
            self._set_ptype(p, "hc")
            return True
        self._set_ptype(p, "hc")
        return True

    def _type_carbon(self, p) -> bool:
        hyb = str(p.GetHybridization())
        if hyb == "SP3":
            fluoro = any(nb.GetSymbol() == "F" for nb in p.GetNeighbors())
            silicon_flag = sum(1 for nb in p.GetNeighbors() if nb.GetSymbol() == "Si")
            if p.IsInRingSize(3):
                self._set_ptype(p, "cx")
            elif p.IsInRingSize(4):
                self._set_ptype(p, "cy")
            elif fluoro:
                self._set_ptype(p, "c3f" if "c3f" in self.param.pt else "c3")
            elif silicon_flag >= 1:
                self._set_ptype(p, "ci" if "ci" in self.param.pt else "c3")
            else:
                self._set_ptype(p, "c3")
            return True
        if hyb == "SP2":
            p_idx = p.GetIdx()
            carbonyl = False
            conj = 0
            for pb in p.GetNeighbors():
                if pb.GetSymbol() == "O":
                    for b in pb.GetBonds():
                        if b.GetBondTypeAsDouble() == 2 and pb.GetTotalDegree() == 1:
                            carbonyl = True
            for b in p.GetBonds():
                if b.GetIsConjugated():
                    conj += 1
            if carbonyl:
                self._set_ptype(p, "c")
            elif p.GetIsAromatic():
                self._set_ptype(p, "ca")
            elif p.IsInRingSize(3):
                self._set_ptype(p, "cu")
            elif p.IsInRingSize(4):
                self._set_ptype(p, "cv")
            elif conj >= 2:
                self._set_ptype(p, "cc")
            else:
                self._set_ptype(p, "c2")
            return True
        if hyb == "SP":
            conj = sum(1 for b in p.GetBonds() if b.GetIsConjugated())
            self._set_ptype(p, "cg" if conj >= 2 else "c1")
            return True
        self._set_ptype(p, "c3")
        return True

    def _type_nitrogen(self, p) -> bool:
        hyb = str(p.GetHybridization())
        degree = p.GetTotalDegree()
        if hyb == "SP":
            self._set_ptype(p, "n1")
            return True
        if degree == 2:
            if p.GetIsAromatic():
                self._set_ptype(p, "nb")
                return True
            bond_orders = [b.GetBondTypeAsDouble() for b in p.GetBonds()]
            self._set_ptype(p, "n2" if 2 in bond_orders else "n3")
            return True
        if degree == 3:
            numHs = p.GetTotalNumHs(includeNeighbors=True)
            if p.GetIsAromatic():
                self._set_ptype(p, "na")
                return True
            if numHs == 1:
                self._set_ptype(p, "n7")
            elif numHs == 2:
                self._set_ptype(p, "n8")
            elif numHs == 3:
                self._set_ptype(p, "n9")
            else:
                self._set_ptype(p, "n3")
            return True
        if degree == 4:
            numHs = p.GetTotalNumHs(includeNeighbors=True)
            if numHs == 1:
                self._set_ptype(p, "nx")
            elif numHs == 2:
                self._set_ptype(p, "ny")
            elif numHs == 3:
                self._set_ptype(p, "nz")
            else:
                self._set_ptype(p, "n4")
            return True
        self._set_ptype(p, "n3")
        return True

    def _type_oxygen(self, p) -> bool:
        if p.GetTotalDegree() == 1:
            self._set_ptype(p, "o")
            return True
        if p.GetTotalNumHs(includeNeighbors=True) == 2:
            self._set_ptype(p, "ow")
            return True
        if p.GetTotalNumHs(includeNeighbors=True) == 1:
            si_flag = sum(1 for nb in p.GetNeighbors() if nb.GetSymbol() == "Si")
            self._set_ptype(p, "oi" if si_flag == 1 else "oh")
            return True
        si_flag = sum(1 for nb in p.GetNeighbors() if nb.GetSymbol() == "Si")
        self._set_ptype(p, "oss" if si_flag == 2 else "os")
        return True

    def _type_phosphorus(self, p) -> bool:
        degree = p.GetTotalDegree()
        if p.GetIsAromatic():
            self._set_ptype(p, "pb")
            return True
        if degree == 2:
            self._set_ptype(p, "p2")
            return True
        if degree == 3:
            self._set_ptype(p, "p3")
            return True
        if degree >= 4:
            self._set_ptype(p, "p5")
            return True
        self._set_ptype(p, "p5")
        return True

    def _type_sulfur(self, p) -> bool:
        degree = p.GetTotalDegree()
        if degree == 1:
            self._set_ptype(p, "s")
            return True
        if degree == 2:
            if p.GetIsAromatic():
                self._set_ptype(p, "ss")
            elif p.GetTotalNumHs(includeNeighbors=True) == 1:
                self._set_ptype(p, "sh")
            else:
                bond_orders = [b.GetBondTypeAsDouble() for b in p.GetBonds()]
                self._set_ptype(p, "s2" if 2 in bond_orders else "ss")
            return True
        if degree in (3, 4):
            self._set_ptype(p, "s4" if degree == 3 else "s6")
            return True
        self._set_ptype(p, "s6")
        return True

    def _assign_btypes(self, mol) -> bool:
        result_flag = True
        for b in mol.GetBonds():
            ba = b.GetBeginAtom().GetProp("ff_type")
            bb = b.GetEndAtom().GetProp("ff_type")
            bt = f"{ba},{bb}"
            if self._set_btype(b, bt):
                continue
            alt1 = self.alt_ptype.get(ba)
            alt2 = self.alt_ptype.get(bb)
            found = False
            if alt1 or alt2:
                candidates = []
                if alt1:
                    candidates.append(f"{alt1},{bb}")
                if alt2:
                    candidates.append(f"{ba},{alt2}")
                if alt1 and alt2:
                    candidates.append(f"{alt1},{alt2}")
                for c in candidates:
                    if self._set_btype(b, c):
                        found = True
                        break
            if not found:
                result_flag = False
        return result_flag

    def _set_btype(self, bond, bt: str) -> bool:
        if bt not in self.param.bt:
            return False
        entry = self.param.bt[bt]
        bond.SetProp("ff_type", entry.tag)
        bond.SetDoubleProp("ff_k", entry.k)
        bond.SetDoubleProp("ff_r0", entry.r0)
        return True

    def _assign_atypes(self, mol) -> bool:
        result_flag = True
        setattr(mol, "angles", {})
        for p in mol.GetAtoms():
            b = p.GetIdx()
            neighbors = list(p.GetNeighbors())
            for i, p1 in enumerate(neighbors):
                a = p1.GetIdx()
                for p2 in neighbors[i + 1:]:
                    c = p2.GetIdx()
                    key1 = f"{a},{b},{c}"
                    key2 = f"{c},{b},{a}"
                    if key1 in mol.angles or key2 in mol.angles:
                        continue
                    pt1 = p1.GetProp("ff_type")
                    pt = p.GetProp("ff_type")
                    pt2 = p2.GetProp("ff_type")
                    at = f"{pt1},{pt},{pt2}"
                    if not self._set_atype(mol, a, b, c, at):
                        alt1 = self.alt_ptype.get(pt1)
                        alt2 = self.alt_ptype.get(pt)
                        alt3 = self.alt_ptype.get(pt2)
                        found = False
                        combos = []
                        if alt1:
                            combos.append(f"{alt1},{pt},{pt2}")
                        if alt2:
                            combos.append(f"{pt1},{alt2},{pt2}")
                        if alt3:
                            combos.append(f"{pt1},{pt},{alt3}")
                        for combo in combos:
                            if self._set_atype(mol, a, b, c, combo):
                                found = True
                                break
                        if not found:
                            self._empirical_angle(mol, a, b, c, pt1, pt, pt2)
        return result_flag

    def _set_atype(self, mol, a: int, b: int, c: int, at: str) -> bool:
        if at not in self.param.at:
            return False
        entry = self.param.at[at]
        ff = _AngleFF(type=entry.tag, k=entry.k, theta0=entry.theta0)
        key = f"{a},{b},{c}"
        mol.angles[key] = _AngleObj(a=a, b=b, c=c, ff=ff)
        return True

    def _empirical_angle(self, mol, a: int, b: int, c: int,
                         pt1: str, pt: str, pt2: str) -> None:
        param_C = {"H": 0.0, "C": 1.339, "N": 1.3, "O": 1.249, "F": 0.0,
                   "Cl": 0.0, "Br": 0.0, "I": 0.0, "P": 0.906, "S": 1.448, "Si": 0.894}
        param_Z = {"H": 0.784, "C": 1.183, "N": 1.212, "O": 1.219, "F": 1.166,
                   "Cl": 1.272, "Br": 1.378, "I": 1.398, "P": 1.620, "S": 1.280, "Si": 1.016}
        elem1 = pt1.rstrip("0123456789").lstrip("h")
        elem = pt.rstrip("0123456789").lstrip("h")
        elem2 = pt2.rstrip("0123456789").lstrip("h")
        c1 = param_C.get(elem1, 0.0)
        c2 = param_C.get(elem, 0.0)
        c3 = param_C.get(elem2, 0.0)
        z1 = param_Z.get(elem1, 1.0)
        z2 = param_Z.get(elem, 1.0)
        z3 = param_Z.get(elem2, 1.0)
        if c1 + c2 + c3 == 0:
            return
        theta0 = c1 + c2 + c3
        k_ang = 0.5 * (z1 * z3) / (z2 * z2) * 100.0
        if k_ang < 20:
            return
        ff = _AngleFF(type=f"{pt1},{pt},{pt2}", k=k_ang, theta0=theta0)
        key = f"{a},{b},{c}"
        mol.angles[key] = _AngleObj(a=a, b=b, c=c, ff=ff)

    def _assign_dtypes(self, mol) -> bool:
        result_flag = True
        setattr(mol, "dihedrals", {})
        for bond in mol.GetBonds():
            p1 = bond.GetBeginAtom()
            p2 = bond.GetEndAtom()
            b = p1.GetIdx()
            c = p2.GetIdx()
            for p1b in p1.GetNeighbors():
                a = p1b.GetIdx()
                if c == a:
                    continue
                for p2b in p2.GetNeighbors():
                    d = p2b.GetIdx()
                    if b == d or a == d:
                        continue
                    key1 = f"{a},{b},{c},{d}"
                    key2 = f"{d},{c},{b},{a}"
                    if key1 in mol.dihedrals or key2 in mol.dihedrals:
                        continue
                    p1bt = p1b.GetProp("ff_type")
                    p1t = p1.GetProp("ff_type")
                    p2t = p2.GetProp("ff_type")
                    p2bt = p2b.GetProp("ff_type")
                    dt = f"{p1bt},{p1t},{p2t},{p2bt}"
                    if not self._set_dtype(mol, a, b, c, d, dt):
                        # Try wildcard match X for outer atoms
                        wildcards = [
                            f"X,{p1t},{p2t},X",
                            f"X,{p1t},{p2t},{p2bt}",
                            f"{p1bt},{p1t},{p2t},X",
                        ]
                        found = False
                        for wc in wildcards:
                            if self._set_dtype(mol, a, b, c, d, wc):
                                found = True
                                break
                        if not found:
                            alt1 = self.alt_ptype.get(p1t)
                            alt2 = self.alt_ptype.get(p2t)
                            if alt1 or alt2:
                                combos = []
                                if alt1:
                                    combos.append(f"X,{alt1},{p2t},X")
                                if alt2:
                                    combos.append(f"X,{p1t},{alt2},X")
                                for combo in combos:
                                    if self._set_dtype(mol, a, b, c, d, combo):
                                        found = True
                                        break
        return result_flag

    def _set_dtype(self, mol, a: int, b: int, c: int, d: int, dt: str) -> bool:
        if dt not in self.param.dt:
            return False
        entry = self.param.dt[dt]
        ff = _DihedralFF(type=entry.tag, k=entry.k, d=entry.d,
                         m=entry.m, n=entry.n)
        key = f"{a},{b},{c},{d}"
        mol.dihedrals[key] = _DihedralObj(a=a, b=b, c=c, d=d, ff=ff)
        return True

    def _assign_charges(self, mol, method: str) -> bool:
        if method == "gasteiger":
            try:
                AllChem.ComputeGasteigerCharges(mol)
                for atom in mol.GetAtoms():
                    charge_val = atom.GetDoubleProp("_GasteigerCharge")
                    if np.isnan(charge_val):
                        charge_val = 0.0
                    atom.SetDoubleProp("AtomicCharge", charge_val)
                return True
            except Exception:
                for atom in mol.GetAtoms():
                    atom.SetDoubleProp("AtomicCharge", 0.0)
                return True
        for atom in mol.GetAtoms():
            atom.SetDoubleProp("AtomicCharge", 0.0)
        return True
