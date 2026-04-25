"""RadonPy backend: delegates to RadonPy's GAFF2_mod force field.

Requires RadonPy to be installed. Falls back to RadonPy's full typing logic
including cyclic polymer support.
"""

from __future__ import annotations

import importlib

from rdkit import Chem


class RadonPyBackend:
    """Backend that delegates force field assignment to RadonPy."""

    def __init__(self, ff_name: str = "gaff2_mod") -> None:
        try:
            from radonpy.core import poly as _poly
            self._poly = _poly
        except ImportError:
            raise ImportError(
                "RadonPy is required for backend='radonpy'. "
                "Install it with: pip install -e /path/to/RadonPy"
            )
        self._ff_name = ff_name
        self._ff = self._create_ff(ff_name)

    def _create_ff(self, ff_name: str):
        registry = {
            "gaff2_mod": "radonpy.ff.gaff2_mod:GAFF2_mod",
            "gaff2": "radonpy.ff.gaff2:GAFF2",
            "gaff": "radonpy.ff.gaff:GAFF",
        }
        if ff_name not in registry:
            raise ValueError(f"Unknown force field: {ff_name}. Valid: {list(registry)}")
        module_path, cls_name = registry[ff_name].rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)()

    @property
    def name(self) -> str:
        return self._ff_name

    def get_param_db(self):
        return self._ff.param

    def smiles_to_mol(self, smiles: str, cyclic: int = 10):
        if cyclic > 0 and smiles.count("*") == 2:
            try:
                mol = self._poly.make_cyclicpolymer(
                    smiles, n=cyclic, return_mol=True, removeHs=False
                )
            except Exception:
                return None
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
        return mol

    def assign(self, mol, charge: str | None = "gasteiger") -> bool:
        try:
            return self._ff.ff_assign(mol, charge=charge)
        except Exception:
            return False
