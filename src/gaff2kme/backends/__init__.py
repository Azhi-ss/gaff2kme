"""Backend abstraction for force field assignment."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FFBackend(Protocol):
    """Protocol that all force field backends must implement."""

    @property
    def name(self) -> str: ...

    def assign(self, mol, charge: str | None = "gasteiger") -> bool: ...

    def get_param_db(self): ...

    def smiles_to_mol(self, smiles: str, cyclic: int = 10):
        """Parse SMILES to RDKit mol. Returns None on failure."""
        ...


_REGISTRY: dict[str, str] = {
    "rdkit_lite": "gaff2kme.backends.rdkit_lite:RdkitLiteBackend",
    "radonpy": "gaff2kme.backends.radonpy_backend:RadonPyBackend",
}


def get_backend(name: str = "rdkit_lite", ff_name: str = "gaff2_mod") -> FFBackend:
    """Get a force field backend by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend: {name}. Valid: {list(_REGISTRY)}")
    module_path, cls_name = _REGISTRY[name].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)(ff_name=ff_name)
