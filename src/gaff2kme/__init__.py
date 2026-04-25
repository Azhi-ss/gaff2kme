"""gaff2kme: SMILES to 190-dim KME descriptors via GAFF2_mod force field.

Works out of the box with `backend="rdkit_lite"` (default, pure RDKit).
Optionally use `backend="radonpy"` if RadonPy is installed.
"""

__version__ = "0.2.0"

from .math import compute_kme, kme_desc_names, gaussian_kernel, kernel_mean, MinMaxScale
from .extraction import smiles_to_assigned_mol, extract_params, build_scaling_factors
from .pipeline import compute, compute_batch, compute_from_csv, desc_names
from .backends import get_backend

__all__ = [
    "compute_kme",
    "kme_desc_names",
    "gaussian_kernel",
    "kernel_mean",
    "MinMaxScale",
    "smiles_to_assigned_mol",
    "extract_params",
    "build_scaling_factors",
    "compute",
    "compute_batch",
    "compute_from_csv",
    "desc_names",
    "get_backend",
]
