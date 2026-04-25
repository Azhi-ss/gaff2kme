"""Layer 3: End-to-end pipeline combining math + extraction layers.

Provides the public API: compute, compute_batch, compute_from_csv, desc_names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backends import get_backend
from .math import FF_MASS, MinMaxScale, compute_kme, kme_desc_names
from .extraction import extract_params, build_scaling_factors


def _n_desc(polar: bool = True) -> int:
    return len(kme_desc_names(polar=polar))


def _build_scaling_from_backend(backend, polar: bool = True) -> dict:
    """Build scaling factors from backend's param database."""
    param = backend.get_param_db()
    ff_epsilon = []
    ff_sigma = []
    for key, val in param.pt.items():
        if key in frozenset({"hw", "n4", "nx", "ny", "nz", "n+"}):
            continue
        ff_epsilon.append(val.epsilon)
        ff_sigma.append(val.sigma)

    ff_k_bond = []
    ff_r0 = []
    for val in param.bt.values():
        if val.k < 20:
            continue
        ff_k_bond.append(val.k)
        ff_r0.append(val.r0)

    ff_k_angle = []
    ff_theta0 = []
    for val in param.at.values():
        if val.k < 20:
            continue
        ff_k_angle.append(val.k)
        ff_theta0.append(val.theta0)

    ff_k_dih = []
    for val in param.dt.values():
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


def compute(
    smiles: str,
    ff_name: str = "gaff2_mod",
    polar: bool = True,
    cyclic: int = 10,
    nk: int = 20,
    backend: str = "rdkit_lite",
) -> np.ndarray:
    """Compute 190-dim KME descriptor for a single SMILES string."""
    be = get_backend(backend, ff_name=ff_name)
    scaling = _build_scaling_from_backend(be, polar=polar)
    mol = be.smiles_to_mol(smiles, cyclic=cyclic)
    if mol is None:
        return np.full(_n_desc(polar), np.nan)
    if not be.assign(mol, charge="gasteiger"):
        return np.full(_n_desc(polar), np.nan)
    params = extract_params(mol, polar=polar)
    return compute_kme(params, scaling, polar=polar, nk=nk)


def compute_batch(
    smiles_list: list[str],
    ff_name: str = "gaff2_mod",
    polar: bool = True,
    cyclic: int = 10,
    nk: int = 20,
    mp: int | None = None,
    backend: str = "rdkit_lite",
) -> pd.DataFrame:
    """Compute KME descriptors for multiple SMILES strings."""
    names = kme_desc_names(polar=polar, nk=nk)
    results = []
    for smi in smiles_list:
        d = compute(smi, ff_name=ff_name, polar=polar, cyclic=cyclic, nk=nk,
                    backend=backend)
        results.append(d)
    return pd.DataFrame(results, columns=names)


def compute_from_csv(
    path: str,
    smiles_col: str = "SMILES",
    ff_name: str = "gaff2_mod",
    polar: bool = True,
    cyclic: int = 10,
    nk: int = 20,
    mp: int | None = None,
    output: str | None = None,
    backend: str = "rdkit_lite",
) -> pd.DataFrame:
    """Compute KME descriptors from a CSV file with a SMILES column."""
    df_in = pd.read_csv(path)
    smiles_list = df_in[smiles_col].tolist()
    df_out = compute_batch(
        smiles_list, ff_name=ff_name, polar=polar, cyclic=cyclic, nk=nk,
        mp=mp, backend=backend,
    )
    df_out.index = df_in.index
    if output is not None:
        df_out.to_csv(output, index=False)
    return df_out


def desc_names(polar: bool = True, nk: int = 20) -> list[str]:
    """Return descriptor column names."""
    return kme_desc_names(polar=polar, nk=nk)
