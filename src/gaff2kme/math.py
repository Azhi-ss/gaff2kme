"""Layer 1: Pure Math KME Transform.

Converts force field parameter arrays into fixed-length Kernel Mean Embedding
descriptors using Gaussian kernels. Zero external dependencies beyond NumPy.
"""

from __future__ import annotations

import numpy as np

FF_MASS: list[float] = [
    1.008, 12.011, 14.007, 15.999, 18.998,
    30.974, 32.067, 35.453, 79.904, 126.904,
]
ELEMENT_NAMES: list[str] = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
PARAM_NAMES_WITH_POLAR: list[str] = [
    "charge", "epsilon", "sigma", "k_bond", "r0",
    "polar", "k_angle", "theta0", "k_dih",
]
PARAM_NAMES_NO_POLAR: list[str] = [
    "charge", "epsilon", "sigma", "k_bond", "r0",
    "k_angle", "theta0", "k_dih",
]


class MinMaxScale:
    """Min-max scaler that maps values from data range to [vmin, vmax]."""

    def __init__(
        self,
        data: list | np.ndarray,
        vmax: float = 1.0,
        vmin: float = 0.0,
    ) -> None:
        arr = np.array(data, dtype=float)
        self.dataMin = float(np.min(arr))
        self.scaleFactor = (float(np.max(arr)) - self.dataMin) * (vmax - vmin)
        self.vmin = vmin
        self.vmax = vmax

    def scale(self, values: list | np.ndarray) -> np.ndarray:
        v = np.array(values, dtype=float)
        if self.scaleFactor == 0.0:
            return np.full_like(v, self.vmin)
        return (v - self.dataMin) / self.scaleFactor + self.vmin

    def unscale(self, values: list | np.ndarray) -> np.ndarray:
        v = np.array(values, dtype=float)
        if self.scaleFactor == 0.0:
            return np.full_like(v, self.dataMin)
        return (v - self.vmin) * self.scaleFactor + self.dataMin


def gaussian_kernel(
    x: np.ndarray,
    xm: np.ndarray,
    sigma: float = 0.1,
) -> np.ndarray:
    return np.exp(-((x - xm) ** 2) / (2 * sigma ** 2))


def kernel_mean(
    vec: np.ndarray,
    centers: np.ndarray,
    kernel_func,
    weights: np.ndarray | None = None,
    sigma: float = 0.1,
) -> np.ndarray:
    if len(vec) == 0:
        return np.zeros(len(centers))
    x = vec.reshape(1, -1)
    xm = centers.reshape(-1, 1)
    y = kernel_func(x, xm, sigma=sigma)
    if weights is None:
        return np.mean(y, axis=1)
    w = np.array(weights, dtype=float)
    w = w / np.sum(w)
    return np.sum(y * w, axis=1)


def _compute_mass_fraction(mass_arr: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(FF_MASS))
    for mass_val in mass_arr:
        diffs = np.abs(np.array(FF_MASS) - mass_val)
        nearest = int(np.argmin(diffs))
        counts[nearest] += 1.0
    total = np.sum(counts)
    if total == 0.0:
        return counts
    return counts / total


def _make_kernel_centers(nk: int) -> np.ndarray:
    full = np.linspace(0.0, 1.0, nk * 2 + 1)
    return full[1:-1:2]


def compute_kme(
    params: dict,
    scaling: dict,
    polar: bool = True,
    nk: int = 20,
) -> np.ndarray:
    param_names = PARAM_NAMES_WITH_POLAR if polar else PARAM_NAMES_NO_POLAR
    mass_frac = _compute_mass_fraction(np.array(params["mass"], dtype=float))
    mu = _make_kernel_centers(nk)
    base_sigma = 1.0 / nk / np.sqrt(2)
    sigma_mass = 0.05
    kme_parts: list[np.ndarray] = [mass_frac]
    for pname in param_names:
        raw = np.array(params[pname], dtype=float)
        scaled = scaling[pname].scale(raw)
        sig = sigma_mass if pname == "mass" else base_sigma
        kme_val = kernel_mean(scaled, mu, gaussian_kernel, sigma=sig)
        kme_parts.append(kme_val)
    return np.concatenate(kme_parts)


def kme_desc_names(
    polar: bool = True,
    nk: int = 20,
) -> list[str]:
    names: list[str] = [f"mass_{e}" for e in ELEMENT_NAMES]
    param_names = PARAM_NAMES_WITH_POLAR if polar else PARAM_NAMES_NO_POLAR
    for pname in param_names:
        for k in range(nk):
            names.append(f"{pname}_{k}")
    return names
