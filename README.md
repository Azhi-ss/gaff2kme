# gaff2kme

SMILES → 190-dim KME descriptors via GAFF2 force field.

Converts molecular SMILES (including polymer pSMILES) into fixed-length 190-dimensional vectors using Kernel Mean Embedding of GAFF2 force field parameters.

## Overview

**GAFF2-KME** represents a molecule through the lens of its force field parameters. Instead of traditional molecular fingerprints (Morgan/ECFP), it captures the **physical chemistry** of a molecule — atom sizes, interaction strengths, bond stiffness, charge distribution — and compresses them into a smooth, fixed-length vector.

### 190-Dimensional Composition

| Category | Parameters | Dimensions |
|----------|-----------|------------|
| Mass fractions (H, C, N, O, F, P, S, Cl, Br, I) | Element composition | 10 |
| Gasteiger charge | Electrostatic distribution | 20 |
| ε_LJ (Lennard-Jones well depth) | van der Waals attraction | 20 |
| σ_LJ (LJ zero-crossing distance) | Atom effective size | 20 |
| K_bond (bond force constant) | Chemical bond strength | 20 |
| r₀ (equilibrium bond length) | Bond length | 20 |
| Bond polarity | Charge difference across bond | 20 |
| K_angle (angle force constant) | Bond angle stiffness | 20 |
| θ₀ (equilibrium bond angle) | Bond angle | 20 |
| K_dihedral (dihedral barrier) | Conformational rotation barrier | 20 |
| **Total** | | **190** |

Each continuous parameter is transformed via Gaussian kernel density estimation at 20 evenly-spaced grid points, producing a smooth distribution representation that is invariant to molecule size.

Set `polar=False` to exclude bond polarity → 170 dimensions.

## Installation

```bash
pip install -e .
```

Requires [RDKit](https://www.rdkit.org/). No other external dependencies.

### Optional: RadonPy backend

```bash
pip install -e /path/to/RadonPy
```

The `radonpy` backend delegates to RadonPy's native GAFF2_mod implementation. The default `rdkit_lite` backend works without RadonPy.

## Quick Start

### Python API

```python
from gaff2kme import compute, compute_batch, desc_names

# Single molecule
d = compute("c1ccccc1")  # benzene → np.ndarray(190,)

# Polymer pSMILES (cyclic n-mer)
d = compute("*CC*", cyclic=10)  # polyethylene, 10-mer ring

# Batch processing
df = compute_batch(["C", "CC", "c1ccccc1"])  # → pd.DataFrame

# Descriptor names
names = desc_names()  # ["mass_H", "mass_C", ..., "k_dih_19"]
```

### CLI

```bash
# Single SMILES
gaff2kme "c1ccccc1"

# From CSV
gaff2kme -i molecules.csv -o output.csv

# Polymer with custom repeat count
gaff2kme "*CC*" --cyclic 10

# Without polar descriptors (170-dim)
gaff2kme "CC" --no-polar

# Use RadonPy backend
gaff2kme "c1ccccc1" --backend radonpy
```

## pSMILES (Polymer SMILES)

Polymers are represented as pSMILES with `*` marking connection points, e.g. `*CC*` for polyethylene.

Processing flow:

```
*CC*  →  identify 2 wildcard atoms  →  delete wildcards → repeat core (CC)
     →  copy n cores  →  chain head-to-tail  →  close ring (cyclic n-mer)
     →  AddHs  →  GAFF2 parameterization  →  KME 190-dim
```

Default `cyclic=10` (10-mer ring), matching RadonPy convention. Ring closure eliminates end-group effects.

## Architecture

Three-layer design with clear separation of concerns:

```
Layer 1: math.py        Pure NumPy KME transform (zero dependencies)
    ↑
Layer 2: extraction.py  Parameter extraction from force-field-assigned mol
    ↑
Layer 3: pipeline.py    End-to-end: SMILES → 190-dim vector
```

### Backend System

| Backend | Dependency | Description |
|---------|-----------|-------------|
| `rdkit_lite` (default) | RDKit only | Pure RDKit + bundled GAFF2 JSON data |
| `radonpy` | RadonPy | Delegates to RadonPy's GAFF2_mod |

```python
# Switch backend
d = compute("CC", backend="rdkit_lite")  # default, no RadonPy needed
d = compute("CC", backend="radonpy")      # requires RadonPy installed
```

## Force Field Support

| Force Field | Backend | Atom Types |
|-------------|---------|------------|
| `gaff2_mod` (default) | Both | 90 pt, 847 bt |
| `gaff2` | Both | 83 pt, 840 bt |
| `gaff` | Both | 71 pt, 831 bt |

```python
d = compute("CC", ff_name="gaff2")
d = compute("CC", ff_name="gaff")
```

## Testing

```bash
pytest tests/ -v          # All tests (115)
pytest tests/test_math.py # Layer 1 only (46)
pytest tests/test_backends.py  # Backend + pSMILES tests (30)
```

## Reference

This implementation is based on the GAFF2-KME descriptor method from:

> Y. Hayashi, et al. "RadonPy: Automated physical property calculation using all-atom classical molecular dynamics simulations"

The KME descriptor converts GAFF2 force field parameter distributions into fixed-length vectors using Gaussian kernel mean embedding, enabling machine learning on molecular dynamics representations.
