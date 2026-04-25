"""CLI entry point for gaff2kme."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gaff2kme",
        description="Convert SMILES to 190-dim KME descriptors using GAFF2_mod force field",
    )
    parser.add_argument("smiles", nargs="?", help="Single SMILES string")
    parser.add_argument("-i", "--input", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output CSV file path")
    parser.add_argument("--smiles-col", default="SMILES", help="SMILES column name in CSV (default: SMILES)")
    parser.add_argument("--ff", default="gaff2_mod", choices=["gaff2_mod", "gaff2", "gaff"], help="Force field (default: gaff2_mod)")
    parser.add_argument("--backend", default="rdkit_lite", choices=["rdkit_lite", "radonpy"], help="Backend (default: rdkit_lite)")
    parser.add_argument("--cyclic", type=int, default=10, help="Cyclic polymer degree (default: 10)")
    parser.add_argument("--mp", type=int, default=None, help="Number of parallel processes")
    parser.add_argument("--no-polar", dest="polar", action="store_false", help="Disable polar descriptors (170-dim)")
    parser.add_argument("--nk", type=int, default=20, help="Number of kernel centers (default: 20)")

    args = parser.parse_args()

    if args.smiles and args.input:
        parser.error("Provide either a SMILES string or --input, not both")
    if not args.smiles and not args.input:
        parser.error("Provide a SMILES string or --input CSV file")

    from .pipeline import compute, compute_from_csv

    if args.smiles:
        d = compute(args.smiles, ff_name=args.ff, polar=args.polar,
                    cyclic=args.cyclic, nk=args.nk, backend=args.backend)
        print(" ".join(f"{v:.8f}" if not np.isnan(v) else "nan" for v in d))
    else:
        output = args.output or "gaff2kme_output.csv"
        df = compute_from_csv(
            args.input,
            smiles_col=args.smiles_col,
            ff_name=args.ff,
            polar=args.polar,
            cyclic=args.cyclic,
            nk=args.nk,
            mp=args.mp,
            output=output,
            backend=args.backend,
        )
        print(f"Output written to {output} ({df.shape[0]} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
