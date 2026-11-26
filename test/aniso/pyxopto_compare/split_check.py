#!/usr/bin/env python3
"""Consistency check of test 4: one tensor, slab cut into two element regions = whole slab.

    python split_check.py --case A --nphoton 1e7 [--split 3 7]

The slab of a single-layer case (A or B) is meshed as two stacked boxes with distinct element
labels (same mua, g, n, same tensor) and compared with the 6-element slab. The photons cross
the internal face, change label and look up the tensor of the new element: nothing may change.
Independent random streams (statistical test): R, T, and chi^2 of T(x, y) and R, T(optical path).
For information, also the run with the same seed as the whole slab (nearly identical photons
until floating-point differences of the two meshes make the histories diverge).
"""

import argparse
import sys

import numpy as np
from scipy.stats import chi2 as chi2dist

from run_pmmc import simulate


def compare(a, b, label):
    ok = True
    print("\n" + label)
    for face, name in (("top", "R"), ("bottom", "T")):
        ta, tb = a[face].sum(), b[face].sum()
        z = (ta - tb) / np.sqrt(a[face + "2"].sum() + b[face + "2"].sum())
        line = "  %s: %.6f vs %.6f, z = %+.2f" % (name, ta, tb, z)
        ok &= abs(z) < 4
        for what, axes in (("opl", (1, 2)), ("map", 0)):
            ha, hb = a[face].sum(axes), b[face].sum(axes)
            va, vb = a[face + "2"].sum(axes), b[face + "2"].sum(axes)
            if what == "map":
                ha, hb, va, vb = (x[1:-1, 1:-1] for x in (ha, hb, va, vb))   # drop the clipped edge bins
            sel = (va > 0) & (vb > 0) & (ha ** 2 > 100 * va) & (hb ** 2 > 100 * vb)   # >= ~100 photons per bin
            x2 = ((ha[sel] - hb[sel]) ** 2 / (va[sel] + vb[sel])).sum()
            nb = int(sel.sum())
            p = chi2dist.sf(x2, nb)
            ok &= p > 1e-3
            line += ";  %s chi2/%d = %.2f (p = %.2g)" % (what, nb, x2 / nb, p)
        print(line)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A", choices=["A", "B"])
    ap.add_argument("--nphoton", type=float, default=1e7)
    ap.add_argument("--split", type=float, nargs="+", default=[3.0, 7.0])
    ap.add_argument("--compute", default="opencl", choices=["opencl", "cuda", "sse"])
    ap.add_argument("--gpuid", type=int, default=1)
    a = ap.parse_args()
    gid = a.gpuid if a.compute != "sse" else -1

    whole = simulate(a.case, a.nphoton, a.compute, gid, seed=1648335518)
    split = simulate(a.case, a.nphoton, a.compute, gid, seed=27182818, split=a.split)
    same = simulate(a.case, a.nphoton, a.compute, gid, seed=1648335518, split=a.split)

    print("case %s, %d photons, split %s mm" % (a.case, int(a.nphoton), a.split))
    ok = compare(split, whole, "split (independent seed) vs whole:")
    compare(same, whole, "split (same seed) vs whole, for information (correlated, chi2 not meaningful):")
    print("\nPASSED" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
