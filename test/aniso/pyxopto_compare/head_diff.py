#!/usr/bin/env python3
"""Test 5: four-layer head, Fig. 5 of Pini et al., Neurophotonics 12, 045003 (2025).

    python head_diff.py [--plot head_diff.png]

Needs mmc_HX.npz, mmc_HY.npz (run_pmmc.py) and pyxopto_HX.npz, pyxopto_HY.npz (run_pyxopto.py):
white-matter fibres along x (HX) or y (HY). For each code, relative difference of the
time-gated reflectance of the two orientations (Eq. 5 of the paper, here as the ratio of the
gated integrals):

    dI(x, y) = 2 [I_y - I_x] / [I_y + I_x],   I = integral of R(x, y, t) over t1 < t < t2

on the 5 mm pixels of the quadrant x, y > 0 (the four quadrants are folded: the layered
geometry with fibres along x or y is symmetric under x -> -x and y -> -y). Then dI of MMC
vs dI of PyXOpto, pixel by pixel, with the variances of both codes.
"""

import argparse
import os

import numpy as np

from common import C_MM_NS, axes, edges

HERE = os.path.dirname(os.path.abspath(__file__))
GATES = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5))   # ns


def load(code, case):
    d = np.load(os.path.join(HERE, "%s_%s.npz" % (code, case)))
    return d["top"], d["top2"]


def fold(h):
    """Sum the four mirror pixels of h[..., y, x] into the quadrant x, y > 0."""
    n = h.shape[-1] // 2
    return h[..., n:, n:] + h[..., n:, n - 1::-1] + h[..., n - 1::-1, n:] + h[..., n - 1::-1, n - 1::-1]


def gated(h, v, plc, t1, t2):
    sel = (plc / C_MM_NS >= t1) & (plc / C_MM_NS < t2)
    return fold(h[sel].sum(0)), fold(v[sel].sum(0))


def rel_diff(a, va, b, vb):
    """2 (a - b) / (a + b) and its variance (a = I_y, b = I_x)."""
    s = a + b
    return 2 * (a - b) / s, 16 * (b ** 2 * va + a ** 2 * vb) / s ** 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()

    xy, pl = axes("HY")
    ple = edges(pl)
    plc = 0.5 * (ple[1:] + ple[:-1])
    xe = edges(xy)
    xq = xe[len(xe) // 2:]            # pixel edges of the quadrant, mm

    data = {code: {case: load(code, case) for case in ("HX", "HY")} for code in ("mmc", "pyxopto")}
    res = {}
    for code in data:
        res[code] = []
        for t1, t2 in GATES:
            iy, vy = gated(*data[code]["HY"], plc, t1, t2)
            ix, vx = gated(*data[code]["HX"], plc, t1, t2)
            res[code].append(rel_diff(iy, vy, ix, vx))

    ok_all = True
    print("relative difference dI = 2 (I_y - I_x) / (I_y + I_x), white-matter fibres along y vs x")
    for k, (t1, t2) in enumerate(GATES):
        dm, vm = res["mmc"][k]
        dp, vp = res["pyxopto"][k]
        good = (vm < 0.05 ** 2) & (vp < 0.05 ** 2)   # pixels where both codes know dI to 5 points
        z = (dm - dp)[good] / np.sqrt(vm + vp)[good]
        x2 = float(np.mean(z ** 2)) if good.any() else np.nan
        print("\ngate %.1f-%.1f ns: %d pixels with sigma(dI) < 0.05, reduced chi2 MMC vs PyXOpto = %.2f, max |z| = %.1f"
              % (t1, t2, good.sum(), x2, np.abs(z).max() if good.any() else np.nan))
        for code, (d, v) in (("MMC", (dm, vm)), ("PyXOpto", (dp, vp))):
            print("  %-8s dI along x axis (y < 5 mm), %%:  " % code
                  + " ".join("%+6.1f" % (100 * d[0, i]) if v[0, i] < 0.05 ** 2 else "   -  " for i in range(d.shape[1])))
            print("  %-8s dI along y axis (x < 5 mm), %%:  " % code
                  + " ".join("%+6.1f" % (100 * d[i, 0]) if v[i, 0] < 0.05 ** 2 else "   -  " for i in range(d.shape[0])))
        ok_all &= bool(good.any()) and x2 < 2
    print("\n(columns: pixel centres %s mm)" % ", ".join("%g" % c for c in 0.5 * (xq[1:] + xq[:-1])))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, len(GATES), figsize=(4 * len(GATES), 7.5), constrained_layout=True)
        lim = max(np.nanmax(np.abs(np.where(res[c][k][1] < 0.05 ** 2, res[c][k][0], 0)))
                  for c in res for k in range(len(GATES)))
        for row, (code, who) in enumerate((("mmc", "MMC aniso"), ("pyxopto", "PyXOpto"))):
            for k, (t1, t2) in enumerate(GATES):
                d, v = res[code][k]
                img = np.where(v < 0.05 ** 2, 100 * d, np.nan)
                ax[row, k].set_facecolor("0.85")      # blank pixels (sigma > 5 points), not zero
                im = ax[row, k].pcolormesh(xq, xq, img, cmap="RdBu_r", vmin=-100 * lim, vmax=100 * lim)
                ax[row, k].set_aspect("equal")
                ax[row, k].set_title("%s, %.1f–%.1f ns" % (who, t1, t2), fontsize=9)
                ax[row, k].set_xlabel("x (mm)"); ax[row, k].set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, label=r"$\Delta I$ = 2(I$_y$ − I$_x$)/(I$_y$ + I$_x$)  (%)", shrink=0.8)
        fig.suptitle("Four-layer head (Pini et al. 2025, Fig. 5c): white-matter fibres along y vs x "
                     "(grey: σ > 5 points)")
        fig.savefig(a.plot, dpi=120)
    print("\nPASSED" if ok_all else "\nCHECK")


if __name__ == "__main__":
    main()
