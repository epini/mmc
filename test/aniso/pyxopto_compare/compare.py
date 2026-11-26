#!/usr/bin/env python3
"""Compare mmc_<case>.npz and pyxopto_<case>.npz.

    python3 compare.py --case A [--plot compare_A.png]

MMC launches the photon inside the medium (no specular loss at the source), PyXOpto
removes the specular reflection and reports it separately: MMC is scaled by (1 - Rsp).
Uncertainties assume counting statistics with weights <= 1 (exact for mua = 0); with
absorption and an mmc_<case>.npz from run_pmmc.py, also a chi^2 with the exact variance.
"""

import argparse
import os

import numpy as np

from common import CASES, XY_AXIS, PL_AXIS, edges, layers

HERE = os.path.dirname(os.path.abspath(__file__))


def moments(img, xc):
    """Centroid and covariance of a 2D map img[y, x] on bin centres xc (both axes)."""
    X, Y = np.meshgrid(xc, xc)
    s = img.sum()
    mx, my = (img * X).sum() / s, (img * Y).sum() / s
    cxx = (img * (X - mx) ** 2).sum() / s
    cyy = (img * (Y - my) ** 2).sum() / s
    cxy = (img * (X - mx) * (Y - my)).sum() / s
    return np.array([mx, my, cxx, cyy, cxy])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A", choices=sorted(CASES))
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()

    m = np.load(os.path.join(HERE, "mmc_%s.npz" % a.case))
    p = np.load(os.path.join(HERE, "pyxopto_%s.npz" % a.case))
    Nm, Np = float(m["nphoton"]), float(p["nphoton"])
    rsp = float(p["specular"])
    scale = 1.0 - rsp

    xe = edges(XY_AXIS)
    xc = 0.5 * (xe[1:] + xe[:-1])
    ple = edges(PL_AXIS)
    plc = 0.5 * (ple[1:] + ple[:-1])
    inner = (slice(1, -1), slice(1, -1))   # drop the clipped edge bins of the maps

    print("case %s: %s" % (a.case, CASES[a.case]))
    for d, M in layers(a.case):
        print("layer %g mm, tensor (1/mm):\n%s" % (d, np.array2string(M, precision=3)))
    print("photons: MMC %.1e, PyXOpto %.1e; specular %.5f\n" % (Nm, Np, rsp))

    results = {}
    for face in ("top", "bottom"):
        hm, hp = m[face] * scale, p[face]
        tot_m, tot_p = hm.sum(), hp.sum()
        sig = np.sqrt(tot_m * (1 - tot_m) / Nm + tot_p * (1 - tot_p) / Np)
        name = "R" if face == "top" else "T"
        print("%s total: MMC %.5f  PyXOpto %.5f  diff %+.5f (%.1f sigma)"
              % (name, tot_m, tot_p, tot_m - tot_p, (tot_m - tot_p) / sig))

        im, ip = hm.sum(0)[inner], hp.sum(0)[inner]
        mm_, mp_ = moments(im, xc[1:-1]), moments(ip, xc[1:-1])
        lbl = ["<x>", "<y>", "var x", "var y", "cov xy"]
        print("  map moments    " + "  ".join("%9s" % l for l in lbl))
        print("  MMC            " + "  ".join("%9.3f" % v for v in mm_))
        print("  PyXOpto        " + "  ".join("%9.3f" % v for v in mp_))

        # per-bin z-scores on the steady-state map, bins with >= 100 expected counts
        var = im / Nm + ip / Np
        ok = (ip * Np >= 100) & (im * Nm >= 100)
        z = (im[ok] - ip[ok]) / np.sqrt(var[ok])
        print("  map: %d bins, reduced chi2 = %.3f, max |z| = %.1f" % (ok.sum(), np.mean(z ** 2), np.abs(z).max()))

        pm, pp = hm.sum((1, 2)), hp.sum((1, 2))
        okp = (pp * Np >= 100) & (pm * Nm >= 100)
        zp = (pm[okp] - pp[okp]) / np.sqrt(pm[okp] / Nm + pp[okp] / Np)
        print("  optical path: %d bins, reduced chi2 = %.3f" % (okp.sum(), np.mean(zp ** 2)))

        # with absorption: exact MMC variance (sum of w^2, saved by run_pmmc.py), taken for both codes,
        # in the bins where the weight exp(-mua*opl/n) stays above PyXOpto's roulette threshold 1e-4:
        # below it the lottery (survival 1%, weight x100) makes the PyXOpto variance ~100 times larger
        c = CASES[a.case]
        if c["mua"] > 0 and face + "2" in m:
            vm = m[face + "2"].sum((1, 2)) * scale ** 2
            okw = (vm > 0) & (pm ** 2 > 100 * vm) & (np.exp(-c["mua"] * plc / c["n"]) > 1e-4)
            x2 = ((pm[okw] - pp[okw]) ** 2 / (2 * vm[okw])).sum()
            print("  optical path, exact variance, weight > 1e-4 (opl < %.0f mm): %d bins, reduced chi2 = %.3f"
                  % (plc[okw].max(), okw.sum(), x2 / okw.sum()))
        print()
        results[face] = (im, ip, pm, pp)

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(14, 8.5))
        ext = [xc[1], xc[-2], xc[1], xc[-2]]
        for row, face in enumerate(("top", "bottom")):
            im, ip, pm, pp = results[face]
            name = "R" if face == "top" else "T"
            vmax = np.log10(max(im.max(), ip.max()))
            for col, (img, who) in enumerate(((im, "MMC aniso"), (ip, "PyXOpto"))):
                a_ = ax[row, col]
                a_.imshow(np.log10(np.maximum(img, 1e-12)), origin="lower", extent=ext,
                          vmin=vmax - 5, vmax=vmax, cmap="viridis")
                lv = vmax - np.arange(0.5, 5, 1.0)
                a_.contour(xc[1:-1], xc[1:-1], np.log10(np.maximum(im, 1e-12)), lv[::-1], colors="r", linewidths=0.7)
                a_.contour(xc[1:-1], xc[1:-1], np.log10(np.maximum(ip, 1e-12)), lv[::-1], colors="w",
                           linewidths=0.7, linestyles="--")
                a_.set_title("%s(x,y) %s  (contours: red MMC, white dashed PyXOpto)" % (name, who), fontsize=9)
                a_.set_xlabel("x (mm)"); a_.set_ylabel("y (mm)")
                a_.set_xlim(-25, 25); a_.set_ylim(-25, 25)
            a_ = ax[row, 2]
            a_.semilogy(plc, pm, ".", ms=3, label="MMC aniso")
            a_.semilogy(plc, pp, "-", lw=1, label="PyXOpto")
            a_.set_xlabel("optical path length (mm)"); a_.set_ylabel("%s per bin" % name)
            a_.set_ylim(max(pp.max() * 1e-5, 1e-9), pp.max() * 2); a_.legend()
        c = CASES[a.case]
        if "eig" in c:
            what = "full scattering tensor, eig %s mm$^{-1}$" % (c["eig"],)
        else:
            what = ("two layers, fibres along x (top) and y (bottom), $\\mu_{s\\parallel}$ = %g, "
                    "$\\mu_{s\\perp}$ = %g mm$^{-1}$" % (c["mus_par"], c["mus_perp"]))
        fig.suptitle("Case %s — %s, g = %g, $\\mu_a$ = %g mm$^{-1}$" % (a.case, what, c["g"], c["mua"]))
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120)


if __name__ == "__main__":
    main()
