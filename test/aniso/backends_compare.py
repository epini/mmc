#!/usr/bin/env python3
"""Anisotropic scattering: CUDA vs OpenCL vs CPU kernels, through pmmc (pmmc built with -DBUILD_CUDA=ON).

    python backends_compare.py --case B --nphoton 1e7 [--gpuid 1]

Same slab, observables and tests as gpu_vs_cpu.py (pyxopto_compare/common.py, full tensor with
eigenvalues (5, 10, 20) 1/mm, all off-diagonals), run on the three backends with the same photon
count and independent random streams; every pair is compared, z = diff / sqrt(err1^2 + err2^2),
pass if |z| < 4, T(optical path) and T(x, y) by chi^2, pass if p > 1e-3.
Also, on CUDA: elemmus = mus*I against vanilla MMC (statistical), and an anisotropic run that must
differ from the isotropic one (the tensor really reaches the CUDA kernel).
"""

import argparse
import itertools
import sys

import numpy as np
from scipy.stats import chi2 as chi2dist

from gpu_vs_cpu import CASES, binned, chi2, observables, run, slab_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="B", choices=["A", "B"])
    ap.add_argument("--nphoton", type=float, default=1e6)
    ap.add_argument("--gpuid", type=int, default=1)
    a = ap.parse_args()
    c = CASES[a.case]
    L, n, mua = c["L"], c["n"], c["mua"]

    cfg = slab_cfg(a.case, a.nphoton, 1648335518)
    backends = {"cuda": (cfg, "cuda", a.gpuid), "opencl": (dict(cfg, seed=314159265), "opencl", a.gpuid),
                "cpu": (dict(cfg, seed=27182818), "sse", -1)}
    res = {k: run(*v) for k, v in backends.items()}
    print("case %s, %d photons: " % (a.case, cfg["nphoton"])
          + ", ".join("%s %.1f s" % (k, r["seconds"]) for k, r in res.items()))
    obs = {k: observables(r, mua, n, L) for k, r in res.items()}

    allok = True
    keys = list(obs["cuda"][0])
    print("\n%-12s" % "" + "".join("%14s" % k for k in res)
          + "".join("%16s" % ("z %s-%s" % p) for p in itertools.combinations(res, 2)))
    for q in keys:
        line = "%-12s" % q + "".join("%14.6g" % obs[k][0][q][0] for k in res)
        for k1, k2 in itertools.combinations(res, 2):
            (m1, e1), (m2, e2) = obs[k1][0][q], obs[k2][0][q]
            z = (m1 - m2) / np.hypot(e1, e2)
            allok &= abs(z) < 4
            line += "%16.2f" % z
        print(line)

    print()
    for k1, k2 in itertools.combinations(res, 2):
        r1, r2 = res[k1], res[k2]
        s1, s2 = np.abs(r1["p"][:, 2] - L) < 1e-3, np.abs(r2["p"][:, 2] - L) < 1e-3
        for key in ("opl", "xy"):
            h1, h12 = binned(r1, obs[k1][1], s1, n, key)
            h2, h22 = binned(r2, obs[k2][1], s2, n, key)
            x2, nb = chi2(h1, h2, r1["N"], r2["N"], h12, h22)
            p = chi2dist.sf(x2 * nb, nb)
            allok &= p > 1e-3
            print("%-6s vs %-6s T(%-3s): reduced chi^2 = %.2f over %4d bins, p = %.3g" % (k1, k2, key, x2, nb, p))

    # isotropic regression on CUDA: elemmus = mus*I vs no elemmus, mus = 10 1/mm
    iso = dict(cfg, elemmus=np.tile(np.float32([10, 10, 10, 0, 0, 0]), (6, 1)))
    iso["prop"] = [[0, 0, 1, 1], [mua, 10.0, c["g"], n]]
    van = {k: v for k, v in iso.items() if k != "elemmus"}
    oi, _ = observables(run(iso, "cuda", a.gpuid), mua, n, L)
    ov, _ = observables(run(dict(van, seed=27182818), "cuda", a.gpuid), mua, n, L)
    print("\nCUDA isotropic tensor vs CUDA vanilla (mus = 10/mm), and the anisotropic run:")
    print("%-12s %14s %14s %8s %14s" % ("", "iso tensor", "vanilla", "z", "aniso"))
    for k in ("R", "T", "<opl>_T", "cov_xx_T"):
        z = (oi[k][0] - ov[k][0]) / np.hypot(oi[k][1], ov[k][1])
        allok &= abs(z) < 4
        print("%-12s %14.6g %14.6g %8.2f %14.6g" % (k, oi[k][0], ov[k][0], z, obs["cuda"][0][k][0]))
    za = (obs["cuda"][0]["T"][0] - oi["T"][0]) / np.hypot(obs["cuda"][0]["T"][1], oi["T"][1])
    allok &= abs(za) > 10
    print("aniso vs iso tensor on CUDA, T: z = %.1f (must be large)" % za)

    print("\nALL PASSED" if allok else "\nSOME CHECKS FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
