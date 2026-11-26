#!/usr/bin/env python3
"""Anisotropic scattering: GPU (OpenCL/CUDA) kernel vs CPU, through pmmc.

    python gpu_vs_cpu.py --case A --nphoton 1e6 [--compute opencl] [--gpuid 1]

Slab of the PyXOpto comparison (pyxopto_compare/common.py): L = 10 mm, n = 1.4, HG g = 0.8,
full tensor with eigenvalues (5, 10, 20) 1/mm rotated so that Mxy, Mxz, Myz != 0.
Same number of photons on the two backends (independent random streams), detected photons
on the top (R) and bottom (T) faces. Each observable is compared with its statistical error,
z = (GPU - CPU) / sqrt(err_GPU^2 + err_CPU^2), pass if |z| < 4; the binned T(optical path) and
T(x, y) by chi^2, pass if p > 1e-3.
Also: on the GPU, elemmus = mus*I must reproduce vanilla MMC (isotropic regression, statistical:
the GPU is not bitwise reproducible because of the atomic float sums).
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import chi2 as chi2dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pyxopto_compare"))
from common import CASES, mus_tensor  # noqa: E402

import pmmc  # noqa: E402


def slab_cfg(case, nphoton, seed, W=400.0):
    c = CASES[case]
    L = c["L"]
    cube = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    node = np.array([[x * W - W / 2, y * W - W / 2, z * L] for x, y, z in cube])
    elem = np.array([[1, 2, 8, 4], [1, 3, 4, 8], [1, 2, 6, 8], [1, 5, 8, 6], [1, 3, 8, 7], [1, 5, 7, 8]])
    M = mus_tensor(case)
    tend = 5e-8
    return {
        "nphoton": int(nphoton), "seed": seed, "node": node, "elem": elem, "elemprop": np.ones(6, int),
        "prop": [[0, 0, 1, 1], [c["mua"], float(np.trace(M) / 3), c["g"], c["n"]]],
        "srcpos": [0.0, 0.0, 0.0], "srcdir": [0, 0, 1], "e0": 1,
        "tstart": 0, "tend": tend, "tstep": tend,
        "detpos": [[0.0, 0.0, L / 2, 2 * W]], "issaveexit": 1, "maxdetphoton": int(nphoton),
        "method": "elem", "elemmus": np.tile(np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]],
                                                       np.float32), (6, 1)),
    }


def run(cfg, compute, gpuid):
    cfg = dict(cfg, compute=compute, gpuid=gpuid)
    t0 = time.time()
    res = pmmc.mmclab(cfg)
    dt = time.time() - t0
    d = res["detp"]                   # ppath in mm, w0 = launch weight
    return dict(nscat=np.asarray(d["nscat"])[:, 0], ppath=np.asarray(d["ppath"])[:, 0], p=np.asarray(d["p"]),
                w0=np.asarray(d["w0"]), N=cfg["nphoton"], seconds=dt)


def observables(r, mua, n, L):
    """Per-photon weighted quantities and their standard errors (sums over launched photons)."""
    w = r["w0"] * np.exp(-mua * r["ppath"])
    out = {}
    for face, z in (("R", 0.0), ("T", L)):
        s = np.abs(r["p"][:, 2] - z) < 1e-3
        ws, N = w[s], r["N"]
        out[face] = (ws.sum() / N, np.sqrt((ws ** 2).sum()) / N)
        if face == "T":
            W = ws.sum()
            for name, q in (("<x>_T", r["p"][s, 0]), ("<y>_T", r["p"][s, 1]),
                            ("<opl>_T", r["ppath"][s] * n), ("<nscat>_T", r["nscat"][s])):
                m = (ws * q).sum() / W
                out[name] = (m, np.sqrt((ws ** 2 * (q - m) ** 2).sum()) / W)
            x, y = r["p"][s, 0] - out["<x>_T"][0], r["p"][s, 1] - out["<y>_T"][0]
            for name, q in (("cov_xx_T", x * x), ("cov_yy_T", y * y), ("cov_xy_T", x * y)):
                m = (ws * q).sum() / W
                out[name] = (m, np.sqrt((ws ** 2 * (q - m) ** 2).sum()) / W)
    return out, w


def chi2(a, b, Na, Nb, wa2, wb2, minw=10):
    """Reduced chi^2 of two weight histograms (per launched photon), bins with enough counts."""
    va, vb = wa2 / Na ** 2, wb2 / Nb ** 2
    ok = (va > 0) & (vb > 0) & (a * Na > minw) & (b * Nb > minw)
    return ((a[ok] - b[ok]) ** 2 / (va[ok] + vb[ok])).sum() / max(ok.sum(), 1), int(ok.sum())


def binned(r, w, sel, n, key):
    if key == "opl":
        idx = np.clip(np.floor(r["ppath"][sel] * n / 20.0).astype(int), 0, 149)   # 20 mm bins up to 3 m
        shape = (150,)
    else:
        ix = np.clip(np.floor((r["p"][sel, 0] + 50) / 2).astype(int), 0, 49)     # 2 mm bins, +-50 mm
        iy = np.clip(np.floor((r["p"][sel, 1] + 50) / 2).astype(int), 0, 49)
        idx, shape = iy * 50 + ix, (2500,)
    h, h2 = np.zeros(shape), np.zeros(shape)
    np.add.at(h, idx, w[sel])
    np.add.at(h2, idx, w[sel] ** 2)
    return h / r["N"], h2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A", choices=sorted(CASES))
    ap.add_argument("--nphoton", type=float, default=1e6)
    ap.add_argument("--compute", default="opencl", choices=["opencl", "cuda"])
    ap.add_argument("--gpuid", type=int, default=1)
    a = ap.parse_args()
    c = CASES[a.case]
    L, n, mua = c["L"], c["n"], c["mua"]

    cfg = slab_cfg(a.case, a.nphoton, 1648335518)
    gpu = run(cfg, a.compute, a.gpuid)
    cpu = run(dict(cfg, seed=27182818), "sse", -1)
    print("case %s, %d photons: CPU %.1f s, GPU (%s) %.1f s, speed-up %.0fx\n"
          % (a.case, cfg["nphoton"], cpu["seconds"], a.compute, gpu["seconds"], cpu["seconds"] / gpu["seconds"]))

    og, wg = observables(gpu, mua, n, L)
    oc, wc = observables(cpu, mua, n, L)
    allok = True
    print("%-12s %14s %14s %8s" % ("", "GPU", "CPU", "z"))
    for k in og:
        z = (og[k][0] - oc[k][0]) / np.hypot(og[k][1], oc[k][1])
        allok &= abs(z) < 4
        print("%-12s %14.6g %14.6g %8.2f" % (k, og[k][0], oc[k][0], z))

    sg, sc = np.abs(gpu["p"][:, 2] - L) < 1e-3, np.abs(cpu["p"][:, 2] - L) < 1e-3
    for key in ("opl", "xy"):
        hg, hg2 = binned(gpu, wg, sg, n, key)
        hc, hc2 = binned(cpu, wc, sc, n, key)
        x2, nb = chi2(hg, hc, gpu["N"], cpu["N"], hg2, hc2)
        p = chi2dist.sf(x2 * nb, nb)
        allok &= p > 1e-3
        print("T(%s): reduced chi^2 = %.2f over %d bins, p = %.3g" % (key, x2, nb, p))


    # isotropic regression on the GPU: elemmus = mus*I vs no elemmus, mus = 10 1/mm
    iso = dict(cfg, elemmus=np.tile(np.float32([10, 10, 10, 0, 0, 0]), (6, 1)))
    iso["prop"] = [[0, 0, 1, 1], [mua, 10.0, c["g"], n]]
    van = {k: v for k, v in iso.items() if k != "elemmus"}
    oi, _ = observables(run(iso, a.compute, a.gpuid), mua, n, L)
    ov, _ = observables(run(dict(van, seed=27182818), a.compute, a.gpuid), mua, n, L)
    print("\nGPU isotropic tensor vs GPU vanilla (mus = 10/mm):")
    for k in ("R", "T", "<opl>_T", "cov_xx_T"):
        z = (oi[k][0] - ov[k][0]) / np.hypot(oi[k][1], ov[k][1])
        allok &= abs(z) < 4
        print("%-12s %14.6g %14.6g %8.2f" % (k, oi[k][0], ov[k][0], z))

    print("\nALL PASSED" if allok else "\nSOME CHECKS FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
