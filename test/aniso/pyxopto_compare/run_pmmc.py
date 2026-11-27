#!/usr/bin/env python3
"""MMC (branch aniso) side of the comparison through pmmc, GPU by default, any case of common.py.

    python run_pmmc.py --case C --nphoton 1e8                 # compute="opencl", gpuid=1
    python run_pmmc.py --case A --nphoton 1e7 --compute sse   # CPU

Writes mmc_<case>.npz with the same layout as run_mmc.py (the command-line version) and
run_pyxopto.py: weight per launched photon in (optical path length, y, x) bins for the top (R)
and bottom (T) faces, clipped into the edge bins.

Mesh: one 6-tetrahedron box per layer of the case, stacked in z, laterally W wide; the face
between two boxes is triangulated the same way on both sides (conforming). Layer k has the
element label k+1 with the mua, g, n of that layer (Fresnel at faces where n changes).
--split d1 d2 ... cuts a single-layer case into boxes of these thicknesses (same tensor in
all of them), for the consistency check in split_check.py.
"""

import argparse
import os
import sys
import time

import numpy as np

from common import CASES, C_MM_NS, axes, stack, bin_clipped

import pmmc

HERE = os.path.dirname(os.path.abspath(__file__))
CUBE = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
TETS = [[1, 2, 8, 4], [1, 3, 4, 8], [1, 2, 6, 8], [1, 5, 8, 6], [1, 3, 8, 7], [1, 5, 7, 8]]


def stacked_mesh(thick, W):
    """Nodes (4 per z level) and elements of len(thick) boxes stacked from z = 0 down to sum(thick)."""
    z = np.concatenate([[0.0], np.cumsum(thick)])
    node = np.array([[x * W - W / 2, y * W - W / 2, zl] for zl in z for x, y, _ in CUBE[:4]])
    elem, label = [], []
    for k in range(len(thick)):
        for t in TETS:   # local cube node j -> level k + j // 4, corner j % 4
            elem.append([(k + (j - 1) // 4) * 4 + (j - 1) % 4 + 1 for j in t])
            label.append(k + 1)
    return node, np.array(elem), np.array(label)


def packed(M):
    return np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]], np.float32)


def simulate(case, nphoton, compute="opencl", gpuid=1, seed=1648335518, split=None, W=400.0,
             batch=5e6, verbose=False):
    """R and T histograms (weight per launched photon) for a case; split = list of box thicknesses."""
    XY_AXIS, PL_AXIS = axes(case)
    lay = stack(case)
    if split is not None:
        if len(lay) != 1 or abs(sum(split) - lay[0]["d"]) > 1e-9:
            raise ValueError("--split needs a single-layer case and thicknesses that add up to L")
        lay = [dict(lay[0], d=d) for d in split]
    thick = [l["d"] for l in lay]
    L = sum(thick)
    node, elem, label = stacked_mesh(thick, W)
    elemmus = np.array([packed(lay[k - 1]["M"]) for k in label])
    med = [[l["mua"], float(np.trace(l["M"]) / 3), l["g"], l["n"]] for l in lay]   # mus unused with elemmus
    mua = np.array([l["mua"] for l in lay])
    nidx = np.array([l["n"] for l in lay])

    tend = 1.5 * PL_AXIS[1] / C_MM_NS    # ns -> s below; beyond the last optical-path bin
    shape = (PL_AXIS[2], XY_AXIS[2], XY_AXIS[2])
    hist = {f: np.zeros(shape) for f in ("top", "bottom", "top2", "bottom2")}   # sums of w and w^2
    side, done, k, t0 = 0.0, 0, 0, time.time()
    while done < nphoton:
        n = int(min(batch, nphoton - done))
        cfg = {
            "nphoton": n, "seed": seed + 7919 * k, "node": node, "elem": elem, "elemprop": label,
            "prop": [[0, 0, 1, 1]] + med, "elemmus": elemmus,
            "srcpos": [0.0, 0.0, 0.0], "srcdir": [0, 0, 1], "e0": 1,
            "tstart": 0, "tend": tend * 1e-9, "tstep": tend * 1e-9,
            "detpos": [[0.0, 0.0, L / 2, 2 * W]], "issaveexit": 1, "maxdetphoton": n,
            "method": "elem", "compute": compute, "gpuid": gpuid,
            # no Russian roulette: detp keeps the launch weight w0, not the x roulettesize of the
            # survivors, so w0 exp(-mua ppath) would miss them below minenergy; tend stops the photons
            "minenergy": 0.0,
        }
        if not verbose:   # pmmc prints the MMC banner and log on stdout
            sys.stdout.flush()
            fd, keep = sys.stdout.fileno(), os.dup(sys.stdout.fileno())
            os.dup2(os.open(os.devnull, os.O_WRONLY), fd)
        try:
            d = pmmc.mmclab(cfg)["detp"]
        finally:
            if not verbose:
                os.dup2(keep, fd)
                os.close(keep)
        p = np.asarray(d["p"])
        ppath = np.asarray(d["ppath"]).reshape(len(p), -1)  # mm per layer
        w = np.asarray(d["w0"]) * np.exp(-ppath @ mua)      # detp stores the launch weight
        opl = ppath @ nidx                                  # optical path length, mm
        ix, iy, ipl = bin_clipped(p[:, 0], XY_AXIS), bin_clipped(p[:, 1], XY_AXIS), bin_clipped(opl, PL_AXIS)
        on = {}
        for face, zf in (("top", 0.0), ("bottom", L)):
            on[face] = np.abs(p[:, 2] - zf) < 1e-3
            idx = (ipl[on[face]], iy[on[face]], ix[on[face]])
            np.add.at(hist[face], idx, w[on[face]])
            np.add.at(hist[face + "2"], idx, w[on[face]] ** 2)
        side += w[~(on["top"] | on["bottom"])].sum()
        done += n
        k += 1
        print("  %d / %d photons, %.0f s" % (done, int(nphoton), time.time() - t0), flush=True)
    # top2/bottom2: variance of top/bottom per bin (sum of w^2 / N^2)
    return dict(top=hist["top"] / done, bottom=hist["bottom"] / done, top2=hist["top2"] / done ** 2,
                bottom2=hist["bottom2"] / done ** 2, side=side / done, nphoton=done,
                seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="C", choices=sorted(CASES))
    ap.add_argument("--nphoton", type=float, default=1e8)
    ap.add_argument("--compute", default="opencl", choices=["opencl", "cuda", "sse"])
    ap.add_argument("--gpuid", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1648335518)
    ap.add_argument("--split", type=float, nargs="+", default=None)
    ap.add_argument("--out", default=None, help="output file [mmc_<case>.npz]")
    a = ap.parse_args()

    r = simulate(a.case, a.nphoton, a.compute, a.gpuid if a.compute != "sse" else -1, a.seed, a.split)
    out = a.out or os.path.join(HERE, "mmc_%s.npz" % a.case)
    np.savez_compressed(out, case=a.case, **r)
    print("R=%.5f T=%.5f side=%.2e (%d photons, %.0f s) -> %s"
          % (r["top"].sum(), r["bottom"].sum(), r["side"], r["nphoton"], r["seconds"], out))


if __name__ == "__main__":
    main()
