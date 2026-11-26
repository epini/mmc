#!/usr/bin/env python3
"""MMC (branch aniso) side of the comparison, CPU.

    python3 run_mmc.py --case A --nphoton 1e7

Writes mmc_<case>.npz with the same layout as run_pyxopto.py: weight per launched photon
in (optical path length, y, x) bins for the top (R) and bottom (T) faces, clipped into the
edge bins. The slab is a 6-tetrahedron box, laterally much wider than the detector grid.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from common import CASES, XY_AXIS, PL_AXIS, mus_tensor, bin_clipped

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from slab_vs_ade import load_mch  # noqa: E402

MMC = os.path.join(HERE, "..", "..", "..", "bin", "mmc")
C_MM_NS = 299.792458


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A", choices=["A", "B"])   # single layer; any case: run_pmmc.py
    ap.add_argument("--nphoton", type=float, default=1e7)
    ap.add_argument("--W", type=float, default=400.0, help="lateral size of the box, mm")
    ap.add_argument("--seed", type=int, default=1648335518)
    a = ap.parse_args()

    c = CASES[a.case]
    L, W, tag = c["L"], a.W, "mmc_%s" % a.case
    M = mus_tensor(a.case)

    sx, sz = W / 60.0, L / 60.0
    cube = [[0, 0, 0], [60, 0, 0], [0, 60, 0], [60, 60, 0], [0, 0, 60], [60, 0, 60], [0, 60, 60], [60, 60, 60]]
    nodes = [[x * sx - W / 2, y * sx - W / 2, z * sz] for x, y, z in cube]   # source at the origin
    elems = [[1, 2, 8, 4, 1], [1, 3, 4, 8, 1], [1, 2, 6, 8, 1], [1, 5, 8, 6, 1], [1, 3, 8, 7, 1], [1, 5, 7, 8, 1]]
    packed = np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]], np.float32)
    np.tile(packed, (6, 1)).astype(np.float32).tofile(os.path.join(HERE, tag + "_elemmus.bin"))

    tend = 1.5 * PL_AXIS[1] / C_MM_NS            # beyond the last optical-path bin
    cfg = {
        "Session": {"Photons": int(a.nphoton), "RNGSeed": a.seed, "ID": tag},
        "Domain": {"Media": [{"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0},
                             {"mua": c["mua"], "mus": float(np.trace(M) / 3), "g": c["g"], "n": c["n"]}]},
        "Mesh": {"MeshNode": nodes, "MeshElem": elems, "InitElem": 2, "ElemMusFile": tag + "_elemmus.bin"},
        "Forward": {"T0": 0.0, "T1": tend, "Dt": tend},
        "Optode": {"Source": {"Pos": [0.0, 0.0, 0.0], "Dir": [0, 0, 1]},
                   "Detector": [{"Pos": [0.0, 0.0, L / 2], "R": 2 * W}]},
    }
    with open(os.path.join(HERE, tag + ".json"), "w") as f:
        json.dump(cfg, f, indent=1)

    t0 = time.time()
    out = subprocess.run([MMC, "-f", tag + ".json", "-G", "-1", "-M", "S", "-S", "0", "-d", "1", "-x", "1",
                          "-H", str(int(a.nphoton))], cwd=HERE, capture_output=True, text=True)
    if out.returncode != 0 or "ERROR" in out.stdout + out.stderr:
        print(out.stdout[-3000:], out.stderr[-3000:])
        sys.exit(1)

    cols, hdr = load_mch(os.path.join(HERE, tag + ".mch"))
    N = hdr["total"]
    p = cols["p"]
    # the .mch stores the launch weight; absorption is applied from the partial path, as in mcx/mmc
    w = cols["w"][:, 0] * np.exp(-c["mua"] * cols["ppath"][:, 0])
    opl = cols["ppath"][:, 0] * c["n"]           # optical path length, mm
    ix, iy, ipl = bin_clipped(p[:, 0], XY_AXIS), bin_clipped(p[:, 1], XY_AXIS), bin_clipped(opl, PL_AXIS)

    res = {}
    for face, z in (("top", 0.0), ("bottom", L)):
        s = np.abs(p[:, 2] - z) < 1e-3
        h = np.zeros((PL_AXIS[2], XY_AXIS[2], XY_AXIS[2]))
        np.add.at(h, (ipl[s], iy[s], ix[s]), w[s])
        res[face] = h / N
    side = ~((np.abs(p[:, 2]) < 1e-3) | (np.abs(p[:, 2] - L) < 1e-3))

    np.savez_compressed(os.path.join(HERE, tag + ".npz"), top=res["top"], bottom=res["bottom"],
                        side=w[side].sum() / N, nphoton=N, seconds=time.time() - t0, case=a.case)
    print("R=%.5f T=%.5f side=%.2e (%d photons, %.0f s)" % (res["top"].sum(), res["bottom"].sum(),
                                                             w[side].sum() / N, N, time.time() - t0))
    for f in (tag + ".mch",):
        os.remove(os.path.join(HERE, f))


if __name__ == "__main__":
    main()
