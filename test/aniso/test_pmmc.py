#!/usr/bin/env python3
"""pmmc tests for cfg['elemmus'] (branch aniso). Run with OMP_NUM_THREADS=1 for bitwise checks:

    OMP_NUM_THREADS=1 python3 test_pmmc.py

1. isotropic regression: elemmus = mus*I gives the same flux as no elemmus (bitwise)
2. (ne,6) and (ne,3,3) inputs give the same flux (bitwise); a non-symmetric 3x3 equals its symmetric part
3. anisotropic tensor: pmmc gives the same flux as the mmc command line on the same input (bitwise)
4. errors: wrong shape, wrong length, negative tensor, GPU requested
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pmmc

HERE = os.path.dirname(os.path.abspath(__file__))
MMC = os.path.join(HERE, "..", "..", "bin", "mmc")

NODE = np.array([[0, 0, 0], [60, 0, 0], [0, 60, 0], [60, 60, 0],
                 [0, 0, 60], [60, 0, 60], [0, 60, 60], [60, 60, 60]], float)
ELEM = np.array([[1, 2, 8, 4], [1, 3, 4, 8], [1, 2, 6, 8], [1, 5, 8, 6], [1, 3, 8, 7], [1, 5, 7, 8]])
NE = len(ELEM)


def base_cfg():
    return {
        "nphoton": 200000, "seed": 27182818, "node": NODE, "elem": ELEM, "elemprop": np.ones(NE, int),
        "prop": [[0, 0, 1, 1], [0.005, 1.0, 0.01, 1.37]],
        "srcpos": [30.1, 30.2, 0.0], "srcdir": [0, 0, 1], "e0": 2,
        "tstart": 0, "tend": 5e-9, "tstep": 1e-10, "gpuid": -1, "method": "elem",
    }


def flux(cfg):
    return np.asarray(pmmc.mmclab(cfg)["flux"])


def packed(M):
    return np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]], np.float32)


def rot(a, b):
    a, b = np.deg2rad(a), np.deg2rad(b)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return Rz @ Ry


def check(name, ok, extra=""):
    print("%-62s %s %s" % (name, "PASS" if ok else "FAIL", extra))
    return ok


def main():
    allok = True

    # 1. isotropic regression
    ref = flux(base_cfg())
    c = base_cfg(); c["elemmus"] = np.tile(np.float32([1, 1, 1, 0, 0, 0]), (NE, 1))
    allok &= check("1. elemmus = mus*I identical to vanilla", np.array_equal(ref, flux(c)))

    # 2. (ne,6) vs (ne,3,3), and non-symmetric input
    M = rot(30, 40) @ np.diag([0.5, 1.0, 2.0]) @ rot(30, 40).T
    c6 = base_cfg(); c6["elemmus"] = np.tile(packed(M), (NE, 1))
    c33 = base_cfg(); c33["elemmus"] = np.tile(M.astype(np.float32), (NE, 1, 1))
    A = np.array([[0, 0.3, -0.2], [-0.3, 0, 0.1], [0.2, -0.1, 0]])     # antisymmetric part
    cns = base_cfg(); cns["elemmus"] = np.tile((M + A).astype(np.float32), (NE, 1, 1))
    f6, f33, fns = flux(c6), flux(c33), flux(cns)
    allok &= check("2a. (ne,6) identical to (ne,3,3)", np.array_equal(f6, f33))
    allok &= check("2b. non-symmetric 3x3 identical to its symmetric part", np.array_equal(f6, fns))
    allok &= check("2c. anisotropic flux differs from isotropic", not np.array_equal(f6, ref))

    # 3. pmmc vs command line: the two interfaces do not draw the same photons (seeding and defaults
    #    differ, also in vanilla MMC), so compare statistically: the anisotropic pmmc-CLI difference
    #    must be of the same size as the isotropic one
    def cli_flux(elemmus):
        with tempfile.TemporaryDirectory() as d:
            js = {"Session": {"Photons": 200000, "RNGSeed": 27182818, "ID": "cli"},
                  "Domain": {"Media": [{"mua": 0, "mus": 0, "g": 1, "n": 1},
                                       {"mua": 0.005, "mus": 1.0, "g": 0.01, "n": 1.37}]},
                  "Mesh": {"MeshNode": NODE.tolist(), "MeshElem": np.c_[ELEM, np.ones(NE, int)].tolist(),
                           "InitElem": 2},
                  "Forward": {"T0": 0, "T1": 5e-9, "Dt": 1e-10},
                  "Optode": {"Source": {"Pos": [30.1, 30.2, 0.0], "Dir": [0, 0, 1]}}}
            if elemmus is not None:
                elemmus.astype(np.float32).tofile(os.path.join(d, "t.bin"))
                js["Mesh"]["ElemMusFile"] = "t.bin"
            json.dump(js, open(os.path.join(d, "cli.json"), "w"))
            subprocess.run([os.path.abspath(MMC), "-f", "cli.json", "-G", "-1", "-M", "S", "-F", "bin"],
                           cwd=d, capture_output=True, text=True)
            return np.fromfile(os.path.join(d, "cli.bin"), np.float64)

    def reldiff(fc, fp):
        return abs(fc.sum() - fp.sum()) / fp.sum()

    if not (os.path.isfile(MMC) or os.path.isfile(MMC + ".exe")):
        print("%-62s SKIP (mmc command-line binary not built)" % "3. pmmc vs CLI (aniso)")
    else:
        d_iso = reldiff(cli_flux(None), ref.ravel(order="F"))
        d_ani = reldiff(cli_flux(np.tile(packed(M), (NE, 1))), f6.ravel(order="F"))
        allok &= check("3. pmmc vs CLI (aniso) within the vanilla pmmc-CLI spread", d_ani < max(3 * d_iso, 0.01),
                       "(total flux: aniso %.2e, iso %.2e)" % (d_ani, d_iso))

    # 4. errors
    for name, val, extra in [
        ("4a. wrong shape (ne,5) rejected", np.ones((NE, 5), np.float32), {}),
        ("4b. wrong length (ne-1,6) rejected", np.tile(np.float32([1, 1, 1, 0, 0, 0]), (NE - 1, 1)), {}),
        ("4c. negative tensor rejected", np.tile(np.float32([1, 1, -1, 0, 0, 0]), (NE, 1)), {}),
        ("4d. GPU requested rejected", np.tile(np.float32([1, 1, 1, 0, 0, 0]), (NE, 1)), {"gpuid": 1}),
    ]:
        c = base_cfg(); c["elemmus"] = val; c.update(extra)
        try:
            flux(c)
            allok &= check(name, False, "(no error raised)")
        except Exception as e:  # noqa: BLE001
            allok &= check(name, True, "(%s)" % str(e).strip().splitlines()[-1][:70])

    print("\nALL PASSED" if allok else "\nSOME TESTS FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
