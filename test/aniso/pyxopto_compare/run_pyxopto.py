#!/usr/bin/env python3
"""PyXOpto side of the comparison: run on a machine with PyXOpto + an OpenCL GPU.

    python run_pyxopto.py --case A --nphoton 1e8
    python run_pyxopto.py --case B --nphoton 1e8

Writes pyxopto_<case>.npz next to this script. Uses mcml AnisotropicLayer with the
full 3x3 scattering tensor from common.py and a scalar Henyey-Greenstein phase function.
"""

import argparse
import os
import time

import numpy as np
from xopto.mcml import mc

from common import CASES, XY_AXIS, PL_AXIS, layers

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A", choices=sorted(CASES))
    ap.add_argument("--nphoton", type=float, default=1e8)
    ap.add_argument("--batch", type=float, default=1e7, help="photons per run() call")
    ap.add_argument("--device", type=int, default=0, help="index in mc.clinfo.gpus()")
    a = ap.parse_args()

    c = CASES[a.case]
    mm = 1e-3
    pf = mc.mcpf.Hg(c["g"])

    # one AnisotropicLayer per slab layer (cases C, D: two), tensors in 1/m, between two d = 0 outer layers
    slab = [mc.mclayer.AnisotropicLayer(d=d * mm, n=c["n"], mua=c["mua"] / mm, mus=M / mm, pf=pf)
            for d, M in layers(a.case)]
    outer = [mc.mclayer.AnisotropicLayer(d=0.0, n=1.0, mua=0.0, mus=0.0, pf=pf) for _ in range(2)]
    stack = mc.mclayer.Layers([outer[0]] + slab + [outer[1]])
    source = mc.mcsource.Line((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    def det():
        return mc.mcdetector.CartesianPl(
            mc.mcdetector.Axis(XY_AXIS[0] * mm, XY_AXIS[1] * mm, XY_AXIS[2]),
            mc.mcdetector.Axis(XY_AXIS[0] * mm, XY_AXIS[1] * mm, XY_AXIS[2]),
            plaxis=mc.mcdetector.Axis(PL_AXIS[0] * mm, PL_AXIS[1] * mm, PL_AXIS[2]))

    detectors = mc.mcdetector.Detectors(top=det(), bottom=det(), specular=mc.mcdetector.Total())
    devices = mc.clinfo.gpus() or mc.clinfo.cpus()   # CPU fallback (e.g. pocl) for testing
    print("OpenCL devices:", devices)
    mc_obj = mc.Mc(stack, source, detectors, cl_devices=devices[a.device])
    mc_obj.rmax = 1.0                             # 1 m: photons are never killed laterally

    for d, M in layers(a.case):
        print("layer %g mm, tensor (1/mm):\n" % d, np.round(M, 4))
    t0 = time.time()
    done, res = 0, None
    while done < a.nphoton:
        n = int(min(a.batch, a.nphoton - done))
        if res is None:
            _, _, res = mc_obj.run(n, verbose=False)
        else:
            _, _, res = mc_obj.run(n, out=(None, None, res), verbose=False)
        done += n
        print("  %d / %d photons, %.1f s" % (done, int(a.nphoton), time.time() - t0))

    # weight per launched photon in each (pl, y, x) bin (not divided by the bin area)
    top = res.top.raw / max(res.top.nphotons, 1)
    bottom = res.bottom.raw / max(res.bottom.nphotons, 1)
    specular = float(np.sum(res.specular.reflectance))
    out = os.path.join(HERE, "pyxopto_%s.npz" % a.case)
    np.savez_compressed(out, top=top, bottom=bottom, specular=specular,
                        nphoton=done, seconds=time.time() - t0, case=a.case)
    print("R=%.5f T=%.5f specular=%.5f -> %s" % (top.sum(), bottom.sum(), specular, out))


if __name__ == "__main__":
    main()
