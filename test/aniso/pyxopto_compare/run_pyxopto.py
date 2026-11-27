#!/usr/bin/env python3
"""PyXOpto side of the comparison: run on a machine with PyXOpto + an OpenCL GPU.

    python run_pyxopto.py --case A --nphoton 1e8
    python run_pyxopto.py --case HY --nphoton 1e8

Writes pyxopto_<case>.npz next to this script. Uses mcml AnisotropicLayer with the
full 3x3 scattering tensor from common.py and a scalar Henyey-Greenstein phase function,
one layer per layer of the case (own mua, g, n). Besides the mean, top2/bottom2 = variance
of the mean per bin, estimated from the spread between the batches (the PyXOpto roulette
makes the counting estimate wrong at low weight).
"""

import argparse
import os
import time

import numpy as np
from xopto.mcml import mc

from common import CASES, axes, stack

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A", choices=sorted(CASES))
    ap.add_argument("--nphoton", type=float, default=1e8)
    ap.add_argument("--batch", type=float, default=1e7, help="photons per run() call")
    ap.add_argument("--device", type=int, default=0, help="index in mc.clinfo.gpus()")
    a = ap.parse_args()

    mm = 1e-3
    xy, pl = axes(a.case)
    lay = stack(a.case)

    # one AnisotropicLayer per slab layer, tensors in 1/m, between two d = 0 outer layers (n = 1)
    slab = [mc.mclayer.AnisotropicLayer(d=l["d"] * mm, n=l["n"], mua=l["mua"] / mm, mus=l["M"] / mm,
                                        pf=mc.mcpf.Hg(l["g"])) for l in lay]
    outer = [mc.mclayer.AnisotropicLayer(d=0.0, n=1.0, mua=0.0, mus=0.0, pf=mc.mcpf.Hg(lay[0]["g"]))
             for _ in range(2)]
    layers = mc.mclayer.Layers([outer[0]] + slab + [outer[1]])
    source = mc.mcsource.Line((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    def det():
        return mc.mcdetector.CartesianPl(
            mc.mcdetector.Axis(xy[0] * mm, xy[1] * mm, xy[2]),
            mc.mcdetector.Axis(xy[0] * mm, xy[1] * mm, xy[2]),
            plaxis=mc.mcdetector.Axis(pl[0] * mm, pl[1] * mm, pl[2]))

    detectors = mc.mcdetector.Detectors(top=det(), bottom=det(), specular=mc.mcdetector.Total())
    devices = mc.clinfo.gpus() or mc.clinfo.cpus()   # CPU fallback (e.g. pocl) for testing
    print("OpenCL devices:", devices)
    mc_obj = mc.Mc(layers, source, detectors, cl_devices=devices[a.device])
    mc_obj.rmax = 1.0                             # 1 m: photons are never killed laterally

    for l in lay:
        print("%s: %g mm, n %g, g %g, mua %g, tensor (1/mm):\n" % (l["name"], l["d"], l["n"], l["g"], l["mua"]),
              np.round(l["M"], 4))
    t0 = time.time()
    done, nb = 0, 0
    acc = {f: 0.0 for f in ("top", "bottom", "top2", "bottom2")}
    specular = 0.0
    while done < a.nphoton:
        n = int(min(a.batch, a.nphoton - done))
        _, _, res = mc_obj.run(n, verbose=False)
        for f in ("top", "bottom"):
            h = getattr(res, f).raw / n              # weight per launched photon, this batch
            acc[f] = acc[f] + h * n
            acc[f + "2"] = acc[f + "2"] + h ** 2 * n
        specular += float(np.sum(res.specular.reflectance)) * n
        done += n
        nb += 1
        print("  %d / %d photons, %.1f s" % (done, int(a.nphoton), time.time() - t0))

    # mean per launched photon in each (pl, y, x) bin (not divided by the bin area); variance of the
    # mean from the weighted spread of the batch means (needs >= 2 batches)
    out = {}
    for f in ("top", "bottom"):
        mean = acc[f] / done
        out[f] = mean
        if nb > 1:
            out[f + "2"] = np.maximum(acc[f + "2"] / done - mean ** 2, 0.0) / (nb - 1)
    specular /= done
    path = os.path.join(HERE, "pyxopto_%s.npz" % a.case)
    np.savez_compressed(path, specular=specular, nphoton=done, nbatch=nb, seconds=time.time() - t0,
                        case=a.case, **out)
    print("R=%.5f T=%.5f specular=%.5f -> %s" % (out["top"].sum(), out["bottom"].sum(), specular, path))


if __name__ == "__main__":
    main()
