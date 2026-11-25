#!/usr/bin/env python3
"""Validation of anisotropic MMC (branch aniso) against the generalized ADE.

Slab of thickness L, laterally wide, pencil beam along +z at the centre of the
top face. The scattering tensor is diagonal (principal axes = lab axes) or
rotated around z by `--rot` degrees. Transmitted photons (exit z = L) give:
  - T(t): time-resolved total transmittance, vs generalized_ade.tt_ade
  - var(x)(t), var(y)(t), cov(x,y)(t) of the transmitted profile, whose slopes
    in the diffusive regime are 2*D_i (vs generalized_ade.d_tensor_ade)

Usage:
  python3 slab_vs_ade.py --mus 1 3 4 --g 0 --nphoton 2e6 [--rot 30]

Requires: numpy, matplotlib, generalized_ade (github.com/epini/ADE),
and the mmc binary of this branch built in ../../bin/mmc.
"""

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np

C_MM_NS = 299.792458  # speed of light in vacuum, mm/ns
HERE = os.path.dirname(os.path.abspath(__file__))
MMC = os.path.join(HERE, "..", "..", "bin", "mmc")


def build_case(tag, L, W, mus, g, n, mua, rot_deg, nphoton, tend, seed):
    """Write mesh-in-JSON input and the per-element tensor file."""
    sx, sz = W / 60.0, L / 60.0  # scale the 6-tet cube of examples/validation
    cube_nodes = [[0, 0, 0], [60, 0, 0], [0, 60, 0], [60, 60, 0],
                  [0, 0, 60], [60, 0, 60], [0, 60, 60], [60, 60, 60]]
    nodes = [[x * sx, y * sx, z * sz] for x, y, z in cube_nodes]
    elems = [[1, 2, 8, 4, 1], [1, 3, 4, 8, 1], [1, 2, 6, 8, 1],
             [1, 5, 8, 6, 1], [1, 3, 8, 7, 1], [1, 5, 7, 8, 1]]

    th = np.deg2rad(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    M = R @ np.diag(mus) @ R.T
    packed = np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]], np.float32)
    np.tile(packed, (len(elems), 1)).astype(np.float32).tofile(os.path.join(HERE, tag + "_elemmus.bin"))

    cfg = {
        "Session": {"Photons": int(nphoton), "RNGSeed": seed, "ID": tag},
        "Domain": {"Media": [{"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0},
                             {"mua": mua, "mus": float(np.mean(mus)), "g": g, "n": n}]},
        "Mesh": {"MeshNode": nodes, "MeshElem": elems, "InitElem": 2,
                 "ElemMusFile": tag + "_elemmus.bin"},
        "Forward": {"T0": 0.0, "T1": tend, "Dt": tend},
        "Optode": {"Source": {"Pos": [W / 2 + 0.17, W / 2 + 0.33, 0.0], "Dir": [0, 0, 1]},
                   "Detector": [{"Pos": [W / 2, W / 2, L], "R": 2 * W}]},
    }
    with open(os.path.join(HERE, tag + ".json"), "w") as f:
        json.dump(cfg, f, indent=1)
    return M


def load_mch(fname):
    """Minimal reader of the MCX/MMC detected-photon file (see matlab/loadmch.m)."""
    with open(fname, "rb") as f:
        buf = f.read()
    rows, pos, hdr = [], 0, None
    while pos + 4 <= len(buf) and buf[pos:pos + 4] == b"MCXH":
        hd = struct.unpack_from("<7I", buf, pos + 4)
        unitmm, seedbyte, normalizer, respin, srcnum, savedetflag = struct.unpack_from("<fIfiII", buf, pos + 32)
        pos += 4 + 7 * 4 + 6 * 4 + 2 * 4
        maxmedia, colcount, saved = hd[1], hd[3], hd[6]
        dat = np.frombuffer(buf, "<f4", saved * colcount, pos).reshape(saved, colcount)
        pos += saved * colcount * 4 + saved * seedbyte
        rows.append(dat)
        hdr = dict(maxmedia=maxmedia, total=hd[4], savedetflag=savedetflag, unitmm=unitmm)
    data = np.vstack(rows)
    # column layout from savedetflag bits: detid, nscat, ppath, mom, p, v, w
    lens = [1, hdr["maxmedia"], hdr["maxmedia"], hdr["maxmedia"], 3, 3, 1]
    if hdr["savedetflag"] == 0:
        # mmc writes 0 in the header: default flags (detid, nscat, ppath, w) + exit p, v with -x 1
        base = 1 + 2 * hdr["maxmedia"] + 1
        hdr["savedetflag"] = 0x47 | (0x30 if data.shape[1] == base + 6 else 0)
    names = ["detid", "nscat", "ppath", "mom", "p", "v", "w"]
    cols, c = {}, 0
    for b, (name, ln) in enumerate(zip(names, lens)):
        if hdr["savedetflag"] & (1 << b):
            cols[name] = data[:, c:c + ln]
            c += ln
    if "ppath" in cols:
        cols["ppath"] = cols["ppath"] * hdr["unitmm"]
    return cols, hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mus", type=float, nargs=3, default=[1.0, 3.0, 4.0])
    ap.add_argument("--g", type=float, default=0.0)
    ap.add_argument("--n", type=float, default=1.3)
    ap.add_argument("--mua", type=float, default=0.0)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--W", type=float, default=200.0)
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--nphoton", type=float, default=1e6)
    ap.add_argument("--tend", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1648335518)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()

    tag = a.tag or "slab_mus%g-%g-%g_g%g_rot%g" % (*a.mus, a.g, a.rot)
    M = build_case(tag, a.L, a.W, a.mus, a.g, a.n, a.mua, a.rot, a.nphoton, a.tend, a.seed)

    cmd = [MMC, "-f", tag + ".json", "-G", "-1", "-M", "S", "-S", "0", "-d", "1", "-x", "1",
           "-H", str(int(a.nphoton)), "-b", "1"]
    out = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if out.returncode != 0 or "ERROR" in out.stdout + out.stderr:
        print(out.stdout[-2000:], out.stderr[-2000:])
        sys.exit(1)
    speed = [l for l in out.stdout.splitlines() if "speed" in l]

    cols, hdr = load_mch(os.path.join(HERE, tag + ".mch"))
    p = cols["p"]
    t = cols["ppath"][:, 0] * a.n / C_MM_NS            # ns
    w = cols["w"][:, 0] if "w" in cols else np.ones(len(t))
    tr = np.abs(p[:, 2] - a.L) < 1e-3                   # transmitted
    x, y, t, w = p[tr, 0] - a.W / 2, p[tr, 1] - a.W / 2, t[tr], w[tr]
    N = hdr["total"]

    from generalized_ade import d_tensor_ade, tt_ade
    Dx, Dy, Dz = d_tensor_ade(a.n, *a.mus, a.g)

    # T(t)
    dt = 0.02
    edges = np.arange(0, a.tend + dt / 2, dt)
    tc = 0.5 * (edges[1:] + edges[:-1])
    Tmc, _ = np.histogram(t, edges, weights=w)
    Tmc = Tmc / N / dt
    Tth = tt_ade(tc, a.L, a.n, 1.0, *a.mus, a.g, a.mua)
    Ttot_mc = w.sum() / N

    # second moments of the transmitted profile vs time (rotated frame of the tensor)
    th = np.deg2rad(a.rot)
    xr = np.cos(th) * x + np.sin(th) * y
    yr = -np.sin(th) * x + np.cos(th) * y
    tb = np.arange(0.3, 2.0 + 1e-9, 0.1)
    vx, vy, cxy, tm = [], [], [], []
    for t0, t1 in zip(tb[:-1], tb[1:]):
        s = (t >= t0) & (t < t1)
        if s.sum() < 200:
            continue
        ww = w[s]
        mx, my = np.average(xr[s], weights=ww), np.average(yr[s], weights=ww)
        vx.append(np.average((xr[s] - mx) ** 2, weights=ww))
        vy.append(np.average((yr[s] - my) ** 2, weights=ww))
        cxy.append(np.average((xr[s] - mx) * (yr[s] - my), weights=ww))
        tm.append(np.average(t[s], weights=ww))
    tm, vx, vy, cxy = map(np.array, (tm, vx, vy, cxy))
    # slope of variance vs time = 2 D (diffusive regime, profile is a Gaussian translated in time)
    Dx_mc = np.polyfit(tm, vx, 1)[0] / 2
    Dy_mc = np.polyfit(tm, vy, 1)[0] / 2

    sel = Tth > 0.05 * Tth.max()
    relerr = np.abs(Tmc[sel] - Tth[sel]) / Tth[sel]

    print("case            :", tag)
    print("photons         : %d, transmitted %d" % (N, tr.sum()), *speed)
    print("D_ADE  (mm2/ns) : Dx=%.3f Dy=%.3f Dz=%.3f" % (Dx, Dy, Dz))
    print("D_MC   (mm2/ns) : Dx=%.3f Dy=%.3f   (from var slope, tensor frame)" % (Dx_mc, Dy_mc))
    print("D rel. diff     : Dx %+.2f%%  Dy %+.2f%%" % (100 * (Dx_mc / Dx - 1), 100 * (Dy_mc / Dy - 1)))
    print("cov(x',y')/sqrt(var x' var y') mean: %+.3f (0 if axes match the tensor frame)"
          % np.mean(cxy / np.sqrt(vx * vy)))
    print("T total MC      : %.5f" % Ttot_mc)
    print("T(t) median |rel err| where T > 5%% of peak: %.2f%%" % (100 * np.median(relerr)))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].semilogy(tc, Tmc, ".", ms=3, label="MMC aniso")
        ax[0].semilogy(tc, Tth, "-", label="ADE")
        ax[0].set_xlabel("t (ns)"); ax[0].set_ylabel("T(t) (1/ns)"); ax[0].legend()
        ax[0].set_ylim(Tth.max() * 1e-4, Tth.max() * 3)
        ax[1].plot(tm, vx, "o", label="var x' MC"); ax[1].plot(tm, vy, "s", label="var y' MC")
        ax[1].plot(tm, 2 * Dx * tm + np.polyfit(tm, vx, 1)[1], "-", label="slope 2Dx ADE")
        ax[1].plot(tm, 2 * Dy * tm + np.polyfit(tm, vy, 1)[1], "-", label="slope 2Dy ADE")
        ax[1].set_xlabel("t (ns)"); ax[1].set_ylabel("variance (mm$^2$)"); ax[1].legend()
        fig.suptitle(tag)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=130)


if __name__ == "__main__":
    main()
