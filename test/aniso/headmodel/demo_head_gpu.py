#!/usr/bin/env python3
"""First GPU demo on the Colin27 head (CLAUDE.md step 2h): isotropic vs anisotropic white matter.

    python demo_head_gpu.py --nphoton 3e7          # conda env pyxoptoenv, after optics.py; GPU OpenCL
    python demo_head_gpu.py --nphoton 2e8 --batch 5e6 --compute cuda   # pmmc built with CUDA
    python demo_head_gpu.py --plot-only            # figures and numbers from OUT/demo_head_gpu.npz

Two runs with the same source, seeds and photons, differing only in the WM (label 4):
  iso:   mu_s = 122.89 1/mm (mu's 3.687), the isotropic value with the same mean D as the tensors;
  aniso: per-element scattering tensor M of optics.py (OFA = FA, mean D fixed).
Other tissues isotropic, Pini et al. 2025 Table 1 (830 nm): scalp+skull, CSF, GM (n, g, mua, mu's);
WM n 1.39, g 0.97, mua 0.043 1/mm. With a tensor given MMC ignores the prop mus column for all
elements, so non-WM elements get mu_s I of their label in elemmus (both runs go through elemmus).
Source: pencil beam on the scalp over the left central region (about C3), along the inward normal.

Outputs (OUT = HEADMODEL_DATA/out):
  demo_head_gpu.npz        nodal CW fluence per batch, exit-photon tallies per batch, and the exits at
                           15-45 mm from the source photon by photon (<case>_raw: distance, azimuth,
                           L_1..L_4, w0; <case>_rawb: batch), for demo_head_td.py
  demo_head_fluence.png    three planes through a point 20 mm under the source: log10 fluence (iso)
                           and relative difference (aniso - iso) / iso where the fluence > 1e-6 of its
                           maximum; log fluence in viridis with tissue boundaries red, difference in
                           RdBu_r (symmetric limits) with tissue boundaries black and
                           |difference| = 3 standard errors (from the batch spread) solid black (thicker);
                           source dot black
  demo_head_gm.png         brain (GM) sensitivity: mean partial path in GM of the photons leaving the
                           scalp vs source-exit distance, and vs azimuth on the 25-35 mm ring
  demo_head_gpu.txt        the numbers
Detected photons: every exit through the scalp (one detector covering the head, issaveexit), weight
w0 exp(-sum mua_k L_k) with no Russian roulette (minenergy 0: detp stores the launch weight) and
tend = 5 ns. The mean partial path in GM, <L_GM> = d(-ln R)/d(mua_GM), is the CW sensitivity of the
detected signal to an absorption change in the grey matter.
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from common import OUT, MESH_ORIGIN, TISSUE, load_mesh

# label: (n, g, mua 1/mm, mu's 1/mm); Pini et al. 2025 Table 1 for 1-3, WM from optics.py
TISS = {1: (1.39, 0.89, 0.019, 0.86), 2: (1.34, 0.89, 0.0026, 0.16), 3: (1.39, 0.90, 0.030, 0.70)}
SRC_DIR_WORLD = np.array([-0.55, 0.0, 0.83])   # from the brain centre towards ~C3 (left, superior)
BRAIN_CENTRE_WORLD = np.array([0.0, -20.0, 20.0])
TEND = 5e-9
RBINS = np.arange(0.0, 62.5, 5.0)               # source-exit distance bins, mm
RING = (25.0, 35.0)
NAZ = 12
DIFF_FLOOR = 1e-6                               # difference maps: fluence > DIFF_FLOOR x max
RAW_R = (15.0, 45.0)                            # exits saved photon by photon (time domain)
DEPTH = 20.0                                    # planes through src + DEPTH * srcdir
FRAC = os.path.join(OUT, "demo_head_gpu")


def exterior_faces(elem):
    """Faces (0-based node triplets) on the outer surface and the opposite node (inside)."""
    loc = np.array([[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [1, 2, 3, 0]])
    e = elem - 1
    f = e[:, loc[:, :3]].reshape(-1, 3)
    opp = e[:, loc[:, 3]].reshape(-1)
    key = np.sort(f, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    once = cnt[inv.ravel()] == 1
    return f[once], opp[once]


def find_source(node, elem):
    """Scalp point hit by the ray from the brain centre along SRC_DIR_WORLD; inward unit normal."""
    f, opp = exterior_faces(elem)
    c = node[f].mean(1)
    o = BRAIN_CENTRE_WORLD + MESH_ORIGIN
    u = SRC_DIR_WORLD / np.linalg.norm(SRC_DIR_WORLD)
    t = (c - o) @ u
    perp = np.linalg.norm((c - o) - t[:, None] * u, axis=1)
    cand = np.flatnonzero((t > 0) & (perp < 2.0))
    k = cand[np.argmax(t[cand])]                       # outermost face along the ray
    a, b, cc = node[f[k]]
    nrm = np.cross(b - a, cc - a)
    nrm /= np.linalg.norm(nrm)
    if (node[opp[k]] - a) @ nrm < 0:
        nrm = -nrm
    # the ray point on that face plane
    tp = ((a - o) @ nrm) / (u @ nrm)
    return o + tp * u, nrm


class Locator:
    """Point -> (element, barycentric weights) in the tetrahedral mesh, by nearest centroids."""

    def __init__(self, node, elem, k=48):
        self.node, self.e = node, elem - 1
        v = node[self.e]
        self.v0 = v[:, 0]
        self.Tinv = np.linalg.inv(np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], -1))
        self.tree = cKDTree(v.mean(1))
        self.k = k

    def __call__(self, X):
        _, cand = self.tree.query(X, k=self.k)
        lam = np.einsum("mkij,mkj->mki", self.Tinv[cand], X[:, None, :] - self.v0[cand])
        bary = np.concatenate([1 - lam.sum(-1, keepdims=True), lam], -1)
        ok = (bary >= -1e-9).all(-1)
        first = np.argmax(ok, 1)
        found = ok[np.arange(len(X)), first]
        el = np.where(found, cand[np.arange(len(X)), first], -1)
        return el, bary[np.arange(len(X)), first]

    def interp(self, el, bary, nodal):
        val = (nodal[self.e[np.maximum(el, 0)]] * bary).sum(-1)
        return np.where(el >= 0, val, np.nan)


def tensors(label, M_wm, aniso):
    """elemmus (ne, 6) for one run: mu_s I for non-WM labels, WM tensor or isotropic WM."""
    E = np.zeros((len(label), 6), np.float32)
    for l, (n, g, mua, musr) in TISS.items():
        E[label == l, :3] = musr / (1 - g)
    wm = label == 4
    E[wm] = M_wm[wm] if aniso else 0.0
    if not aniso:
        E[wm, :3] = MUS_ISO
    return E


def simulate(nphoton, batch, gpuid, seed, verbose, resume=False, compute="opencl"):
    import pmmc
    node, elem, label = load_mesh()
    o = np.load(os.path.join(OUT, "optics_wm.npz"))
    global MUS_ISO, WM
    WM = (float(o["n"]), float(o["g"]), 0.043)
    C0 = 299.792458
    MUS_ISO = C0 / WM[0] / (3 * float(o["d_mean"])) / (1 - WM[1])   # same mean D, isotropic
    src, sdir = find_source(node, elem)
    prop = [[0, 0, 1, 1]] + [[TISS[l][2], TISS[l][3] / (1 - TISS[l][1]), TISS[l][1], TISS[l][0]] for l in (1, 2, 3)]
    prop += [[WM[2], MUS_ISO, WM[1], WM[0]]]
    mua = np.array([p[0] for p in prop[1:]])
    # tangent frame at the source: e1 = anterior (y) projected, e2 = n x e1
    e1 = np.array([0.0, 1.0, 0.0]) - sdir[1] * sdir
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(sdir, e1)
    nb = int(np.ceil(nphoton / batch))
    nr, res = len(RBINS) - 1, {}
    for case in ("iso", "aniso"):
        part = FRAC + "_%s.npz" % case                # each case saved when done; --resume reuses it
        if resume and os.path.exists(part):
            z = np.load(part)
            if int(z["nb"]) == nb and float(z["batch"]) == batch and int(z["seed"]) == seed:
                print("  %s: reusing %s" % (case, part), flush=True)
                res[case] = {k: z[k] for k in ("flux", "rad", "az", "raw", "rawb", "seconds")}
                continue
        E = tensors(label, o["M"], case == "aniso")
        flux = np.zeros((nb, len(node)))
        tal = np.zeros((nb, nr, 6))        # per distance bin: sum w, w L_1..L_4, count
        az = np.zeros((nb, NAZ, 6))
        raw, rawb = [], []                 # per-photon data of the exits RAW_R[0] <= r < RAW_R[1]
        t0 = time.time()
        for k in range(nb):
            cfg = dict(nphoton=int(batch), seed=seed + 7919 * k, node=node, elem=elem, elemprop=label,
                       prop=prop, elemmus=E, srcpos=list(src + 1e-3 * sdir), srcdir=list(sdir),
                       tstart=0, tend=TEND, tstep=TEND, method="elem", compute=compute, gpuid=gpuid,
                       issaveexit=1, detpos=[list(node.mean(0)) + [1000.0]], maxdetphoton=int(batch),
                       minenergy=0.0)
            if not verbose:
                sys.stdout.flush()
                fd, keep = sys.stdout.fileno(), os.dup(sys.stdout.fileno())
                os.dup2(os.open(os.devnull, os.O_WRONLY), fd)
            try:
                r = pmmc.mmclab(cfg)
            finally:
                if not verbose:
                    os.dup2(keep, fd)
                    os.close(keep)
            flux[k] = np.asarray(r["flux"])[:, 0] * TEND          # CW fluence (1/mm^2 per photon)
            d = r["detp"]
            p = np.asarray(d["p"])
            L = np.asarray(d["ppath"]).reshape(len(p), -1)
            w = np.asarray(d["w0"]) * np.exp(-L @ mua)
            rel = p - src
            rd = np.linalg.norm(rel, axis=1)
            ib = np.digitize(rd, RBINS) - 1
            q = np.stack([w, w * L[:, 0], w * L[:, 1], w * L[:, 2], w * L[:, 3], np.ones_like(w)], 1)
            for j in range(nr):
                tal[k, j] = q[ib == j].sum(0)
            ring = (rd >= RING[0]) & (rd < RING[1])
            phi = np.mod(np.arctan2(rel @ e2, rel @ e1), 2 * np.pi)
            ia = (phi / (2 * np.pi / NAZ)).astype(int) % NAZ
            for j in range(NAZ):
                az[k, j] = q[ring & (ia == j)].sum(0)
            sel = (rd >= RAW_R[0]) & (rd < RAW_R[1])
            raw.append(np.column_stack([rd[sel], phi[sel], L[sel], np.asarray(d["w0"])[sel]]).astype(np.float32))
            rawb.append(np.full(sel.sum(), k, np.int16))
            print("  %s: batch %d/%d, %d exits, %.0f s" % (case, k + 1, nb, len(w), time.time() - t0), flush=True)
            del r, d, p, L, w, rel, rd, ib, q, ring, phi, ia, sel     # keep the peak memory per batch low
        res[case] = dict(flux=flux, rad=tal, az=az, raw=np.concatenate(raw), rawb=np.concatenate(rawb),
                         seconds=time.time() - t0)
        np.savez(part, nb=nb, batch=batch, seed=seed, **res[case])
    np.savez_compressed(FRAC + ".npz", src=src, sdir=sdir, e1=e1, e2=e2, batch=batch, nb=nb, compute=compute,
                        mus_iso=MUS_ISO, prop=np.array(prop),
                        **{"%s_%s" % (c, k): v for c in res for k, v in res[c].items()})


def ratio_stats(a, b, num, den=0):
    """Batch-based mean and standard error of sum(num)/sum(den) for a, and of the ratio b/a - 1."""
    ra = a[..., num].sum(0) / a[..., den].sum(0)
    rb = b[..., num].sum(0) / b[..., den].sum(0)
    nb = a.shape[0]
    # jackknife over batches
    ja = np.array([(a[..., num].sum(0) - a[i, ..., num]) / (a[..., den].sum(0) - a[i, ..., den]) for i in range(nb)])
    jb = np.array([(b[..., num].sum(0) - b[i, ..., num]) / (b[..., den].sum(0) - b[i, ..., den]) for i in range(nb)])
    jr = jb / ja - 1
    se = lambda j: np.sqrt((nb - 1) / nb * ((j - j.mean(0)) ** 2).sum(0))
    return ra, se(ja), rb, se(jb), rb / ra - 1, se(jr)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(FRAC + ".npz")
    node, elem, label = load_mesh()
    src, sdir = z["src"], z["sdir"]
    log = []

    def say(s):
        print(s)
        log.append(s)

    nb, batch = int(z["nb"]), float(z["batch"])
    say("Colin27 head, %s GPU, %d x %.3g photons per case (iso %.0f s, aniso %.0f s)"
        % (str(z["compute"]) if "compute" in z else "opencl", nb, batch, float(z["iso_seconds"]), float(z["aniso_seconds"])))
    say("source (mesh frame) %s mm, world RAS %s mm, inward direction %s"
        % (np.round(src, 1), np.round(src - MESH_ORIGIN, 1), np.round(sdir, 3)))
    say("WM isotropic mu_s %.2f 1/mm (mu's %.3f); prop rows [mua mus g n]: %s"
        % (float(z["mus_iso"]), float(z["mus_iso"]) * 0.03, np.round(z["prop"][1:], 4).tolist()))

    # ---- fluence maps, three planes
    Fi, Fa = z["iso_flux"], z["aniso_flux"]
    loc = Locator(node, elem)
    P = src + DEPTH * sdir
    h = 0.5
    ext = 70.0
    gr = np.arange(-ext, ext + h / 2, h)
    U, V = np.meshgrid(gr, gr, indexing="xy")
    planes = [("sagittal (x = %.0f)" % (P[0] - MESH_ORIGIN[0]), 1, 2, 0),
              ("coronal (y = %.0f)" % (P[1] - MESH_ORIGIN[1]), 0, 2, 1),
              ("axial (z = %.0f)" % (P[2] - MESH_ORIGIN[2]), 0, 1, 2)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.6), constrained_layout=True)
    axn = ["x (L→R), mm", "y (P→A), mm", "z (I→S), mm"]
    for j, (title, a1, a2, a0) in enumerate(planes):
        X = np.zeros((U.size, 3))
        X[:, a0] = P[a0]
        X[:, a1] = P[a1] + U.ravel()
        X[:, a2] = P[a2] + V.ravel()
        el, bary = loc(X)
        lab = np.where(el >= 0, label[np.maximum(el, 0)], 0).reshape(U.shape)
        fi = np.array([loc.interp(el, bary, f) for f in Fi])          # per batch
        fa = np.array([loc.interp(el, bary, f) for f in Fa])
        mi, ma = fi.mean(0), fa.mean(0)
        rel = ma / mi - 1
        # jackknife SE of the ratio over batches
        jr = np.array([(fa.sum(0) - fa[i]) / (fi.sum(0) - fi[i]) - 1 for i in range(nb)])
        se = np.sqrt((nb - 1) / nb * ((jr - jr.mean(0)) ** 2).sum(0))
        mi, rel, se = mi.reshape(U.shape), rel.reshape(U.shape), se.reshape(U.shape)
        floor = np.nanmax(mi) * 1e-8
        good = mi > floor
        good_d = mi > np.nanmax(mi) * DIFF_FLOOR          # difference shown only where it is not pure noise
        cu, cv = (P[a1] + gr) - MESH_ORIGIN[a1], (P[a2] + gr) - MESH_ORIGIN[a2]
        s1, s2 = src[a1] - MESH_ORIGIN[a1], src[a2] - MESH_ORIGIN[a2]
        for row, (img, cmap_lab, kw) in enumerate((
                (np.log10(np.where(good, mi, np.nan)), "log10 fluence (iso), 1/mm² per photon",
                 dict(vmin=np.log10(floor), vmax=np.log10(np.nanmax(mi)))),
                (100 * np.where(good_d, rel, np.nan), "(aniso − iso) / iso, %", dict(vmin=-10, vmax=10)))):
            a = ax[row, j]
            # signed difference: RdBu_r, symmetric limits, black/grey contours (CLAUDE.md plot rule)
            diff = row == 1
            if diff:
                a.set_facecolor("0.85")                            # masked (noise) area, not zero
            im = a.pcolormesh(cu, cv, img, cmap="RdBu_r" if diff else "viridis", shading="auto", **kw)
            a.contour(cu, cv, lab, levels=[0.5, 1.5, 2.5, 3.5], colors="black" if diff else "red", linewidths=0.6)
            if diff:
                sig = np.where(good_d, np.abs(rel) / np.maximum(se, 1e-12), 0)
                a.contour(cu, cv, sig, levels=[3.0], colors="black", linewidths=1.6)
            a.plot(s1, s2, "o", ms=7, mfc="black", mec="white")
            a.set_aspect("equal")
            a.set_title(title)
            a.set_xlabel(axn[a1])
            a.set_ylabel(axn[a2])
            fig.colorbar(im, ax=a, shrink=0.8, label=cmap_lab)
        say("plane %-18s: |rel. diff| median %.1f%%, 95th pct %.1f%% (where fluence > 1e-4 of max); "
            "median SE %.1f%% there" % (title, 100 * np.nanmedian(np.abs(rel[mi > np.nanmax(mi) * 1e-4])),
                                        100 * np.nanpercentile(np.abs(rel[mi > np.nanmax(mi) * 1e-4]), 95),
                                        100 * np.nanmedian(se[mi > np.nanmax(mi) * 1e-4])))
    fig.suptitle("Colin27, source ~C3 (dot); planes through the point %d mm under the source.\nTissue "
                 "boundaries: red (top), black (bottom).  Thick black: |Δ| = 3 SE.  Grey fill: fluence < %g of max"
                 % (DEPTH, DIFF_FLOOR))
    fig.savefig(FRAC.replace("demo_head_gpu", "demo_head_fluence") + ".png", dpi=130)
    plt.close(fig)

    # ---- absorbed fraction per tissue from the nodal fluence
    v = node[elem - 1]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), v[:, 3] - v[:, 0])) / 6
    mua_l = z["prop"][:, 0]
    for case, F in (("iso", Fi), ("aniso", Fa)):
        fe = F.mean(0)[elem - 1].mean(1)
        absd = [np.sum(fe[label == l] * vol[label == l]) * mua_l[l] for l in (1, 2, 3, 4)]
        say("absorbed fraction per launched photon (%s): %s" % (case, ", ".join(
            "%s %.4f" % (TISSUE[l], a) for l, a in zip((1, 2, 3, 4), absd))))

    # ---- GM sensitivity of the exiting photons
    rc = 0.5 * (RBINS[1:] + RBINS[:-1])
    ri, ra = z["iso_rad"], z["aniso_rad"]
    gi, sgi, ga, sga, dg, sdg = ratio_stats(ri, ra, 3)
    wi, _, wa, _, dw, sdw = ratio_stats(ri, ra, 4)
    Ri = ri[..., 0].sum(0) / (nb * batch)
    Ra = ra[..., 0].sum(0) / (nb * batch)
    say("exit tallies vs distance from the source (mm): bin centre | R iso, R aniso (per photon) | "
        "<L_GM> iso, aniso (mm) | change | <L_WM> change")
    for j in range(len(rc)):
        say("  %5.1f | %.3e %.3e | %7.3f %7.3f | %+6.2f%% ± %.2f | %+6.2f%% ± %.2f"
            % (rc[j], Ri[j], Ra[j], gi[j], ga[j], 100 * dg[j], 100 * sdg[j], 100 * dw[j], 100 * sdw[j]))
    ai, azi = z["iso_az"], z["aniso_az"]
    agi, asgi, aga, asga, adg, asdg = ratio_stats(ai, azi, 3)
    say("ring %g-%g mm, <L_GM> by azimuth (0 = anterior, 90 = towards e2 = n x anterior):" % RING)
    phc = (np.arange(NAZ) + 0.5) * 360 / NAZ
    for j in range(NAZ):
        say("  %5.1f deg: iso %.3f ± %.3f, aniso %.3f ± %.3f mm, change %+6.2f%% ± %.2f"
            % (phc[j], agi[j], asgi[j], aga[j], asga[j], 100 * adg[j], 100 * asdg[j]))
    say("e2 (mesh frame) = %s: azimuth 90 deg points %s"
        % (np.round(z["e2"], 2), "superior/medial" if z["e2"][2] > 0 else "inferior"))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    ok = (ri[..., 5].sum(0) > 1000) & (rc > 2.5)
    ax[0].semilogy(rc[ok], gi[ok], "o-", color="black", label="iso WM")
    ax[0].semilogy(rc[ok], ga[ok], "s--", color="red", label="aniso WM")
    ax[0].set_xlabel("source–exit distance, mm")
    ax[0].set_ylabel("mean partial path in GM, mm")
    ax[0].legend()
    ax[1].errorbar(rc[ok], 100 * dg[ok], 2 * sdg[ok] * 100, fmt="o-", color="black", capsize=3, label="<L_GM>")
    ax[1].errorbar(rc[ok], 100 * dw[ok], 2 * sdw[ok] * 100, fmt="s--", color="red", capsize=3, label="<L_WM>")
    ax[1].axhline(0, color="gray", lw=0.7)
    ax[1].set_xlabel("source–exit distance, mm")
    ax[1].set_ylabel("aniso / iso − 1, %  (±2 SE)")
    ax[1].legend()
    ax[2].errorbar(phc, 100 * adg, 2 * asdg * 100, fmt="o-", color="black", capsize=3)
    ax[2].axhline(0, color="gray", lw=0.7)
    ax[2].set_xticks(np.arange(0, 361, 90))
    ax[2].set_xlabel("azimuth around the source, deg (0 = anterior)")
    ax[2].set_ylabel("<L_GM> aniso / iso − 1, %  (±2 SE)")
    ax[2].set_title("ring %g–%g mm" % RING)
    fig.suptitle("Brain (GM) sensitivity of the photons leaving the scalp: <L_GM> = d(−ln R)/dμa,GM")
    fig.savefig(FRAC.replace("demo_head_gpu", "demo_head_gm") + ".png", dpi=130)
    plt.close(fig)
    with open(FRAC + ".txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nphoton", type=float, default=3e7)
    ap.add_argument("--batch", type=float, default=3e6)
    ap.add_argument("--gpuid", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1648335518)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--compute", default="opencl", choices=["opencl", "cuda"])
    ap.add_argument("--resume", action="store_true", help="reuse a finished case (demo_head_gpu_<case>.npz)")
    a = ap.parse_args()
    if not a.plot_only:
        simulate(a.nphoton, a.batch, a.gpuid, a.seed, a.verbose, a.resume, a.compute)
    plot()


if __name__ == "__main__":
    main()
