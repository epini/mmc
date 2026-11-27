#!/usr/bin/env python3
"""Whole-head transmission pilot (CLAUDE.md step 2c): Colin27, isotropic vs anisotropic WM.

    export HEADMODEL_DATA=~/headmodel_data
    python transmit.py --nphoton 1e6 --batch 5e5 --tag test          # quick test
    python transmit.py --nphoton 1e8 --batch 5e6 --tag pilot         # pilot, 20 batches
    python transmit.py --tag pilot --plot-only

Same optics as demo_head_gpu.py (Pini et al. 2025 Table 1 for scalp+skull, CSF, GM; WM tensors of
optics.py, or the isotropic WM with the same mean D). The corpus callosum (WM elements with JHU label
genu/body/splenium) gets its own label 5 with the WM optics, only so that MMC reports its partial path.
No Russian roulette (minenergy 0: absorption only as weight w0 exp(-sum mua_k L_k)), tend = 10 ns.
iso and aniso use the same seeds batch by batch (paired ratio, jackknife over batches).

Sources (pencil beam on the scalp along the inward normal; the point is where a ray from CENTRE hits
the scalp): 'ear' (T7-ish, left, detect right: along the CC axons) and 'forehead' (Fpz-ish, detect
occiput: across them). Detectors on the opposite side, around the antipodal scalp point A (the ray
from CENTRE the other way): exits within 10, 20, 30 mm of A, and every exit in the opposite hemisphere
((p - CENTRE) . u < 0, u = direction to the source). One MMC detector covers the head (issaveexit);
exits are selected afterwards.

Outputs in HEADMODEL_DATA/out/transmission/: transmit_<tag>.npz (nodal fluence per batch, the
opposite-hemisphere exits packet by packet), transmit_<tag>.txt (numbers),
transmit_<tag>_<source>_fluence.png (log10 fluence in viridis, aniso/iso - 1 in RdBu_r),
transmit_<tag>_weights.png (weight distribution and T vs angle from A).
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import OUT as HOUT, MESH_ORIGIN, load_mesh          # noqa: E402
from demo_head_gpu import Locator, exterior_faces                 # noqa: E402

OUT = os.path.join(HOUT, "transmission")
# label: (n, g, mua 1/mm, mu's 1/mm), as demo_head_gpu.py
TISS = {1: (1.39, 0.89, 0.019, 0.86), 2: (1.34, 0.89, 0.0026, 0.16), 3: (1.39, 0.90, 0.030, 0.70)}
WM_MUA = 0.043
NAMES = ["scalp+skull", "CSF", "GM", "WM (no CC)", "CC"]
JHU_CC = (3, 4, 5)                       # genu, body, splenium
CENTRE = np.array([0.0, -18.0, 5.0])      # world RAS, mm
SOURCES = {"ear": np.array([-1.0, 0.0, 0.0]),          # T7-ish -> right side
           "forehead": np.array([0.0, 1.0, -0.08])}    # Fpz-ish -> occiput
DISCS = [10.0, 20.0, 30.0]
TEND = 10e-9
C0 = 299.792458                          # mm/ns


def scalp_hit(node, elem, o, u):
    """Outermost exterior face hit by the ray o + t u; returns the point and the inward unit normal."""
    f, opp = exterior_faces(elem)
    a, b, cc = node[f[:, 0]], node[f[:, 1]], node[f[:, 2]]
    e1, e2 = b - a, cc - a                                   # Moller-Trumbore, all faces at once
    pv = np.cross(u, e2)
    det = (e1 * pv).sum(1)
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    tv = o - a
    bu = (tv * pv).sum(1) / det
    qv = np.cross(tv, e1)
    bv = (qv @ u) / det
    t = (e2 * qv).sum(1) / det
    hit = (bu >= 0) & (bv >= 0) & (bu + bv <= 1) & (t > 0)
    k = np.flatnonzero(hit)[np.argmax(t[hit])]               # outermost exterior face along the ray
    nrm = np.cross(e1[k], e2[k])
    nrm /= np.linalg.norm(nrm)
    if (node[opp[k]] - a[k]) @ nrm < 0:
        nrm = -nrm
    return o + t[k] * u, nrm


def geometry(node, elem):
    o = CENTRE + MESH_ORIGIN
    g = {}
    for s, u in SOURCES.items():
        u = u / np.linalg.norm(u)
        src, sdir = scalp_hit(node, elem, o, u)
        A, _ = scalp_hit(node, elem, o, -u)
        g[s] = dict(u=u, src=src, sdir=sdir, A=A)
    return o, g


def quiet_call(fun, *a, verbose=False):
    if verbose:
        return fun(*a)
    sys.stdout.flush()
    fd, keep = sys.stdout.fileno(), os.dup(sys.stdout.fileno())
    os.dup2(os.open(os.devnull, os.O_WRONLY), fd)
    try:
        return fun(*a)
    finally:
        os.dup2(keep, fd)
        os.close(keep)


def simulate(a):
    import pmmc
    node, elem, label = load_mesh()
    jhu = np.load(os.path.join(HOUT, "colin27_hcp1065_elem.npz"))["jhu"]
    lab = label.copy()
    lab[(label == 4) & np.isin(jhu, JHU_CC)] = 5
    o = np.load(os.path.join(HOUT, "optics_wm.npz"))
    n_wm, g_wm = float(o["n"]), float(o["g"])
    mus_iso = C0 / n_wm / (3 * float(o["d_mean"])) / (1 - g_wm)     # same mean D, isotropic
    prop = [[0, 0, 1, 1]] + [[TISS[l][2], TISS[l][3] / (1 - TISS[l][1]), TISS[l][1], TISS[l][0]] for l in (1, 2, 3)]
    prop += [[WM_MUA, mus_iso, g_wm, n_wm]] * 2
    mua = np.array([p[0] for p in prop[1:]])
    nidx = np.array([p[3] for p in prop[1:]])
    centre, geo = geometry(node, elem)
    nb = int(np.ceil(a.nphoton / a.batch))
    E = {}
    for case in ("iso", "aniso"):
        e = np.zeros((len(lab), 6), np.float32)
        for l, (n, g, _, musr) in TISS.items():
            e[lab == l, :3] = musr / (1 - g)
        wm = lab >= 4
        if case == "aniso":
            e[wm] = o["M"][wm]
        else:
            e[wm, :3] = mus_iso
        E[case] = e
    out = dict(nb=nb, batch=a.batch, seed=a.seed, compute=a.compute, mus_iso=mus_iso, prop=np.array(prop),
               centre=centre, ncc=int((lab == 5).sum()))
    for s in a.sources:
        G = geo[s]
        for k_, v in G.items():
            out["%s_%s" % (s, k_)] = v
        for case in ("iso", "aniso"):
            flux = np.zeros((nb, len(node)), np.float32)
            nexit = np.zeros(nb, np.int64)
            pk = []                                   # opposite-hemisphere exits: batch, p, L(5), w0
            t0 = time.time()
            for k in range(nb):
                cfg = dict(nphoton=int(a.batch), seed=a.seed + 7919 * k, node=node, elem=elem, elemprop=lab,
                           prop=prop, elemmus=E[case], srcpos=list(G["src"] + 1e-3 * G["sdir"]),
                           srcdir=list(G["sdir"]), tstart=0, tend=TEND, tstep=TEND, method="elem",
                           compute=a.compute, gpuid=a.gpuid, issaveexit=1,
                           detpos=[list(node.mean(0)) + [1000.0]], maxdetphoton=int(a.batch), minenergy=0.0)
                r = quiet_call(pmmc.mmclab, cfg, verbose=a.verbose)
                flux[k] = np.asarray(r["flux"])[:, 0] * TEND
                d = r["detp"]
                p = np.asarray(d["p"])
                nexit[k] = len(p)
                far = (p - centre) @ G["u"] < 0
                L = np.asarray(d["ppath"]).reshape(len(p), -1)[far]
                pk.append(np.column_stack([np.full(far.sum(), k), p[far], L, np.asarray(d["w0"])[far]]))
                print("  %s %s: batch %d/%d, %d exits, %d in the opposite hemisphere, %.0f s"
                      % (s, case, k + 1, nb, len(p), far.sum(), time.time() - t0), flush=True)
                del r, d, p, L, far
            out["%s_%s_flux" % (s, case)] = flux
            out["%s_%s_nexit" % (s, case)] = nexit
            out["%s_%s_pk" % (s, case)] = np.concatenate(pk)
            out["%s_%s_seconds" % (s, case)] = time.time() - t0
    out["mua"], out["nidx"] = mua, nidx
    out["sources"] = np.array(a.sources)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "transmit_%s.npz" % a.tag), **out)


def jk(vals):
    """Jackknife SE of leave-one-out values (nb, ...)."""
    nb = len(vals)
    return np.sqrt((nb - 1) / nb * ((vals - vals.mean(0)) ** 2).sum(0))


def analyse(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(os.path.join(OUT, "transmit_%s.npz" % a.tag))
    nb, batch = int(z["nb"]), float(z["batch"])
    N = nb * batch
    mua, nidx = z["mua"], z["nidx"]
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    say("Colin27 whole-head transmission, %s, %d x %.3g photons per case, tend %.0f ns, no roulette"
        % (str(z["compute"]), nb, batch, TEND * 1e9))
    say("WM isotropic mu_s %.2f 1/mm; CC = WM elements with JHU genu/body/splenium (%d elements, label 5)"
        % (float(z["mus_iso"]), int(z["ncc"])))
    sources = [str(s) for s in z["sources"]]
    node, elem, label = load_mesh()
    figw, axw = plt.subplots(len(sources), 4, figsize=(21, 4.6 * len(sources)), constrained_layout=True,
                             squeeze=False)
    for si, s in enumerate(sources):
        src, A, u, centre = z[s + "_src"], z[s + "_A"], z[s + "_u"], z["centre"]
        say()
        say("=== source %s: scalp point (world RAS) %s mm, antipode A %s mm, source-A distance %.1f mm"
            % (s, np.round(src - MESH_ORIGIN, 1), np.round(A - MESH_ORIGIN, 1), np.linalg.norm(src - A)))
        for case in ("iso", "aniso"):
            say("  %s: %.0f s, exits per photon %.5f (the rest still in the head at tend)"
                % (case, float(z["%s_%s_seconds" % (s, case)]), z["%s_%s_nexit" % (s, case)].sum() / N))
        pk = {c: z["%s_%s_pk" % (s, c)] for c in ("iso", "aniso")}
        det = {}
        for c, P in pk.items():
            b = P[:, 0].astype(int)
            p = P[:, 1:4]
            L = P[:, 4:9]
            w = P[:, 9] * np.exp(-L @ mua)
            t = (L @ nidx) / C0                                   # ns
            dA = np.linalg.norm(p - A, axis=1)
            cosang = ((p - centre) @ (-u)) / np.linalg.norm(p - centre, axis=1)
            sel = {"disc %g mm" % R: dA < R for R in DISCS}
            sel["hemisphere"] = np.ones(len(w), bool)
            det[c] = dict(b=b, w=w, L=L, t=t, sel=sel, ang=np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        for dn in det["iso"]["sel"]:
            say("  -- detector %s" % dn)
            Tb = {}
            for c in ("iso", "aniso"):
                D = det[c]
                m = D["sel"][dn]
                w, b = D["w"][m], D["b"][m]
                Tb[c] = np.bincount(b, w, minlength=nb) / batch
                T = w.sum() / N
                ess = w.sum() ** 2 / max((w ** 2).sum(), 1e-300)
                se_pp = np.sqrt(max((w ** 2).sum() / N - T ** 2, 0) / N)
                loo = (Tb[c].sum() - Tb[c]) / (nb - 1)
                if m.sum() == 0:
                    say("     %-5s: 0 packets" % c)
                    continue
                lw = np.log10(w)
                top = np.sort(w)[::-1]
                L = D["L"][m]
                Lm = (w[:, None] * L).sum(0) / w.sum()
                fwm = w[(L[:, 3] + L[:, 4]) > 0].sum() / w.sum()
                fcc = w[L[:, 4] > 0].sum() / w.sum()
                tm = (w * D["t"][m]).sum() / w.sum()
                say("     %-5s: %d packets, ESS %.1f, T = %.3e ± %.2e (jackknife; per packet %.2e, %.1f%%), "
                    "log10 w median %.1f [%.1f, %.1f], top packet %.0f%% of T, top 10 %.0f%%"
                    % (c, m.sum(), ess, T, jk(loo), se_pp, 100 * jk(loo) / T, np.median(lw), lw.min(), lw.max(),
                       100 * top[0] / w.sum(), 100 * top[:10].sum() / w.sum()))
                say("            mean time of flight %.3f ns; weighted partial paths %s mm; weight fraction "
                    "through WM %.3f, through CC %.3f"
                    % (tm, ", ".join("%s %.1f" % (nm, x) for nm, x in zip(NAMES, Lm)), fwm, fcc))
            if (Tb["iso"] > 0).all() and Tb["iso"].sum() > 0 and Tb["aniso"].sum() > 0:
                si_, sa_ = Tb["iso"].sum(), Tb["aniso"].sum()
                rat = sa_ / si_
                loo = ((sa_ - Tb["aniso"]) / (si_ - Tb["iso"]))
                se = jk(loo)
                se_ind = rat * np.hypot(jk((si_ - Tb["iso"]) / (nb - 1)) / (si_ / nb),
                                        jk((sa_ - Tb["aniso"]) / (nb - 1)) / (sa_ / nb))
                rho = np.corrcoef(Tb["iso"], Tb["aniso"])[0, 1]
                need = N * (se / rat / 0.01) ** 2
                say("     aniso/iso = %.4f ± %.4f (paired jackknife; unpaired would be ± %.4f, batch correlation "
                    "%.2f) -> %.2g photons per case for 1%% on the ratio" % (rat, se, se_ind, rho, need))
            else:
                say("     aniso/iso: not defined (batches with no detected weight)")
        # weights figure: distribution of log10 w and contribution to T, and T vs angle from A
        bins = np.linspace(-45, 0, 91)
        abins = np.arange(0, 91, 5)
        for c, col in (("iso", "black"), ("aniso", "red")):
            D = det[c]
            if len(D["w"]) == 0:
                continue
            lw = np.log10(D["w"])
            axw[si, 0].hist(lw, bins, histtype="step", color=col, label=c)
            axw[si, 1].hist(lw, bins, weights=D["w"] / N, histtype="step", color=col, label=c)
            Ta = np.bincount(np.digitize(D["ang"], abins) - 1, D["w"], minlength=len(abins))[:len(abins) - 1] / N
            ac = 0.5 * (abins[1:] + abins[:-1])
            axw[si, 2].semilogy(ac[Ta > 0], Ta[Ta > 0], "o-" if c == "iso" else "s--", color=col, label=c)
            tb = np.linspace(0, TEND * 1e9, 41)
            for dn, ls in (("hemisphere", "-"), ("disc 30 mm", ":")):
                m = D["sel"][dn]
                Tt = np.histogram(D["t"][m], tb, weights=D["w"][m] / N)[0]
                tc = 0.5 * (tb[1:] + tb[:-1])
                if (Tt > 0).any():
                    axw[si, 3].semilogy(tc[Tt > 0], Tt[Tt > 0], ls, color=col, label="%s, %s" % (c, dn))
        axw[si, 0].set_yscale("log")
        axw[si, 0].set_xlabel("log10 packet weight at exit")
        axw[si, 0].set_ylabel("packets")
        axw[si, 0].set_title("%s: opposite-hemisphere exits" % s)
        axw[si, 1].set_yscale("log")
        axw[si, 1].set_xlabel("log10 packet weight at exit")
        axw[si, 1].set_ylabel("contribution to T (per launched photon)")
        axw[si, 1].set_title("which weights make T")
        axw[si, 2].set_xlabel("angle from the antipode A seen from the head centre, deg")
        axw[si, 2].set_ylabel("T per 5° band (per launched photon)")
        axw[si, 3].set_xlabel("time of flight, ns (tend = %g ns)" % (TEND * 1e9))
        axw[si, 3].set_ylabel("T per 0.25 ns (per launched photon)")
        axw[si, 3].set_title("when the transmitted weight arrives")
        for k_ in range(4):
            axw[si, k_].legend(fontsize=8)
        fluence_fig(z, s, node, elem, label, a.tag)
    figw.savefig(os.path.join(OUT, "transmit_%s_weights.png" % a.tag), dpi=110)
    plt.close(figw)
    with open(os.path.join(OUT, "transmit_%s.txt" % a.tag), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")


def fluence_fig(z, s, node, elem, label, tag):
    import matplotlib.pyplot as plt
    nb = int(z["nb"])
    src, A = z[s + "_src"], z[s + "_A"]
    P = 0.5 * (src + A)
    ax0 = int(np.argmax(np.abs(z[s + "_u"])))                  # main axis source -> A
    other = [i for i in range(3) if i != ax0]
    # plane 1: contains the axis and z (or y if the axis is z); plane 2: axial (normal z)
    planes = []
    for nrm in other:
        a1, a2 = [i for i in range(3) if i != nrm]
        planes.append((nrm, a1, a2))
    names = "xyz"
    axn = ["x (L→R), mm", "y (P→A), mm", "z (I→S), mm"]
    loc = Locator(node, elem)
    h, ext = 1.0, 100.0
    gr = np.arange(-ext, ext + h / 2, h)
    U, V = np.meshgrid(gr, gr, indexing="xy")
    Fi, Fa = z[s + "_iso_flux"].astype(float), z[s + "_aniso_flux"].astype(float)
    fig, ax = plt.subplots(2, len(planes), figsize=(7.2 * len(planes), 12), constrained_layout=True)
    for j, (a0, a1, a2) in enumerate(planes):
        X = np.zeros((U.size, 3))
        X[:, a0] = P[a0]
        X[:, a1] = P[a1] + U.ravel()
        X[:, a2] = P[a2] + V.ravel()
        el, bary = loc(X)
        lab = np.where(el >= 0, label[np.maximum(el, 0)], 0).reshape(U.shape)
        fi = np.array([loc.interp(el, bary, f) for f in Fi])
        fa = np.array([loc.interp(el, bary, f) for f in Fa])
        mi = fi.mean(0).reshape(U.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = (fa.sum(0) / fi.sum(0) - 1).reshape(U.shape)
            jr = np.array([(fa.sum(0) - fa[i]) / (fi.sum(0) - fi[i]) - 1 for i in range(nb)])
            se = jk(jr).reshape(U.shape)
        cu, cv = P[a1] + gr - MESH_ORIGIN[a1], P[a2] + gr - MESH_ORIGIN[a2]
        good = mi > 0
        lm = np.log10(np.where(good, mi, np.nan))
        vmax = np.nanmax(lm)
        im = ax[0, j].pcolormesh(cu, cv, lm, cmap="viridis", vmin=vmax - 20, vmax=vmax, shading="auto")
        ax[0, j].contour(cu, cv, lab, levels=[0.5, 1.5, 2.5, 3.5], colors="red", linewidths=0.6)
        fig.colorbar(im, ax=ax[0, j], shrink=0.8, label="log10 fluence (iso), 1/mm² per photon")
        ok = good & (se < 0.2) & np.isfinite(rel)
        ax[1, j].set_facecolor("0.85")
        im = ax[1, j].pcolormesh(cu, cv, 100 * np.where(ok, rel, np.nan), cmap="RdBu_r", vmin=-30, vmax=30,
                                 shading="auto")
        ax[1, j].contour(cu, cv, lab, levels=[0.5, 1.5, 2.5, 3.5], colors="black", linewidths=0.6)
        sig = np.where(ok, np.abs(rel) / np.maximum(se, 1e-12), 0)
        ax[1, j].contour(cu, cv, sig, levels=[3.0], colors="0.3", linewidths=1.2, linestyles="dashed")
        fig.colorbar(im, ax=ax[1, j], shrink=0.8, label="aniso / iso − 1, %")
        for row in (0, 1):
            a = ax[row, j]
            for pt, mk in ((src, "o"), (A, "s")):
                a.plot(pt[a1] - MESH_ORIGIN[a1], pt[a2] - MESH_ORIGIN[a2], mk, ms=7,
                       mfc="black" if row else "white", mec="red" if row == 0 else "white")
            a.set_aspect("equal")
            a.set_xlabel(axn[a1])
            a.set_ylabel(axn[a2])
            a.set_title("plane %s = %.0f mm" % (names[a0], P[a0] - MESH_ORIGIN[a0]))
    fig.suptitle("Colin27, source %s (circle) -> antipode (square). Top: iso fluence, 20 decades, tissue "
                 "boundaries red.\nBottom: aniso/iso − 1 where its SE < 20%% (grey elsewhere), dashed: "
                 "|Δ| = 3 SE, boundaries black" % s)
    fig.savefig(os.path.join(OUT, "transmit_%s_%s_fluence.png" % (tag, s)), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nphoton", type=float, default=1e6)
    ap.add_argument("--batch", type=float, default=5e5)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--tag", default="test")
    ap.add_argument("--gpuid", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1648335518)
    ap.add_argument("--compute", default="cuda", choices=["cuda", "opencl"])
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        simulate(a)
    analyse(a)


if __name__ == "__main__":
    main()
